"""
Hybrid Retriever — BGE-M3 + BM25 + BGE-Reranker with RRRF Fusion.

Implements a three-stage retrieval pipeline:
  Stage 1  Dense retrieval   via BAAI/bge-m3 (multi-lingual, late-interaction)
  Stage 2  Sparse retrieval  via rank_bm25 with a financial-text analyzer
  Stage 3  RRRF fusion       combines dense + sparse ranks → cross-encoder rerank

The main entry point is :class:`HybridRetriever`.

Usage
-----
    >>> retriever = HybridRetriever(
    ...     dense_model="BAAI/bge-m3",
    ...     sparse_model="bm25",
    ...     reranker_model="BAAI/bge-reranker-v2-m3",
    ...     device="cuda",
    ...     top_k=20,
    ...     final_k=5,
    ... )
    >>> chunks = retriever.retrieve("What was Apple's revenue growth in FY2024?")
    >>> for c in chunks:
    ...     print(c.text[:80], c.score)
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagEmbedding, FlagReranker
from rank_bm25 import BM25Plus

# Type alias for retrieval mode
RetrievalMode = Literal["dense_only", "sparse_only", "hybrid", "rerank"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """
    A retrieved document chunk with its fused retrieval scores.

    Attributes
    ----------
    chunk_id : str
        Unique identifier for this chunk.
    text : str
        The actual text content of the chunk.
    source_doc : str
        Name of the source document (e.g. "annual_report_2024.pdf").
    page_number : int
        Page number where this chunk originates.
    token_count : int
        Approximate token count (for context window management).
    score : float
        The final fused + reranked score (higher = better).
    dense_score : float, internal
        Raw cosine similarity from BGE-M3 dense retrieval.
    sparse_score : float, internal
        Raw BM25 score.
    rank_dense : int, internal
        Dense retrieval rank (1-indexed).
    rank_sparse : int, internal
        BM25 retrieval rank (1-indexed).
    """

    chunk_id: str
    text: str
    source_doc: str
    page_number: int
    token_count: int
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rank_dense: int = 0
    rank_sparse: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_doc": self.source_doc,
            "page_number": self.page_number,
            "token_count": self.token_count,
            "score": round(self.score, 4),
        }


# ---------------------------------------------------------------------------
# RRRF Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    results_by_system: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRRF) for combining multiple rank lists.

    RRRF is rank-based (not score-based) making it robust to calibration
    differences between retrieval systems. It does not require learned weights.

    Parameters
    ----------
    results_by_system : list of ranked lists
        Each inner list contains (doc_id, score) tuples ordered by decreasing relevance.
    k : float, default 60
        RRF damping parameter. Higher values give more weight to lower ranks.

    Returns
    -------
    list of (doc_id, fused_score) sorted by decreasing fused score.

    Reference
    ---------
    RRF: "Reciprocal Rank Fusion for Consolidating Information from Multiple
    IR Systems" (Webb et al., 2006)
    """
    scores: dict[str, float = {}

    for results in results_by_system:
        for rank, (doc_id, _) in enumerate(results, start=1):
            doc_id = str(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs


# ---------------------------------------------------------------------------
# Dense Retriever — BGE-M3
# ---------------------------------------------------------------------------

class BGEEmbeddingRetriever:
    """
    Dense retrieval using BGE-M3 with ColBERT-style late interaction.

    BGE-M3 supports 100+ languages and achieves SOTA on multilingual benchmarks.
    It is ideal for financial documents with mixed Chinese-English terminology.

    Parameters
    ----------
    model_name_or_path : str, default "BAAI/bge-m3"
    device : str, optional,  "cuda" or "cpu"
    batch_size : int, default 32
    max_length : int, default 1024
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 1024,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self.model = BGEM3FlagEmbedding(
            model_name_or_path=model_name_or_path,
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": batch_size, "max_length": max_length},
            use_fp16=(device == "cuda"),
        )

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Encode a list of query strings into dense vectors."""
        return self.model.encode(queries)["dense_vecs"]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Encode a list of document texts into dense vectors."""
        return self.model.encode(texts)["dense_vecs"]

    def retrieve(
        self, query: str, document_texts: list[str], top_k: int = 20
    ) -> list[tuple[int, float]]:
        """
        Retrieve top_k documents for a single query.

        Returns
        -------
        list of (doc_index, cosine_similarity) sorted descending.
        """
        q_vec = self.embed_queries([query])[0]
        d_vecs = self.embed_documents(document_texts)

        # Cosine similarity (normalize → dot product)
        similarities = np.dot(d_vecs, q_vec).tolist()
        ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Sparse Retriever — BM25
# ---------------------------------------------------------------------------

class BM25SparseRetriever:
    """
    Sparse bag-of-words retrieval using BM25Plus (BM25 with document length
    normalisation independence).

    A custom tokenizer is applied to handle financial terminology
    (splitting CamelCase, preserving numeric tokens like "FY2024", etc.).
    """

    def __init__(
        self,
        token_min_len: int = 2,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.token_min_len = token_min_len
        self.k1 = k1
        self.b = b
        self._bm25: Optional[BM25Plus] = None
        self._tokenized_corpus: list[list[str]] = []
        self._documents: list[str] = []

    def index(self, documents: list[str]) -> None:
        """
        Build the BM25 index from a list of documents.

        Parameters
        ----------
        documents : list of str
            Each element is a full text of one document/chunk.
        """
        self._documents = documents
        self._tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self._bm25 = BM25Plus(
            self._tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )

    def retrieve(
        self, query: str, top_k: int = 20
    ) -> list[tuple[int, float]]:
        """
        Retrieve top_k documents for a query.

        Returns
        -------
        list of (doc_index, bm25_score) sorted descending.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call index() first.")
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize financial text.

        Splits on whitespace and punctuation, keeps CamelCase splits,
        lowercases, and filters short tokens.
        """
        import re

        # Split CamelCase, preserve numbers and financial symbols
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        tokens = re.findall(r"[\w\.]+", text.lower())
        return [t for t in tokens if len(t) >= self.token_min_len]


# ---------------------------------------------------------------------------
# Reranker — BGE-Reranker v2
# ---------------------------------------------------------------------------

class BGEReranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

    Re-scores (query, document) pairs with a cross-encoder for
    precise contextual relevance ranking.

    Parameters
    ----------
    model_name_or_path : str, default "BAAI/bge-reranker-v2-m3"
    device : str, optional
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = FlagReranker(
            model_name_or_path,
            use_fp16=(device == "cuda"),
            device=device,
        )

    def rerank(
        self, query: str, documents: list[str], top_k: int = 5
    ) -> list[tuple[int, float]]:
        """
        Re-rank a list of documents for a given query.

        Parameters
        ----------
        query : str
        documents : list of str
        top_k : int, default 5

        Returns
        -------
        list of (doc_index, reranker_score) sorted descending.
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.model.compute_score(pairs)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Main Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Hybrid retrieval pipeline combining BGE-M3 (dense), BM25 (sparse),
    RRRF fusion, and BGE-Reranker for precision re-ranking.

    Parameters
    ----------
    dense_model : str, default "BAAI/bge-m3"
        HuggingFace model name for BGE-M3 dense embeddings.
    sparse_model : {"bm25"}, default "bm25"
        Sparse retrieval backend.
    reranker_model : str, default "BAAI/bge-reranker-v2-m3"
        HuggingFace model name for the cross-encoder reranker.
    device : str, optional
        "cuda" or "cpu". Auto-detected if omitted.
    top_k : int, default 20
        Number of documents to retrieve per branch (dense + sparse)
        before fusion and reranking.
    final_k : int, default 5
        Number of documents to return after fusion and reranking.
    rrrf_k : float, default 60.0
        RRRF damping parameter.
    collection : Optional[list[dict]], default None
        Pre-loaded document collection. Each dict must have keys:
        "chunk_id", "text", "source_doc", "page_number", "token_count".

    Examples
    --------
    >>> retriever = HybridRetriever()
    >>> retriever.load_collection([
    ...     {"chunk_id": "c1", "text": "Apple revenue grew 7%", "source_doc": "a.pdf", "page_number": 5, "token_count": 20},
    ... ])
    >>> chunks = retriever.retrieve("Apple revenue FY2024")
    """

    def __init__(
        self,
        dense_model: str = "BAAI/bge-m3",
        sparse_model: Literal["bm25"] = "bm25",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        top_k: int = 20,
        final_k: int = 5,
        rrrf_k: float = 60.0,
        collection: Optional[list[dict]] = None,
    ) -> None:
        self.top_k = top_k
        self.final_k = final_k
        self.rrrf_k = rrrf_k

        self.dense_retriever = BGEEmbeddingRetriever(
            model_name_or_path=dense_model,
            device=device,
        )
        self.sparse_retriever = BM25SparseRetriever()
        self.reranker = BGEReranker(
            model_name_or_path=reranker_model,
            device=device,
        )

        self._collection: list[dict] = []
        if collection is not None:
            self.load_collection(collection)

    def load_collection(self, collection: list[dict]) -> None:
        """
        Load and index a document collection for retrieval.

        Parameters
        ----------
        collection : list of dict
            Each dict must contain:
              - chunk_id: str
              - text: str
              - source_doc: str
              - page_number: int
              - token_count: int
        """
        self._collection = collection
        texts = [doc["text"] for doc in collection]
        self.sparse_retriever.index(texts)

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode = "rerank",
    ) -> list[RetrievedChunk]:
        """
        Retrieve and rank document chunks for a query.

        Parameters
        ----------
        query : str
            Natural language query.
        mode : RetrievalMode, default "rerank"
            Retrieval strategy:
              - "dense_only"    : BGE-M3 dense retrieval only
              - "sparse_only"   : BM25 sparse retrieval only
              - "hybrid"        : RRRF fusion of dense + sparse (no rerank)
              - "rerank"        : hybrid → BGE-Reranker (default, recommended)

        Returns
        -------
        list of RetrievedChunk sorted by descending relevance score.
        """
        if not self._collection:
            raise RuntimeError(
                "No collection loaded. Call load_collection() first."
            )

        texts = [doc["text"] for doc in self._collection]

        # Stage 1 & 2 — parallel dense + sparse retrieval
        dense_results = self.dense_retriever.retrieve(query, texts, top_k=self.top_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=self.top_k)

        # Build ranked lists for RRRF
        if mode == "dense_only":
            fused = [(idx, score) for idx, score in dense_results]
        elif mode == "sparse_only":
            fused = [(idx, score) for idx, score in sparse_results]
        else:
            # Pad shorter list to match lengths for RRRF
            dense_ranked = [(str(idx), score) for idx, score in dense_results]
            sparse_ranked = [(str(idx), score) for idx, score in sparse_results]
            fused = reciprocal_rank_fusion(
                [dense_ranked, sparse_ranked], k=self.rrrf_k
            )

        # Extract texts for reranking (or return directly if no rerank)
        fused_indices = [int(doc_id) for doc_id, _ in fused[: self.final_k]]
        fused_texts = [texts[i] for i in fused_indices]
        fused_scores = {int(doc_id): score for doc_id, score in fused}

        if mode == "rerank":
            reranked = self.reranker.rerank(
                query, fused_texts, top_k=len(fused_texts)
            )
            final_indices = [fused_indices[i] for i, _ in reranked]
            final_scores = reranked
        else:
            final_indices = fused_indices
            final_scores = [(i, fused_scores[i]) for i in final_indices]

        # Build RetrievedChunk objects
        chunks: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(final_scores, start=1):
            doc = self._collection[idx]
            # Track individual branch ranks for debugging
            dense_rank = next((r + 1 for r, (i, _) in enumerate(dense_results) if i == idx), 0)
            sparse_rank = next((r + 1 for r, (i, _) in enumerate(sparse_results) if i == idx), 0)

            chunks.append(
                RetrievedChunk(
                    chunk_id=doc["chunk_id"],
                    text=doc["text"],
                    source_doc=doc["source_doc"],
                    page_number=doc["page_number"],
                    token_count=doc["token_count"],
                    score=score,
                    dense_score=dense_results[idx][1] if idx < len(dense_results) else 0.0,
                    sparse_score=sparse_results[idx][1] if idx < len(sparse_results) else 0.0,
                    rank_dense=dense_rank,
                    rank_sparse=sparse_rank,
                )
            )

        return chunks

    def retrieve_multi_query(
        self, queries: list[str], mode: RetrievalMode = "rerank"
    ) -> list[RetrievedChunk]:
        """
        Retrieve documents for multiple queries (e.g. sub-tasks from a
        decomposed complex question) and merge results via RRRF.

        Parameters
        ----------
        queries : list of str
        mode : RetrievalMode

        Returns
        -------
        Merged list of RetrievedChunk.
        """
        all_results: list[list[tuple[int, float]]] = []
        for q in queries:
            chunks = self.retrieve(q, mode=mode)
            all_results.append(
                [(int(c.chunk_id.split("_")[-1]), c.score) for c in chunks]
            )

        # Flatten chunk pool
        chunk_pool: dict[int, RetrievedChunk] = {}
        for q in queries:
            for c in self.retrieve(q, mode="hybrid"):
                idx = int(c.chunk_id.split("_")[-1])
                if idx not in chunk_pool:
                    chunk_pool[idx] = c

        # RRRF merge across queries
        str_results = [
            [(str(idx), score) for idx, score in results]
            for results in all_results
        ]
        fused = reciprocal_rank_fusion(str_results, k=self.rrrf_k)
        top_indices = [int(doc_id) for doc_id, _ in fused[: self.final_k]]

        return [chunk_pool[i] for i in top_indices if i in chunk_pool]
