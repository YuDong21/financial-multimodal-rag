"""
retrieval: Hybrid retrieval engine for financial multimodal RAG.
Exposes BGE-M3 dense retrieval, BM25 sparse retrieval, and BGE-Reranker fusion.
"""

from .hybrid_retriever import (
    RetrievedChunk,
    HybridRetriever,
    BM25SparseRetriever,
    BGEEmbeddingRetriever,
    BGEReranker,
)

__all__ = [
    "RetrievedChunk",
    "HybridRetriever",
    "BM25SparseRetriever",
    "BGEEmbeddingRetriever",
    "BGEReranker",
]
