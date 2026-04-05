"""
Retrieval Augmentation MCP Tools.

Category 2: Hybrid retrieval and reranking tools.

Tools:
  - retrieval_hybrid_search : Execute hybrid search (BGE-M3 dense + BM25 sparse + RRRF)
  - retrieval_rerank        : Re-rank retrieved chunks with BGE-Reranker
  - retrieval_multi_query   : Multi-query retrieval for sub-tasks with RRRF merge
"""

from __future__ import annotations

from typing import Any, Optional

from .base import MCPTool


# ---------------------------------------------------------------------------
# Tool: Hybrid Search
# ---------------------------------------------------------------------------

class RetrievalHybridSearchTool(MCPTool):
    """
    Execute a hybrid dual-branch retrieval query.

    Combines BGE-M3 dense retrieval and BM25 sparse retrieval via
    Reciprocal Rank Fusion (RRRF), returning top-K fused results.
    """

    name = "retrieval_hybrid_search"
    description = (
        "Execute a hybrid retrieval query combining dense vector search (BGE-M3) "
        "and sparse keyword search (BM25) with Reciprocal Rank Fusion (RRRF). "
        "Returns the top-K most relevant document chunks for a financial query. "
        "Use this as the primary retrieval step for any complex question."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query string.",
            },
            "top_k": {
                "type": "integer",
                "default": 20,
                "description": "Number of results to retrieve per branch before fusion.",
            },
            "mode": {
                "type": "string",
                "enum": ["dense_only", "sparse_only", "hybrid", "rerank"],
                "default": "hybrid",
                "description": "Retrieval mode.",
            },
            "collection_name": {
                "type": "string",
                "default": "financial_reports",
                "description": "Target document collection name.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, retriever: Optional[Any] = None) -> None:
        self._retriever = retriever

    def execute(
        self,
        query: str,
        top_k: int = 20,
        mode: str = "hybrid",
        collection_name: str = "financial_reports",
    ) -> dict[str, Any]:
        """Execute hybrid retrieval and return results."""
        if self._retriever is None:
            return {
                "error": "Retriever not initialized. Set up HybridRetriever in MCP server.",
                "query": query,
                "mode": mode,
            }

        try:
            chunks = self._retriever.retrieve(query=query, mode=mode)
            return {
                "query": query,
                "mode": mode,
                "num_results": len(chunks),
                "results": [
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text[:300],
                        "source_doc": c.source_doc,
                        "page_number": c.page_number,
                        "score": round(float(c.score), 4),
                        "dense_score": round(float(getattr(c, "dense_score", 0.0)), 4),
                        "sparse_score": round(float(getattr(c, "sparse_score", 0.0)), 4),
                    }
                    for c in chunks[:top_k]
                ],
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "query": query}


# ---------------------------------------------------------------------------
# Tool: Re-rank
# ---------------------------------------------------------------------------

class RetrievalRerankTool(MCPTool):
    """
    Re-rank retrieved document chunks using BGE-Reranker v2.

    Takes a list of (query, document_text) pairs and returns them
    re-scored by cross-encoder relevance.
    """

    name = "retrieval_rerank"
    description = (
        "Re-rank a list of document chunks using the BGE-Reranker cross-encoder. "
        "Use after hybrid_search to improve precision. "
        "The reranker computes a more accurate relevance score than embedding similarity."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Original user query."},
            "documents": {
                "type": "array",
                "description": "List of document texts to re-rank.",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string"},
                        "text": {"type": "string"},
                        "source_doc": {"type": "string"},
                        "page_number": {"type": "integer"},
                    },
                },
            },
            "top_k": {
                "type": "integer",
                "default": 5,
                "description": "Return top-K re-ranked results.",
            },
        },
        "required": ["query", "documents"],
    }

    def __init__(self, reranker: Optional[Any] = None) -> None:
        self._reranker = reranker

    def execute(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Re-rank documents and return ordered results."""
        if self._reranker is None:
            return {"error": "Reranker not initialized.", "query": query}

        if not documents:
            return {"query": query, "results": [], "num_results": 0}

        texts = [d["text"] for d in documents]
        try:
            reranked = self._reranker.rerank(query, texts, top_k=len(texts))
            results = []
            for idx, score in reranked[:top_k]:
                doc = documents[idx]
                results.append(
                    {
                        "chunk_id": doc.get("chunk_id", ""),
                        "text": doc["text"][:300],
                        "source_doc": doc.get("source_doc", ""),
                        "page_number": doc.get("page_number", 0),
                        "rerank_score": round(float(score), 4),
                    }
                )
            return {
                "query": query,
                "num_results": len(results),
                "results": results,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "query": query}


# ---------------------------------------------------------------------------
# Tool: Multi-Query Retrieval
# ---------------------------------------------------------------------------

class RetrievalMultiQueryTool(MCPTool):
    """
    Retrieve documents for multiple sub-queries and merge via RRRF.

    Used in the fast-path when a complex question has been decomposed
    into N sub-tasks, each with its own retrieval query. All results
    are pooled and re-ranked together.
    """

    name = "retrieval_multi_query"
    description = (
        "Execute retrieval for multiple sub-queries simultaneously and merge "
        "results using Reciprocal Rank Fusion (RRRF). "
        "Use when a complex question has been decomposed into independent sub-tasks "
        "(e.g., one per modality or per time period). "
        "Each sub-task query retrieves independently, then all results are pooled."
    )

    parameters = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "description": "List of sub-query strings.",
                "items": {"type": "string"},
            },
            "top_k_per_query": {
                "type": "integer",
                "default": 10,
                "description": "Retrieve top-K per sub-query before merging.",
            },
            "final_k": {
                "type": "integer",
                "default": 10,
                "description": "Return top-K after RRRF merge.",
            },
            "mode": {
                "type": "string",
                "enum": ["dense_only", "sparse_only", "hybrid"],
                "default": "hybrid",
            },
        },
        "required": ["queries"],
    }

    def __init__(self, retriever: Optional[Any] = None) -> None:
        self._retriever = retriever

    def execute(
        self,
        queries: list[str],
        top_k_per_query: int = 10,
        final_k: int = 10,
        mode: str = "hybrid",
    ) -> dict[str, Any]:
        """Execute multi-query retrieval with RRRF merge."""
        if self._retriever is None:
            return {"error": "Retriever not initialized.", "num_queries": len(queries)}

        if not queries:
            return {"results": [], "num_results": 0}

        try:
            # Flatten all results for return
            all_chunks = self._retriever.retrieve_multi_query(
                queries=queries, mode=mode
            )
            results = [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text[:300],
                    "source_doc": c.source_doc,
                    "page_number": c.page_number,
                    "score": round(float(getattr(c, "score", 0.0)), 4),
                }
                for c in all_chunks[:final_k]
            ]
            return {
                "num_queries": len(queries),
                "queries": queries,
                "num_results": len(results),
                "results": results,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "queries": queries}
