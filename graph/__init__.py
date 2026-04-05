"""
graph: LangGraph workflow orchestration for financial multimodal RAG.

Fast path (complex):  query → input → router → task_decomp → retrieval_plan
                      → fast_retrieval → fast_rerank → evidence_check
                      → [loop | mcp_tool] → generation

Slow path (complex):   query → input → router → slow_retrieval → slow_rerank
                      → generation

Simple path:           query → input → router → direct_generation
"""

from .workflow import (
    GraphState,
    Route,
    RetrievalStrategy,
    RetrievedDoc,
    ToolCall,
    SubTask,
    FinancialRAGWorkflow,
    InputNode,
    SemanticRouterNode,
    DirectGenerationNode,
    TaskDecompositionNode,
    RetrievalPlanningNode,
    SlowRetrievalNode,
    SlowRerankNode,
    FastRetrievalNode,
    FastRerankNode,
    EvidenceSufficiencyNode,
    MCPToolNode,
    GenerationNode,
    MAX_EVIDENCE_RETRIES,
)

__all__ = [
    "GraphState",
    "Route",
    "RetrievalStrategy",
    "RetrievedDoc",
    "ToolCall",
    "SubTask",
    "FinancialRAGWorkflow",
    "InputNode",
    "SemanticRouterNode",
    "DirectGenerationNode",
    "TaskDecompositionNode",
    "RetrievalPlanningNode",
    "SlowRetrievalNode",
    "SlowRerankNode",
    "FastRetrievalNode",
    "FastRerankNode",
    "EvidenceSufficiencyNode",
    "MCPToolNode",
    "GenerationNode",
    "MAX_EVIDENCE_RETRIES",
]
