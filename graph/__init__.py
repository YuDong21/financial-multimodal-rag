"""
graph: LangGraph workflow orchestration for financial multimodal RAG.

Token Budget Architecture
------------------------
input → truncation (TokenBudgetManager) → semantic_router → ...

TruncationNode (memory.context_manager.TruncationNode):
  - Runs BEFORE semantic_router on every request
  - Checks: total_tokens_in_context >= budget_threshold?
  - If YES → sliding window (keep last K turns verbatim)
            older turns → Qwen2.5-1.5B semantic compression → memory_summary
  - Writes back to GraphState: short_term_context, memory_summary, total_tokens_in_context

GraphState as Global Parameter
------------------------------
In LangGraph, the GraphState dict is the SINGLE source of truth.
It is passed by reference (not copied) to every node.
All nodes read from and write to the SAME dict — writes in one node
are immediately visible to all downstream nodes within a session.

The budget_manager (TokenBudgetManager) is a shared singleton on the
workflow object. TruncationNode and DirectGenerationNode both hold
references to the SAME budget_manager instance.
"""

from .workflow import (
    FinancialRAGWorkflow,
    InputNode,
    TruncationNode,
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
    VerificationNode,
)
from .state import (
    GraphState,
    Route,
    RetrievalStrategy,
    TaskType,
    FallbackReason,
    RetrievedDoc,
    SubTask,
    ToolCandidate,
    Citation,
)

__all__ = [
    # Workflow
    "FinancialRAGWorkflow",
    "InputNode",
    "TruncationNode",
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
    "VerificationNode",
    # State
    "GraphState",
    "Route",
    "RetrievalStrategy",
    "TaskType",
    "FallbackReason",
    "RetrievedDoc",
    "SubTask",
    "ToolCandidate",
    "Citation",
]
