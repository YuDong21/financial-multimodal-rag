"""
graph: LangGraph workflow orchestration for financial multimodal RAG.

Fast path (COMPLEX_FAST):
  input → router → task_decomposition → retrieval_planning
  → fast_retrieval → fast_rerank → evidence_check
  → [loop | mcp_tool_call] → generation → verification → end

Slow path (COMPLEX_SLOW):
  input → router → slow_retrieval → slow_rerank → generation → verification → end

Simple path (SIMPLE):
  input → router → direct_generation → end

GraphState fields (graph/state.py):
  - question, query_rewritten, original_question
  - route (SIMPLE/SLOW/FAST), routing_reasoning, task_type
  - sub_tasks, retrieval_queries, retrieval_strategy
  - slow/fast_retrieved_docs, slow/fast_reranked_docs
  - evidence_snippets, evidence_score, evidence_reasoning
  - fallback_triggered, fallback_count
  - candidate_tools, tool_call_results
  - answer_draft, answer_final, citations, answer_groundedness_score
  - error_message, node_errors
  - short_term_context, memory_summary, total_tokens_used
"""

from .workflow import (
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
    ConversationTurn,
)

__all__ = [
    # Workflow
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
    "VerificationNode",
    # State types
    "GraphState",
    "Route",
    "RetrievalStrategy",
    "TaskType",
    "FallbackReason",
    "RetrievedDoc",
    "SubTask",
    "ToolCandidate",
    "Citation",
    "ConversationTurn",
]
