"""
LangGraph GraphState — Comprehensive shared state for the financial multimodal RAG workflow.

The GraphState carries all intermediate and persistent data across every node
in the graph. It is the single source of truth for the entire execution context.

Why GraphState?
---------------
LangGraph is a stateful, branching graph — not a simple linear chain. Without a
unified state object, every node would need to pass parameters explicitly, leading
to: (a) parameter explosion as the number of nodes grows, and (b) state
inconsistency between nodes that maintain their own partial state.

GraphState solves this by providing a shared, mutable context that every node
reads from and writes to. Nodes are pure functions: state_in → state_out.

Critical Fields
---------------
question           : Raw user query (immutable anchor)
query_rewritten    : Normalized / rephrased version after query processing
route              : Routing decision (SIMPLE / SLOW / FAST)
task_type          : The type of task being performed
retrieval_queries  : The actual query/queries used for retrieval
retrieved_docs     : Raw retrieved evidence (before reranking)
reranked_docs      : Post-reranking evidence
evidence_snippets  : The specific text spans used as evidence
evidence_score     : Sufficiency score from verification
fallback_triggered : Whether the slow/fast path fell back to MCP tools
candidate_tools   : Tool options considered for this step
tool_call_results  : Actual tool execution results
answer_draft       : Pre-verification answer draft
answer_final       : Verified, citation-aware final answer
citations          : All citation metadata for the final answer
short_term_context : Recent conversation turns (rolling window)
memory_summary     : Compressed summary of older conversation history
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Optional

import operator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Route(str, Enum):
    """Routing decisions from the SemanticRouter node."""

    SIMPLE = "simple"        # Single-hop, bypass retrieval
    COMPLEX_SLOW = "slow"    # Multi-hop, full retrieval, no planning loop
    COMPLEX_FAST = "fast"    # Complex, task decomposition + evidence verification loop


class RetrievalStrategy(str, Enum):
    """Modality routing for retrieval planning."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    ALL = "all"


class TaskType(str, Enum):
    """The type of task being performed."""

    FACTUAL_QUERY = "factual_query"           # Simple fact lookup
    COMPARATIVE_ANALYSIS = "comparative"     # Cross-company or cross-period comparison
    TREND_ANALYSIS = "trend"                 # Time-series analysis
    RISK_ASSESSMENT = "risk"                 # Risk factor identification
    FINANCIAL_CALC = "financial_calc"         # Ratio / formula calculation
    CROSS_TABLE_SYNTHESIS = "cross_table"     # Synthesizing info across multiple tables
    UNKNOWN = "unknown"


class FallbackReason(str, Enum):
    """Why a fallback to MCP tools was triggered."""

    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    NO_RETRIEVAL_RESULTS = "no_retrieval_results"
    TABLE_MODALITY_MISSING = "table_modality_missing"
    CHART_MODALITY_MISSING = "chart_modality_missing"
    CROSS_PAGE_TABLE = "cross_page_table"
    CALCULATION_REQUIRED = "calculation_required"
    OCR_CORRECTION_NEEDED = "ocr_correction_needed"
    NONE = "none"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievedDoc:
    """
    A retrieved document chunk with source metadata for citation.

    Attributes
    ----------
    chunk_id, text, source_doc, page_number — identification
    retrieval_strategy — which modality this chunk was retrieved from
    dense_score, sparse_score, rerank_score — per-branch scores (for debugging)
    """

    chunk_id: str
    text: str
    source_doc: str
    page_number: int
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.ALL
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class SubTask:
    """
    A sub-task produced by TaskDecompositionNode.

    Attributes
    ----------
    task_id       : Unique identifier
    description   : What information this sub-task needs
    strategy      : RetrievalStrategy (text / table / figure / all)
    key_terms     : Search keywords for this sub-task
    status        : "pending" | "retrieved" | "failed"
    retrieved_ids  : chunk_ids of retrieved docs for this sub-task
    """

    task_id: str
    description: str
    strategy: RetrievalStrategy
    key_terms: list[str] = field(default_factory=list)
    status: str = "pending"
    retrieved_ids: list[str] = field(default_factory=list)


@dataclass
class ToolCandidate:
    """
    A tool that was considered for execution but may or may not have been called.
    """

    tool_name: str
    reason: str  # Why this tool was considered
    selected: bool  # Whether it was actually called
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None


@dataclass
class Citation:
    """
    A citation embedded in the final answer.

    Attributes
    ----------
    source_doc    : Document name
    page_number   : Page number
    text          : The cited text span
    chunk_id      : Source chunk ID
    in_answer     : The text in the answer that uses this citation
    verified      : True if verified against retrieved evidence
    """

    source_doc: str
    page_number: int
    text: str
    chunk_id: str = ""
    in_answer: str = ""
    verified: bool = False


@dataclass
class ConversationTurn:
    """
    A single conversational exchange unit for short-term context.
    """

    role: str  # "user" or "assistant"
    content: str
    token_count: int = 0


# ---------------------------------------------------------------------------
# Graph State (TypedDict)
# ---------------------------------------------------------------------------

class GraphState(dict):
    """
    Shared state carried through the entire LangGraph pipeline.

    All fields are optional — each node reads and writes only the fields
    it owns. This ensures nodes remain decoupled.

    Usage in a node:
        def my_node(state: GraphState) -> GraphState:
            question = state["question"]           # Read
            state["my_new_field"] = "computed"    # Write
            return state
    """

    # ─── Input ──────────────────────────────────────────────────────────────
    question: Annotated[str, operator.add]  # Raw user question (accumulated if multi-turn)
    session_id: str                         # Session identifier

    # ─── Query Processing ─────────────────────────────────────────────────
    query_rewritten: Optional[str]          # Normalized / cleaned query version
    original_question: Optional[str]        # Saved for reference

    # ─── Routing ──────────────────────────────────────────────────────────
    route: Optional[Route]                  # Routing decision
    routing_reasoning: Optional[str]        # LLM's reasoning for the route
    task_type: Optional[TaskType]           # Classified task type

    # ─── Task Decomposition (Fast Path) ───────────────────────────────────
    sub_tasks: list[SubTask]                # Decomposed sub-tasks
    current_task_index: int                  # Which sub-task is being processed

    # ─── Retrieval Queries ─────────────────────────────────────────────────
    retrieval_queries: list[str]            # The actual query strings used for retrieval
    retrieval_strategy: RetrievalStrategy    # Dominant retrieval strategy

    # ─── Slow Path ─────────────────────────────────────────────────────────
    slow_retrieved_docs: list[RetrievedDoc]
    slow_reranked_docs: list[RetrievedDoc]

    # ─── Fast Path ─────────────────────────────────────────────────────────
    fast_retrieved_docs: list[RetrievedDoc]
    fast_reranked_docs: list[RetrievedDoc]
    fast_retrieval_attempts: int             # Loop counter for circuit breaker

    # ─── Evidence ──────────────────────────────────────────────────────────
    evidence_snippets: list[str]            # Specific text spans used as evidence
    evidence_score: Optional[float]          # Sufficiency score (0.0–1.0)
    evidence_reasoning: Optional[str]        # LLM reasoning for sufficiency

    # ─── Fallback ──────────────────────────────────────────────────────────
    fallback_triggered: FallbackReason       # Why fallback was triggered
    fallback_count: int                     # Number of fallback retries

    # ─── MCP Tools ─────────────────────────────────────────────────────────
    candidate_tools: list[ToolCandidate]    # Tools considered at this step
    tool_call_results: list[ToolCandidate]  # Actually executed tool calls

    # ─── Answer ────────────────────────────────────────────────────────────
    answer_draft: Optional[str]             # Pre-verification draft
    answer_final: Optional[str]             # Verified final answer
    citations: list[Citation]               # Citation metadata for the answer
    answer_groundedness_score: Optional[float]  # Score from verification

    # ─── Error ─────────────────────────────────────────────────────────────
    error_message: Optional[str]
    node_errors: dict[str, str]             # Per-node error tracking

    # ─── Memory ────────────────────────────────────────────────────────────
    short_term_context: list[ConversationTurn]  # Recent turns (rolling window)
    memory_summary: Optional[str]               # Compressed summary of older history
    total_tokens_used: int                     # Running token count for budget management
