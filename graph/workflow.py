"""
LangGraph Workflow for Financial Multimodal RAG.

Implements a state machine with two distinct complex paths sharing the same entry:

  ┌─────────────────────────────────────────────────────────────────┐
  │                         QUERY INPUT                             │
  └─────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │              SEMANTIC ROUTER  (Qwen3-0.6B)                     │
  │  simple  →  DIRECT_GENERATION ──────────────────────────────┐   │
  │  complex (slow) → SLOW_RETRIEVAL ──────────────────────┐   │   │
  │  complex (fast) → TASK_DECOMPOSITION ──────────────┐   │   │   │
  └──────────────────────────────────────────────────┘   │   │   │
      │                                               │   │   │   │
      │ slow_complex                                  │   │   │   │
      │                                               ▼   ▼   │   │
      │                              ┌─────────────────────────┤   │
      │                              │  SLOW_RETRIEVAL (hybrid) │   │
      │                              │  BGE-M3 + BM25 + RRRF    │   │
      │                              └────────────┬────────────┘   │
      │                                           │                │
      │                                           ▼                │
      │                              ┌─────────────────────────┤   │
      │                              │  SLOW_RERANK            │   │
      │                              │  BGE-Reranker v2        │   │
      │                              └────────────┬────────────┘   │
      │                                           │                │
      │                                           ▼                │
      │                                    GENERATION              │
      │                                         │                  │
      ├─────────────────────────────────────────┼──────────────────┤
      │ fast_complex                            │                  │
      │                                         ▼                  │
      │                         ┌─────────────────────────┐          │
      │                         │  TASK_DECOMPOSITION      │          │
      │                         │  (sub-task planning)     │          │
      │                         └────────────┬────────────┘          │
      │                                      │                       │
      │                                      ▼                       │
      │                         ┌─────────────────────────┐          │
      │                         │  RETRIEVAL_PLANNING      │          │
      │                         │  (route per modality)    │          │
      │                         └────────────┬────────────┘          │
      │                                      │                       │
      │                                      ▼                       │
      │                         ┌─────────────────────────┐          │
      │                         │  FAST_RETRIEVAL         │          │
      │                         │  (multi-query, hybrid)  │          │
      │                         └────────────┬────────────┘          │
      │                                      │                       │
      │                                      ▼                       │
      │                         ┌─────────────────────────┐          │
      │                         │  EVIDENCE_SUFFICIENCY   │          │
      │                         └────────────┬────────────┘          │
      │           ┌─────────────────────────┼───────────────────┐  │
      │           │ insufficient            │ sufficient         │  │
      │           ▼                         ▼                     │  │
      │  ┌───────────────┐         ┌─────────────────┐            │  │
      │  │FAST_RETRIEVAL│         │  MCP_TOOL_CALL  │            │  │
      │  │  (loop back)  │         │  (DeepDoc/Calc) │            │  │
      │  └───────┬───────┘         └────────┬────────┘            │  │
      │          │                          │                     │  │
      │          └──────────────────────────┴─────────────────────┘  │
      │                               │                               │
      │                               ▼                               │
      │                        GENERATION  ──────────────────────── END
      └─────────────────────────────────────────────────────────────┘

Key Design Decisions
--------------------
1. **Separate slow/fast retrieval nodes** — Each complex path has its own
   isolated retrieval + rerank chain, preventing state pollution between paths.
2. **Conditional edges** — Every routing decision is an explicit function,
   fully inspectable and loggable.
3. **Evidence loop with circuit breaker** — Fast path loops FAST_RETRIEVAL →
   EVIDENCE_CHECK up to MAX_EVIDENCE_RETRIES times before forcing MCP tool use.
4. **Citation-aware generation** — Single shared GenerationNode for both paths.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Literal, Optional, TypedDict

import operator
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_EVIDENCE_RETRIES: int = 2  # Circuit breaker for fast-path evidence loop


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Route(str, Enum):
    """Possible routes from the SemanticRouter node."""

    SIMPLE = "simple"
    COMPLEX_SLOW = "slow"
    COMPLEX_FAST = "fast"


class RetrievalStrategy(str, Enum):
    """Modality routing for retrieval planning."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    ALL = "all"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievedDoc:
    """A retrieved document chunk with source metadata for citation."""

    chunk_id: str
    text: str
    source_doc: str
    page_number: int
    retrieval_strategy: RetrievalStrategy

    # Optional per-node scores (used for logging/debugging)
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class ToolCall:
    """A planned or executed MCP tool call."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any = None
    error: Optional[str] = None


@dataclass
class SubTask:
    """A single sub-task produced by task decomposition."""

    task_id: str
    description: str
    strategy: RetrievalStrategy
    key_terms: list[str]


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """
    Shared state carried through the LangGraph pipeline.
    All fields are optional — each node reads/writes only what it needs.
    """

    # --- Input ---
    question: Annotated[str, operator.add]
    session_id: str

    # --- Routing ---
    route: Optional[Route]
    routing_reasoning: Optional[str]

    # --- Task Decomposition ---
    sub_tasks: list[SubTask]

    # --- Retrieval Planning ---
    retrieval_strategy: RetrievalStrategy

    # --- Slow Path ---
    slow_retrieved_docs: list[RetrievedDoc]
    slow_reranked_docs: list[RetrievedDoc]

    # --- Fast Path ---
    fast_retrieved_docs: list[RetrievedDoc]
    fast_evidence_sufficient: bool
    fast_evidence_reasoning: Optional[str]
    fast_retrieval_attempts: int  # Tracks loop iterations for circuit breaker
    fast_reranked_docs: list[RetrievedDoc]

    # --- MCP ---
    tool_calls: list[ToolCall]

    # --- Generation ---
    answer: Optional[str]
    citations: list[dict[str, Any]]

    # --- Memory ---
    conversation_history: list[dict[str, str]]

    # --- Error ---
    error_message: Optional[str]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def deduplicate_docs(docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
    """Remove duplicate chunks by chunk_id, preserving order."""
    seen: set[str] = set()
    unique: list[RetrievedDoc] = []
    for d in docs:
        if d.chunk_id not in seen:
            seen.add(d.chunk_id)
            unique.append(d)
    return unique


def _make_state(**kwargs: Any) -> GraphState:
    """Create a GraphState dict with defaults."""
    return GraphState(**kwargs)


# ---------------------------------------------------------------------------
# Node: Input Initialization
# ---------------------------------------------------------------------------

class InputNode:
    """
    Initializes the graph state from a raw user question.

    This is the canonical entry point — every query first passes through here
    before reaching the semantic router.
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or str(uuid.uuid4())

    def run(self, question: str) -> GraphState:
        """
        Initialize and return the starting state.

        Parameters
        ----------
        question : str

        Returns
        -------
        GraphState with ``question`` and ``session_id`` set.
        """
        return _make_state(
            question=question,
            session_id=self.session_id,
            route=None,
            routing_reasoning=None,
            sub_tasks=[],
            retrieval_strategy=RetrievalStrategy.ALL,
            slow_retrieved_docs=[],
            slow_reranked_docs=[],
            fast_retrieved_docs=[],
            fast_evidence_sufficient=False,
            fast_evidence_reasoning=None,
            fast_retrieval_attempts=0,
            fast_reranked_docs=[],
            tool_calls=[],
            answer=None,
            citations=[],
            conversation_history=[],
            error_message=None,
        )


# ---------------------------------------------------------------------------
# Node: Semantic Router
# ---------------------------------------------------------------------------

class SemanticRouterNode:
    """
    Classifies the incoming question into ``simple``, ``slow``, or ``fast``.

    Uses Qwen3-0.6B (lightweight, low-latency) to decide the execution path:

      simple  — Single-hop factual question. Bypass retrieval entirely.
      slow    — Multi-hop or cross-temporal question requiring full retrieval
                pipeline but no iterative planning/verification.
      fast    — Complex question requiring task decomposition, retrieval
                planning, and evidence verification loop.
    """

    ROUTER_PROMPT = """You are a semantic query router for a financial RAG system.
Given a user's question about financial reports, classify it into exactly ONE of
three categories:

  simple  — A direct factual question answerable from a single document chunk.
             No sub-task planning needed. Examples:
             "What was Apple's revenue in FY2024?", "What does page 5 say about net profit?"

  slow    — A multi-hop, comparative, or cross-temporal question that needs full
             retrieval and reranking, but does NOT need iterative sub-task planning
             or evidence verification loops. Examples:
             "Compare Apple's revenue trend over the past 3 years",
             "What is the average ROE across all quarters in 2024?"

  fast    — A complex question that requires decomposing into sub-tasks,
             planning retrieval strategies per modality (text/table/figure),
             and verifying evidence sufficiency before generating. Examples:
             "Summarize the key risk factors in the ESG report and calculate the
              debt-to-equity ratio across all peer companies",
             "Identify all cross-page tables about revenue and compute CAGR
              for the last 5 years"

Reply with ONLY one word: simple | slow | fast — no explanation, no punctuation."""  # noqa: E501

    def __init__(self, router: Any) -> None:
        self.router = router

    def run(self, state: GraphState) -> GraphState:
        """
        Classify the question and write ``route`` into state.
        """
        question = state["question"]
        try:
            messages = [
                {"role": "system", "content": self.ROUTER_PROMPT},
                {"role": "user", "content": question},
            ]
            response = self.router.chat(messages=messages)
            label = response.content.strip().lower()

            if "slow" in label:
                state["route"] = Route.COMPLEX_SLOW
            elif "fast" in label:
                state["route"] = Route.COMPLEX_FAST
            else:
                state["route"] = Route.SIMPLE

            state["routing_reasoning"] = response.content
        except Exception as exc:  # noqa: BLE001
            state["route"] = Route.COMPLEX_SLOW  # Fail to slow path on error
            state["routing_reasoning"] = f"Routing error: {exc}"
        return state


# ---------------------------------------------------------------------------
# Node: Direct Generation (Simple Path)
# ---------------------------------------------------------------------------

class DirectGenerationNode:
    """
    Fast-path generation for simple, single-hop questions.

    Bypasses retrieval entirely and generates directly from conversation history.
    Uses Qwen3-8B for quality. No citations (no retrieval was performed).
    """

    def __init__(self, generator: Any, memory_manager: Optional[Any] = None) -> None:
        self.generator = generator
        self.memory_manager = memory_manager

    def run(self, state: GraphState) -> GraphState:
        """
        Generate a direct answer for simple queries.
        """
        question = state["question"]
        messages = list(state.get("conversation_history", []))
        messages.append({"role": "user", "content": question})

        try:
            context = (
                self.memory_manager.get_context_for_prompt(query_tokens=500)
                if self.memory_manager
                else messages
            )
            response = self.generator.chat(messages=context)
            state["answer"] = response.content
            state["citations"] = []
        except Exception as exc:  # noqa: BLE001
            state["error_message"] = f"Direct generation failed: {exc}"
        return state


# ---------------------------------------------------------------------------
# Node: Task Decomposition (Fast Path)
# ---------------------------------------------------------------------------

class TaskDecompositionNode:
    """
    Decomposes a complex fast-path question into parallel sub-tasks.

    Each sub-task carries a ``strategy`` (text/table/figure) so that
    RetrievalPlanningNode can route each to the appropriate modality.

    Uses Qwen3-0.6B for lightweight planning.
    """

    PLANNER_PROMPT = """You are a query planner for a financial RAG system.
Given the user's complex question, break it down into N independent sub-tasks.

For each sub-task provide:
  1. task_id — A unique string identifier (e.g. "t1", "t2")
  2. description — What specific information does this sub-task need?
  3. strategy — One of: text | table | figure | all
     - Use "table" if the answer requires financial table data (revenue, ROE, etc.)
     - Use "figure" if the answer requires chart / figure caption data
     - Use "text" for qualitative descriptions (risk factors, management discussion)
     - Use "all" when no specific modality is dominant
  4. key_terms — List of important financial terms / numbers to search for

Output format (JSON array only, no markdown):
[
  {"task_id": "t1", "description": "...", "strategy": "table", "key_terms": ["revenue", "FY2024"]},
  {"task_id": "t2", "description": "...", "strategy": "text", "key_terms": ["risk factor"]}
]"""  # noqa: E501

    def __init__(self, generator: Any) -> None:
        self.generator = generator

    def run(self, state: GraphState) -> GraphState:
        """
        Decompose the question into sub-tasks.
        """
        import json

        question = state["question"]
        messages = [
            {"role": "system", "content": self.PLANNER_PROMPT},
            {"role": "user", "content": question},
        ]

        try:
            response = self.generator.chat(messages=messages)
            raw = response.content.strip()
            try:
                raw_tasks = json.loads(raw)
            except json.JSONDecodeError:
                raw_tasks = [
                    {
                        "task_id": "t0",
                        "description": question,
                        "strategy": "all",
                        "key_terms": [],
                    }
                ]

            # Normalize to SubTask dataclasses
            sub_tasks = [
                SubTask(
                    task_id=t["task_id"],
                    description=t["description"],
                    strategy=RetrievalStrategy(t.get("strategy", "all")),
                    key_terms=t.get("key_terms", []),
                )
                for t in raw_tasks
            ]
            state["sub_tasks"] = sub_tasks
        except Exception as exc:  # noqa: BLE001
            state["error_message"] = f"Task decomposition failed: {exc}"
            state["sub_tasks"] = [
                SubTask(
                    task_id="t0",
                    description=question,
                    strategy=RetrievalStrategy.ALL,
                    key_terms=[],
                )
            ]
        return state


# ---------------------------------------------------------------------------
# Node: Retrieval Planning (Fast Path)
# ---------------------------------------------------------------------------

class RetrievalPlanningNode:
    """
    Routes each sub-task to its target modality (text / table / figure / all).

    This node reads ``sub_tasks`` from state and sets per-task routing
    decisions. The actual retrieval is delegated to FastRetrievalNode which
    respects these decisions.

    Also handles the case where the overall retrieval strategy is set to a
    dominant modality based on task analysis.
    """

    def run(self, state: GraphState) -> GraphState:
        """
        Analyze sub-tasks and determine the dominant retrieval strategy.

        If most sub-tasks target the same modality, set it as the global
        retrieval strategy. Otherwise default to ALL.
        """
        sub_tasks = state.get("sub_tasks", [])
        if not sub_tasks:
            state["retrieval_strategy"] = RetrievalStrategy.ALL
            return state

        # Count modality distribution across sub-tasks
        strategy_counts: dict[RetrievalStrategy, int] = {}
        for task in sub_tasks:
            s = RetrievalStrategy(task.strategy)
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        dominant = max(strategy_counts, key=strategy_counts.get)  # type: ignore
        # If "all" is dominant, default to ALL; otherwise use dominant
        if dominant == RetrievalStrategy.ALL and len(strategy_counts) > 1:
            # Find the most common non-ALL strategy
            non_all = {k: v for k, v in strategy_counts.items() if k != RetrievalStrategy.ALL}
            if non_all:
                dominant = max(non_all, key=non_all.get)

        state["retrieval_strategy"] = dominant
        return state


# ---------------------------------------------------------------------------
# Node: Slow Retrieval (Slow Path — dual召回)
# ---------------------------------------------------------------------------

class SlowRetrievalNode:
    """
    Hybrid dual-branch retrieval for the slow complex path.

    Executes BGE-M3 dense + BM25 sparse retrieval in parallel,
    fuses ranks via RRRF, and returns top-K chunks WITHOUT a separate
    re-rank step (reranking is handled by SlowRerankNode).
    """

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def run(self, state: GraphState) -> GraphState:
        """
        Execute hybrid retrieval for the slow path.

        Uses the full question as a single query (no sub-task decomposition).
        """
        question = state["question"]
        try:
            chunks = self.retriever.retrieve(query=question, mode="hybrid")
            docs = [
                RetrievedDoc(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    source_doc=c.source_doc,
                    page_number=c.page_number,
                    retrieval_strategy=RetrievalStrategy.ALL,
                    dense_score=getattr(c, "dense_score", 0.0),
                    sparse_score=getattr(c, "sparse_score", 0.0),
                )
                for c in chunks
            ]
            state["slow_retrieved_docs"] = docs
        except Exception as exc:  # noqa: BLE001
            state["error_message"] = f"Slow retrieval failed: {exc}"
            state["slow_retrieved_docs"] = []
        return state


# ---------------------------------------------------------------------------
# Node: Slow Rerank (Slow Path)
# ---------------------------------------------------------------------------

class SlowRerankNode:
    """
    Re-ranks slow-path documents using BGE-Reranker v2.

    Takes the output of SlowRetrievalNode and re-scores with a
    cross-encoder for precision.
    """

    def __init__(self, reranker: Any) -> None:
        self.reranker = reranker

    def run(self, state: GraphState) -> GraphState:
        """
        Re-rank slow-path documents with the cross-encoder.
        """
        docs = state.get("slow_retrieved_docs", [])
        if not docs:
            state["slow_reranked_docs"] = []
            return state

        question = state["question"]
        texts = [d.text for d in docs]

        try:
            reranked_indices = self.reranker.rerank(question, texts, top_k=len(texts))
            reranked: list[RetrievedDoc] = []
            for idx, score in reranked_indices:
                doc = docs[idx]
                doc.rerank_score = float(score)
                reranked.append(doc)
            state["slow_reranked_docs"] = reranked
        except Exception as exc:  # noqa: BLE001
            state["slow_reranked_docs"] = docs  # Fallback to pre-rerank order
            state["error_message"] = f"Slow rerank failed: {exc}"
        return state


# ---------------------------------------------------------------------------
# Node: Fast Retrieval (Fast Path — supports sub-task multi-query)
# ---------------------------------------------------------------------------

class FastRetrievalNode:
    """
    Hybrid retrieval for the fast complex path.

    Retrieves documents for each sub-task in parallel using multi-query
    retrieval (``retriever.retrieve_multi_query``), then merges results
    via RRRF. Respects the ``retrieval_strategy`` set by
    RetrievalPlanningNode for per-modality routing.
    """

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def run(self, state: GraphState) -> GraphState:
        """
        Execute hybrid retrieval across all sub-tasks.
        """
        sub_tasks = state.get("sub_tasks", [])
        question = state["question"]
        strategy = state.get("retrieval_strategy", RetrievalStrategy.ALL)

        # Build per-task queries
        queries = []
        for task in sub_tasks:
            query = task.description
            if task.key_terms:
                query = f"{query} ({', '.join(task.key_terms)})"
            queries.append(query)

        try:
            if queries:
                chunks = self.retriever.retrieve_multi_query(queries, mode="hybrid")
            else:
                # Fallback: single query with full question
                chunks = self.retriever.retrieve(query=question, mode="hybrid")

            docs: list[RetrievedDoc] = []
            for c in chunks:
                # Annotate each doc with the task's strategy
                docs.append(
                    RetrievedDoc(
                        chunk_id=c.chunk_id,
                        text=c.text,
                        source_doc=c.source_doc,
                        page_number=c.page_number,
                        retrieval_strategy=strategy,
                        dense_score=getattr(c, "dense_score", 0.0),
                        sparse_score=getattr(c, "sparse_score", 0.0),
                    )
                )

            # Deduplicate
            state["fast_retrieved_docs"] = deduplicate_docs(docs)
            # Increment attempt counter for circuit breaker
            state["fast_retrieval_attempts"] = state.get("fast_retrieval_attempts", 0) + 1
        except Exception as exc:  # noqa: BLE001
            state["error_message"] = f"Fast retrieval failed: {exc}"
            state["fast_retrieved_docs"] = []
        return state


# ---------------------------------------------------------------------------
# Node: Fast Rerank (Fast Path)
# ---------------------------------------------------------------------------

class FastRerankNode:
    """
    Re-ranks fast-path documents using BGE-Reranker v2.

    Re-scores the output of FastRetrievalNode with a cross-encoder.
    """

    def __init__(self, reranker: Any) -> None:
        self.reranker = reranker

    def run(self, state: GraphState) -> GraphState:
        """
        Re-rank fast-path documents.
        """
        docs = state.get("fast_retrieved_docs", [])
        if not docs:
            state["fast_reranked_docs"] = []
            return state

        question = state["question"]
        texts = [d.text for d in docs]

        try:
            reranked_indices = self.reranker.rerank(question, texts, top_k=len(texts))
            reranked: list[RetrievedDoc] = []
            for idx, score in reranked_indices:
                doc = docs[idx]
                doc.rerank_score = float(score)
                reranked.append(doc)
            state["fast_reranked_docs"] = reranked
        except Exception as exc:  # noqa: BLE001
            state["fast_reranked_docs"] = docs
            state["error_message"] = f"Fast rerank failed: {exc}"
        return state


# ---------------------------------------------------------------------------
# Node: Evidence Sufficiency Check (Fast Path)
# ---------------------------------------------------------------------------

class EvidenceSufficiencyNode:
    """
    Judges whether the retrieved documents sufficiently answer the query.

    Uses a lightweight LLM call. If evidence is insufficient, the fast path
    loops back to FastRetrievalNode for up to MAX_EVIDENCE_RETRIES attempts.
    After the circuit breaker triggers, forces MCP tool use.
    """

    CHECK_PROMPT = """You are an evidence sufficiency judge.
Given the user's question and a list of retrieved document excerpts, determine
whether the retrieved documents contain sufficient evidence to fully answer
the question.

Respond with EXACTLY one word:
  SUFFICIENT — The documents provide enough information.
  INSUFFICIENT — More retrieval or tools are needed.

Question: {question}

Documents:
{docs}

Verdict:"""  # noqa: E501

    def __init__(self, judge_model: Any) -> None:
        self.judge_model = judge_model

    def run(self, state: GraphState) -> GraphState:
        """
        Judge evidence sufficiency and write result to state.
        """
        docs = state.get("fast_reranked_docs", [])
        question = state["question"]
        attempts = state.get("fast_retrieval_attempts", 0)

        if not docs:
            state["fast_evidence_sufficient"] = False
            state["fast_evidence_reasoning"] = "No documents retrieved."
            return state

        docs_text = "\n\n".join(
            f"[{i+1}] {d.text[:300]}" for i, d in enumerate(docs[:10])
        )
        prompt = self.CHECK_PROMPT.format(question=question, docs=docs_text)

        try:
            response = self.judge_model.chat(
                messages=[{"role": "user", "content": prompt}],
                generation_config=Any(temperature=0.0, top_p=1.0, max_tokens=10),  # type: ignore
            )
            verdict = response.content.strip().upper()
            state["fast_evidence_sufficient"] = "SUFFICIENT" in verdict
            state["fast_evidence_reasoning"] = response.content
        except Exception as exc:  # noqa: BLE001
            # Fail open on error — let the circuit breaker handle it
            state["fast_evidence_sufficient"] = True
            state["fast_evidence_reasoning"] = f"Judgment error: {exc}"

        return state


# ---------------------------------------------------------------------------
# Node: MCP Tool Call (Fast Path)
# ---------------------------------------------------------------------------

class MCPToolNode:
    """
    Invokes MCP tools when the fast-path evidence loop is exhausted
    or when retrieval results are missing/modality-deficient.

    Available tools:
    - deepdoc_analysis  — RAGFlow DeepDoc for cross-page table parsing
    - financial_calc    — Financial ratio / formula calculator
    """

    def __init__(self, mcp_tool_registry: Optional[dict[str, Any]] = None) -> None:
        self.tools = mcp_tool_registry or {}

    def run(self, state: GraphState) -> GraphState:
        """
        Execute MCP tool calls and augment the document set.
        """
        docs = list(state.get("fast_reranked_docs", []))
        question = state["question"]

        tool_calls: list[ToolCall] = []

        # Decide which tools to invoke based on retrieval gaps
        table_docs = [d for d in docs if d.retrieval_strategy == RetrievalStrategy.TABLE]

        if not docs:
            # No results at all — try DeepDoc layout analysis
            tool_calls.append(
                ToolCall(tool_name="deepdoc_analysis", arguments={"query": question})
            )
        elif not table_docs:
            # Have text docs but no tables — try financial calculator
            tool_calls.append(
                ToolCall(
                    tool_name="financial_calc",
                    arguments={
                        "table_texts": [],
                        "query": question,
                    },
                )
            )

        # Execute each tool
        for tc in tool_calls:
            tool_fn = self.tools.get(tc.tool_name)
            if tool_fn is None:
                tc.error = f"Tool '{tc.tool_name}' not registered in MCP registry."
                continue
            try:
                result = tool_fn(**tc.arguments)
                tc.result = result
            except Exception as exc:  # noqa: BLE001
                tc.error = str(exc)

        state["tool_calls"] = tool_calls
        return state


# ---------------------------------------------------------------------------
# Node: Generation (Shared — both paths)
# ---------------------------------------------------------------------------

class GenerationNode:
    """
    Final answer generation with inline citation formatting.

    Shared by both slow and fast complex paths. Reads from
    ``slow_reranked_docs`` (slow path) or ``fast_reranked_docs`` (fast path).
    """

    def __init__(self, generator: Any, memory_manager: Optional[Any] = None) -> None:
        self.generator = generator
        self.memory_manager = memory_manager

    def run(self, state: GraphState) -> GraphState:
        """
        Generate a citation-aware answer.

        Determines which path we are in from the presence of
        non-empty slow_reranked_docs vs fast_reranked_docs.
        """
        question = state["question"]
        route = state.get("route", Route.COMPLEX_SLOW)

        # Select docs based on path
        if route == Route.COMPLEX_SLOW:
            docs = state.get("slow_reranked_docs", [])
        else:
            docs = state.get("fast_reranked_docs", [])

        evidence_chunks = [
            {
                "text": d.text,
                "source": d.source_doc,
                "page": d.page_number,
            }
            for d in docs
        ]

        try:
            response = self.generator.generate_answer(
                query=question,
                evidence_chunks=evidence_chunks,
            )
            state["answer"] = response.content
            state["citations"] = [
                {
                    "source_doc": d.source_doc,
                    "page_number": d.page_number,
                    "text": d.text[:200],
                }
                for d in docs
            ]
        except Exception as exc:  # noqa: BLE001
            state["error_message"] = f"Generation failed: {exc}"
            state["answer"] = (
                "I encountered an error while generating the answer. "
                "Please try again or rephrase your question."
            )
        return state


# ---------------------------------------------------------------------------
# Conditional Edge Functions
# ---------------------------------------------------------------------------

def route_after_semantic_router(state: GraphState) -> str:
    """
    Conditional edge: decide next node after semantic router.

    Returns
    -------
    "direct_generation"  if Route.SIMPLE
    "slow_retrieval"     if Route.COMPLEX_SLOW
    "task_decomposition"  if Route.COMPLEX_FAST
    """
    route = state.get("route")
    if route == Route.SIMPLE:
        return "direct_generation"
    elif route == Route.COMPLEX_SLOW:
        return "slow_retrieval"
    return "task_decomposition"


def route_after_evidence_check(state: GraphState) -> str:
    """
    Conditional edge: fast path — route after evidence sufficiency check.

    Loop logic:
      - If NOT sufficient AND attempts < MAX_EVIDENCE_RETRIES → loop back to fast_retrieval
      - Otherwise → MCP tool call (whether sufficient or circuit-broken)
    """
    sufficient = state.get("fast_evidence_sufficient", False)
    attempts = state.get("fast_retrieval_attempts", 0)

    if not sufficient and attempts < MAX_EVIDENCE_RETRIES:
        return "fast_retrieval"  # Loop back
    return "mcp_tool_call"


# ---------------------------------------------------------------------------
# Main Workflow Builder
# ---------------------------------------------------------------------------

class FinancialRAGWorkflow:
    """
    LangGraph-based RAG workflow orchestrator.

    Two complex paths + one simple path, all sharing the same entry point
    and the same final generation node.

    Parameters
    ----------
    router : QwenRouter
        Semantic router (Qwen3-0.6B).
    generator : QwenGenerator
        Answer generator (Qwen3-8B).
    retriever : HybridRetriever
        Hybrid retrieval engine (BGE-M3 + BM25 + RRRF).
    reranker : BGEReranker
        Cross-encoder reranker (BGE-Reranker v2).
    judge_model : Any, optional
        Lightweight LLM for evidence sufficiency check. Defaults to ``router``.
    memory_manager : MemoryManager, optional
        Conversation context manager.
    mcp_tool_registry : dict[str, callable], optional
        MCP tool registry (tool_name → callable).
    session_id : str, optional
        Session ID for conversation tracking.
    """

    def __init__(
        self,
        router: Any,
        generator: Any,
        retriever: Any,
        reranker: Any,
        judge_model: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
        mcp_tool_registry: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.router = router
        self.generator = generator
        self.retriever = retriever
        self.reranker = reranker
        self.judge_model = judge_model or router
        self.memory_manager = memory_manager
        self.session_id = session_id or str(uuid.uuid4())
        self.mcp_tool_registry = mcp_tool_registry or {}

        # Initialize all nodes
        self._input = InputNode(session_id=self.session_id)
        self._semantic_router = SemanticRouterNode(router)
        self._direct_gen = DirectGenerationNode(generator, memory_manager)
        self._task_decomposition = TaskDecompositionNode(generator)
        self._retrieval_planning = RetrievalPlanningNode()
        self._slow_retrieval = SlowRetrievalNode(retriever)
        self._slow_rerank = SlowRerankNode(reranker)
        self._fast_retrieval = FastRetrievalNode(retriever)
        self._fast_rerank = FastRerankNode(reranker)
        self._evidence_check = EvidenceSufficiencyNode(self.judge_model)
        self._mcp_tool = MCPToolNode(self.mcp_tool_registry)
        self._generation = GenerationNode(generator, memory_manager)

        # Build and compile graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Assemble the LangGraph state machine."""
        g = StateGraph(GraphState)

        # ── Nodes ──────────────────────────────────────────────────────────────
        g.add_node("input", lambda state: self._input.run(state["question"]))
        g.add_node("semantic_router", self._semantic_router.run)
        g.add_node("direct_generation", self._direct_gen.run)
        g.add_node("task_decomposition", self._task_decomposition.run)
        g.add_node("retrieval_planning", self._retrieval_planning.run)
        g.add_node("slow_retrieval", self._slow_retrieval.run)
        g.add_node("slow_rerank", self._slow_rerank.run)
        g.add_node("fast_retrieval", self._fast_retrieval.run)
        g.add_node("fast_rerank", self._fast_rerank.run)
        g.add_node("evidence_check", self._evidence_check.run)
        g.add_node("mcp_tool_call", self._mcp_tool.run)
        g.add_node("generation", self._generation.run)

        # ── Entry ─────────────────────────────────────────────────────────────
        g.set_entry_point("input")

        # ── Input → Semantic Router ─────────────────────────────────────────
        g.add_edge("input", "semantic_router")

        # ── Conditional: Semantic Router → (direct_gen | slow_retrieval | task_decomposition) ──
        g.add_conditional_edges(
            "semantic_router",
            route_after_semantic_router,
            {
                "direct_generation": "direct_generation",
                "slow_retrieval": "slow_retrieval",
                "task_decomposition": "task_decomposition",
            },
        )

        # ── Simple path ───────────────────────────────────────────────────────
        g.add_edge("direct_generation", END)

        # ══════════════════════════════════════════════════════════════════════
        # SLOW PATH:  retrieval → rerank → generation
        # ══════════════════════════════════════════════════════════════════════
        g.add_edge("slow_retrieval", "slow_rerank")
        g.add_edge("slow_rerank", "generation")
        g.add_edge("generation", END)

        # ══════════════════════════════════════════════════════════════════════
        # FAST PATH:  task_decomp → retrieval_plan → fast_retrieval
        #            → fast_rerank → evidence_check
        #            → [loop: fast_retrieval] | [mcp_tool_call]
        #            → generation
        # ══════════════════════════════════════════════════════════════════════
        g.add_edge("task_decomposition", "retrieval_planning")
        g.add_edge("retrieval_planning", "fast_retrieval")
        g.add_edge("fast_retrieval", "fast_rerank")
        g.add_edge("fast_rerank", "evidence_check")

        # ── Conditional: evidence_check → (fast_retrieval loop | mcp_tool_call) ──
        g.add_conditional_edges(
            "evidence_check",
            route_after_evidence_check,
            {
                "fast_retrieval": "fast_retrieval",  # Loop back
                "mcp_tool_call": "mcp_tool_call",
            },
        )

        # ── MCP tool call → generation ────────────────────────────────────────
        g.add_edge("mcp_tool_call", "generation")

        return g.compile()

    def run(self, question: str) -> dict[str, Any]:
        """
        Execute the full RAG pipeline for a question.

        Parameters
        ----------
        question : str

        Returns
        -------
        dict with keys: question, route, answer, citations, sub_tasks,
                        slow/fast_retrieved_docs, tool_calls, error_message
        """
        initial_state = self._input.run(question)
        final_state = self.graph.invoke(initial_state)

        # Update memory
        if self.memory_manager is not None:
            self.memory_manager.add_turn("user", question)
            if final_state.get("answer"):
                self.memory_manager.add_turn("assistant", final_state["answer"])

        return dict(final_state)

    def run_with_history(
        self,
        question: str,
        conversation_history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Execute the pipeline with explicit conversation history.
        """
        initial_state = self._input.run(question)
        initial_state["conversation_history"] = conversation_history
        return self.graph.invoke(initial_state)
