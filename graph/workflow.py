"""
LangGraph Workflow for Financial Multimodal RAG.

Implements a state machine with the following node graph:

  INPUT
    │
    ▼
  SEMANTIC_ROUTER  ──simple──▶  SIMPLE_GENERATION  ──────┐
    │                                               │
    │ complex                                       │
    ▼                                               │
  TASK_DECOMPOSITION ──▶ RETRIEVAL ──▶ RERANK ──┐  │
    │                                           │  │
    ▼                                           ▼  ▼
  EVIDENCE_SUFFICIENCY ◀──────              GENERATION
    │ (insufficient)                           │
    │                                           │
    ▼                                           │
  MCP_TOOL_CALL ────────────────────────────────┘

The state dict (GraphState) carries all intermediate results across nodes.
Conditional edges handle the fast/slow branching and loop-back logic.

Key Design Decisions
--------------------
1. **Conditional edges** — Every branching decision is an explicit function,
   not implicit if/else inside a node. This makes the graph fully inspectable.
2. **Isolated nodes** — Each node is a pure function: ``state → state``.
   No side effects. LLM calls, retrieval, and tool execution are all nodes.
3. **Retry via loop-back** — If EvidenceSufficiency fails, we loop back to
   TaskDecomposition (re-plan) or directly to Retrieval (fetch more).
4. **Citation-aware generation** — The generation node formats inline citations
   from the reranked chunk metadata.

Usage
-----
    >>> from models.qwen_llm import QwenRouter, QwenGenerator
    >>> from retrieval import HybridRetriever
    >>> router = QwenRouter()
    >>> generator = QwenGenerator()
    >>> retriever = HybridRetriever()
    >>> workflow = FinancialRAGWorkflow(router, generator, retriever)
    >>> result = workflow.run("What was Apple's total revenue in FY2024?")
    >>> print(result["answer"])
    Apple reported total revenue of $391.0B in fiscal year 2024...
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Optional, TypedDict

import operator
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------


class Route(str, Enum):
    """Possible routes from the SemanticRouter node."""

    SIMPLE = "simple"
    COMPLEX = "complex"


class RetrievalStrategy(str, Enum):
    """Strategy to use for retrieval."""

    TEXT = "text"       # Retrieve from text chunks
    TABLE = "table"     # Retrieve from table segments
    FIGURE = "figure"   # Retrieve from figure captions
    ALL = "all"          # Retrieve all modalities


@dataclass
class RetrievedDoc:
    """A retrieved document chunk with source metadata for citation."""

    chunk_id: str
    text: str
    source_doc: str
    page_number: int
    retrieval_strategy: RetrievalStrategy


@dataclass
class ToolCall:
    """Represents a planned or executed MCP tool call."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any = None
    error: Optional[str] = None


class GraphState(TypedDict, total=False):
    """
    Shared state carried through the LangGraph pipeline.

    All fields are optional — each node only reads/writes the fields it needs.
    """

    # --- Input ---
    question: Annotated[str, operator.add]
    session_id: str

    # --- Routing ---
    route: Optional[Route]
    routing_reasoning: Optional[str]

    # --- Task Decomposition ---
    sub_tasks: list[dict[str, Any]]  # [{task_id, description, strategy}]

    # --- Retrieval ---
    retrieved_docs: list[RetrievedDoc]
    retrieval_strategy: RetrievalStrategy

    # --- Rerank ---
    reranked_docs: list[RetrievedDoc]

    # --- Evidence ---
    evidence_sufficient: bool
    evidence_reasoning: Optional[str]

    # --- Tools ---
    tool_calls: list[ToolCall]

    # --- Generation ---
    answer: Optional[str]
    citations: list[dict[str, Any]]  # [{source_doc, page_number, text}]

    # --- Memory ---
    conversation_history: list[dict[str, str]]

    # --- Error ---
    error_message: Optional[str]


# ---------------------------------------------------------------------------
# Node Definitions
# ---------------------------------------------------------------------------

def _make_state(**kwargs: Any) -> GraphState:
    """Create a GraphState dict with defaults."""
    return GraphState(**kwargs)


class InputNode:
    """
    Initializes the graph state from a raw user question.

    Parameters
    ----------
    session_id : str, optional
        Session identifier for conversation context. Auto-generated if omitted.
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
            retrieved_docs=[],
            retrieval_strategy=RetrievalStrategy.ALL,
            reranked_docs=[],
            evidence_sufficient=False,
            evidence_reasoning=None,
            tool_calls=[],
            answer=None,
            citations=[],
            conversation_history=[],
            error_message=None,
        )


class SemanticRouterNode:
    """
    Classifies the incoming question into ``simple`` or ``complex``.

    Uses Qwen3-0.6B (lightweight, low-latency) to decide whether to bypass
    the full retrieval pipeline.

    Parameters
    ----------
    router : QwenRouter
        Initialized router instance.
    """

    def __init__(self, router: Any) -> None:
        self.router = router

    def run(self, state: GraphState) -> GraphState:
        """
        Classify the question and write ``route`` into state.

        Returns updated state with ``route`` field populated.
        """
        question = state["question"]
        try:
            route = self.router.classify(question)
            reasoning = ""
            if hasattr(route, "content"):
                reasoning = route.content
            state["route"] = Route(route)
            state["routing_reasoning"] = reasoning
        except Exception as exc:  # noqa: BLE001
            # Fail open — treat as complex on routing errors
            state["route"] = Route.COMPLEX
            state["routing_reasoning"] = f"Routing error: {exc}"
        return state


def route_based_on_complexity(state: GraphState) -> str:
    """
    Conditional edge: decide next node based on router output.

    Parameters
    ----------
    state : GraphState

    Returns
    -------
    "simple_generation" if Route.SIMPLE, else "task_decomposition"
    """
    route = state.get("route")
    if route == Route.SIMPLE:
        return "simple_generation"
    return "task_decomposition"


class SimpleGenerationNode:
    """
    Fast-path generation for simple, single-hop questions.

    Bypasses retrieval entirely and generates directly from the conversation
    history context. Uses Qwen3-8B for quality.

    Parameters
    ----------
    generator : QwenGenerator
    memory_manager : MemoryManager, optional
    """

    def __init__(self, generator: Any, memory_manager: Optional[Any] = None) -> None:
        self.generator = generator
        self.memory_manager = memory_manager

    def run(self, state: GraphState) -> GraphState:
        """
        Generate a direct answer for simple queries.

        Returns updated state with ``answer`` and ``citations``.
        """
        question = state["question"]
        messages = state.get("conversation_history", [])
        messages.append({"role": "user", "content": question})

        # Reserve ~6k tokens for generation
        context_messages = (
            self.memory_manager.get_context_for_prompt(query_tokens=500)
            if self.memory_manager
            else messages
        )

        try:
            response = self.generator.chat(messages=context_messages)
            state["answer"] = response.content
            state["citations"] = []  # No retrieval → no citations
        except Exception as exc:  # noqa: BLE001
            state["error_message"] = f"Simple generation failed: {exc}"
        return state


class TaskDecompositionNode:
    """
    Decomposes a complex multi-hop question into parallel sub-tasks.

    Uses Qwen3-0.6B (same model as router) to generate a plan.
    Each sub-task has a ``description`` and a ``retrieval_strategy`` (text/table/figure).

    Parameters
    ----------
    generator : QwenGenerator
        Actually uses the router model for lightweight planning.
    """

    PLANNER_PROMPT = """You are a query planner for a financial RAG system.
Given the user's complex question, break it down into N independent sub-tasks.

For each sub-task provide:
  1. description — What specific information does this sub-task need?
  2. strategy — One of: text | table | figure
     - Use "table" if the answer requires financial table data (revenue, ROE, etc.)
     - Use "figure" if the answer requires chart / figure data
     - Use "text" otherwise (qualitative descriptions, risk factors, etc.)
  3. key_terms — List of important financial terms / numbers to search for

Output format (JSON array):
[
  {{"task_id": "t1", "description": "...", "strategy": "table", "key_terms": ["revenue", "FY2024"]}},
  ...
]

Return ONLY the JSON array, no markdown, no explanation."""

    def __init__(self, generator: Any) -> None:
        self.generator = generator

    def run(self, state: GraphState) -> GraphState:
        """
        Decompose the question into sub-tasks and write to state.

        Returns updated state with ``sub_tasks`` populated.
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
            # Attempt JSON parse — if fails, create a single fallback task
            try:
                sub_tasks = json.loads(raw)
            except json.JSONDecodeError:
                sub_tasks = [
                    {
                        "task_id": "t0",
                        "description": question,
                        "strategy": "all",
                        "key_terms": [],
                    }
                ]
            state["sub_tasks"] = sub_tasks
        except Exception as exc:  # noqa: BLE001
            state["error_message"] = f"Task decomposition failed: {exc}"
            state["sub_tasks"] = [
                {
                    "task_id": "t0",
                    "description": state["question"],
                    "strategy": "all",
                    "key_terms": [],
                }
            ]
        return state


class RetrievalNode:
    """
    Executes hybrid retrieval (BGE-M3 + BM25 + RRRF) for each sub-task.

    Parameters
    ----------
    retriever : HybridRetriever
    """

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def run(self, state: GraphState) -> GraphState:
        """
        Retrieve documents for each sub-task in parallel.

        Returns updated state with ``retrieved_docs`` merged from all sub-tasks.
        """
        sub_tasks = state.get("sub_tasks", [])
        retrieved: list[RetrievedDoc] = []

        for task in sub_tasks:
            strategy = task.get("strategy", "all")
            description = task.get("description", state["question"])
            key_terms = task.get("key_terms", [])

            # Build a query from description + key terms
            query = description
            if key_terms:
                query = f"{description} ({', '.join(key_terms)})"

            try:
                chunks = self.retriever.retrieve(query, mode="hybrid")
                for c in chunks:
                    retrieved.append(
                        RetrievedDoc(
                            chunk_id=c.chunk_id,
                            text=c.text,
                            source_doc=c.source_doc,
                            page_number=c.page_number,
                            retrieval_strategy=RetrievalStrategy(strategy),
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                # Log but continue — partial retrieval is better than none
                state["error_message"] = f"Retrieval failed for task {task.get('task_id')}: {exc}"

        state["retrieved_docs"] = deduplicate_docs(retrieved)
        return state


class RerankNode:
    """
    Re-ranks retrieved documents using BGE-Reranker for precision.

    Parameters
    ----------
    reranker : BGEReranker
    """

    def __init__(self, reranker: Any) -> None:
        self.reranker = reranker

    def run(self, state: GraphState) -> GraphState:
        """
        Re-rank all retrieved docs with the cross-encoder.

        Returns updated state with ``reranked_docs`` (top-k after reranking).
        """
        docs = state.get("retrieved_docs", [])
        if not docs:
            state["reranked_docs"] = []
            return state

        question = state["question"]
        texts = [d.text for d in docs]

        try:
            reranked_scores = self.reranker.rerank(question, texts, top_k=len(texts))
            reranked = []
            for idx, score in reranked_scores:
                docs[idx].score = score  # type: ignore[attr-defined]
                reranked.append(docs[idx])
            state["reranked_docs"] = reranked
        except Exception as exc:  # noqa: BLE001
            # Fall back to pre-rerank order on error
            state["reranked_docs"] = docs
            state["error_message"] = f"Reranking failed: {exc}"

        return state


def route_after_rerank(state: GraphState) -> str:
    """
    Conditional edge: decide whether to call tools or go straight to generation.

    If no docs were retrieved, go directly to generation (will produce a
    fallback answer saying no evidence was found).
    """
    reranked = state.get("reranked_docs", [])
    if not reranked:
        return "generation"
    return "evidence_check"


class EvidenceSufficiencyNode:
    """
    Checks whether the reranked documents actually satisfy the query.

    Uses a lightweight LLM call to judge sufficiency — avoids wasting
    generation tokens on insufficient evidence.

    Parameters
    ----------
    judge_model : Any (lightweight LLM with chat interface)
    """

    CHECK_PROMPT = """You are an evidence sufficiency judge.
Given the user's question and a list of retrieved document excerpts, determine whether
the retrieved documents contain sufficient evidence to answer the question.

Respond with EXACTLY one of:
  SUFFICIENT — The documents provide enough information to answer.
  INSUFFICIENT — The documents do NOT provide enough information; more retrieval or tools needed.

Question: {question}

Documents:
{docs}

Your verdict:"""  # noqa: E501

    def __init__(self, judge_model: Any) -> None:
        self.judge_model = judge_model

    def run(self, state: GraphState) -> GraphState:
        """
        Judge whether reranked docs satisfy the query.

        Returns updated state with ``evidence_sufficient`` boolean.
        """
        docs = state.get("reranked_docs", [])
        question = state["question"]

        if not docs:
            state["evidence_sufficient"] = False
            state["evidence_reasoning"] = "No documents retrieved."
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
            sufficient = "SUFFICIENT" in verdict
            state["evidence_sufficient"] = sufficient
            state["evidence_reasoning"] = response.content
        except Exception as exc:  # noqa: BLE001
            # Fail open — if we can't judge, proceed to generation
            state["evidence_sufficient"] = True
            state["evidence_reasoning"] = f"Judgment error: {exc}"

        return state


def route_evidence_decision(state: GraphState) -> str:
    """
    Conditional edge: loop back to retrieval (with expanded query) or proceed.

    If evidence is insufficient, re-enter the retrieval loop with expanded terms.
    After 2 loops, force generation (circuit breaker).
    """
    if not state.get("evidence_sufficient", False):
        tool_calls = state.get("tool_calls", [])
        if len(tool_calls) >= 2:
            # Circuit breaker — force generation after 2 failed attempts
            return "generation"
        return "mcp_tool_call"
    return "generation"


class MCPToolNode:
    """
    Calls MCP tools when built-in retrieval is insufficient.

    Tools available:
    - deepdoc_analysis : Invoke RAGFlow DeepDoc for cross-page table parsing
    - financial_calc   : Evaluate financial formulas on retrieved table data

    Parameters
    ----------
    mcp_tool_registry : dict[str, callable]
        Map of tool_name → tool function.
    """

    def __init__(self, mcp_tool_registry: Optional[dict[str, Any]] = None) -> None:
        self.tools = mcp_tool_registry or {}

    def run(self, state: GraphState) -> GraphState:
        """
        Execute MCP tool calls for insufficient evidence cases.

        Returns updated state with ``tool_calls`` populated and ``retrieved_docs`` augmented.
        """
        reranked = state.get("reranked_docs", [])
        question = state["question"]

        # Plan which tool to call based on retrieval gaps
        tool_calls: list[ToolCall] = []
        if not reranked:
            # No retrieval results — use DeepDoc to re-parse the document
            tool_calls.append(
                ToolCall(
                    tool_name="deepdoc_analysis",
                    arguments={"query": question},
                )
            )
        else:
            # Sufficient docs found but evidence still flagged insufficient —
            # use financial calculator on retrieved table data
            table_docs = [d for d in reranked if d.retrieval_strategy == RetrievalStrategy.TABLE]
            if table_docs:
                tool_calls.append(
                    ToolCall(
                        tool_name="financial_calc",
                        arguments={
                            "table_texts": [d.text for d in table_docs],
                            "query": question,
                        },
                    )
                )

        # Execute tools
        executed_results: list[RetrievedDoc] = list(reranked)
        for tc in tool_calls:
            tool_fn = self.tools.get(tc.tool_name)
            if tool_fn is None:
                tc.error = f"Tool '{tc.tool_name}' not registered."
            else:
                try:
                    result = tool_fn(**tc.arguments)
                    tc.result = result
                    if isinstance(result, list) and result and hasattr(result[0], "text"):
                        executed_results.extend(result)
                except Exception as exc:  # noqa: BLE001
                    tc.error = str(exc)

        state["tool_calls"] = tool_calls
        state["retrieved_docs"] = deduplicate_docs(executed_results)

        # Re-rerank with augmented document set
        from .workflow import RerankNode

        # Re-run rerank on augmented docs
        # Note: we use the same reranker by re-invoking through the graph
        # For simplicity, just use the existing reranked_docs and add new ones
        state["reranked_docs"] = executed_results[:20]

        return state


class GenerationNode:
    """
    Final answer generation node with inline citation formatting.

    Parameters
    ----------
    generator : QwenGenerator
    memory_manager : MemoryManager, optional
    """

    def __init__(self, generator: Any, memory_manager: Optional[Any] = None) -> None:
        self.generator = generator
        self.memory_manager = memory_manager

    def run(self, state: GraphState) -> GraphState:
        """
        Generate a citation-aware answer from reranked documents.

        Returns updated state with ``answer`` and ``citations``.
        """
        question = state["question"]
        docs = state.get("reranked_docs", [])

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
# Utility functions
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


# ---------------------------------------------------------------------------
# Main Workflow Builder
# ---------------------------------------------------------------------------


class FinancialRAGWorkflow:
    """
    LangGraph-based RAG workflow orchestrator.

    Wraps the compiled state graph and exposes a simple ``run(question)`` API.

    Parameters
    ----------
    router : QwenRouter
        Semantic router (Qwen3-0.6B).
    generator : QwenGenerator
        Answer generator (Qwen3-8B).
    retriever : HybridRetriever
        Hybrid retrieval engine.
    reranker : BGEReranker
        Cross-encoder reranker.
    judge_model : Any, optional
        Lightweight LLM for evidence sufficiency check.
        Defaults to using ``router`` if omitted.
    memory_manager : MemoryManager, optional
        Conversation context manager.
    mcp_tool_registry : dict[str, callable], optional
        MCP tool registry for DeepDoc / financial calculation tools.
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

        # Initialize nodes
        self._input = InputNode(session_id=self.session_id)
        self._semantic_router = SemanticRouterNode(router)
        self._simple_gen = SimpleGenerationNode(generator, memory_manager)
        self._decomposition = TaskDecompositionNode(generator)
        self._retrieval = RetrievalNode(retriever)
        self._rerank = RerankNode(reranker)
        self._evidence_check = EvidenceSufficiencyNode(self.judge_model)
        self._mcp_tool = MCPToolNode(mcp_tool_registry)
        self._generation = GenerationNode(generator, memory_manager)

        # Build and compile graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Assemble the LangGraph state machine."""
        g = StateGraph(GraphState)

        # Add all nodes
        g.add_node("input", lambda state: self._input.run(state["question"]))
        g.add_node("semantic_router", self._semantic_router.run)
        g.add_node("simple_generation", self._simple_gen.run)
        g.add_node("task_decomposition", self._decomposition.run)
        g.add_node("retrieval", self._retrieval.run)
        g.add_node("rerank", self._rerank.run)
        g.add_node("evidence_check", self._evidence_check.run)
        g.add_node("mcp_tool_call", self._mcp_tool.run)
        g.add_node("generation", self._generation.run)

        # Set entry point
        g.set_entry_point("semantic_router")

        # Conditional edge: simple vs complex
        g.add_conditional_edges(
            "semantic_router",
            route_based_on_complexity,
            {
                "simple_generation": "simple_generation",
                "task_decomposition": "task_decomposition",
            },
        )

        # Simple path → end
        g.add_edge("simple_generation", END)

        # Complex path: decomposition → retrieval → rerank → evidence_check
        g.add_edge("task_decomposition", "retrieval")
        g.add_edge("retrieval", "rerank")

        # Conditional edge: check evidence
        g.add_conditional_edges(
            "rerank",
            route_after_rerank,
            {
                "evidence_check": "evidence_check",
                "generation": "generation",
            },
        )

        # Evidence insufficient → MCP tools → retrieval (loop) or generation
        g.add_conditional_edges(
            "evidence_check",
            route_evidence_decision,
            {
                "mcp_tool_call": "mcp_tool_call",
                "generation": "generation",
            },
        )

        # After tools, loop back to rerank (circuit breaker in route_evidence_decision)
        g.add_edge("mcp_tool_call", "retrieval")

        # Final generation → end
        g.add_edge("generation", END)

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
                        retrieved_docs, tool_calls, error_message
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

        Useful when the memory manager is not persistent across sessions.
        """
        initial_state = self._input.run(question)
        initial_state["conversation_history"] = conversation_history
        return self.graph.invoke(initial_state)
