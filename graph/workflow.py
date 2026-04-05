"""
LangGraph Workflow for Financial Multimodal RAG.

Complete three-path pipeline:

  Fast path (COMPLEX_FAST):
    input → semantic_router → task_decomposition → retrieval_planning
      → fast_retrieval → fast_rerank → evidence_check
      → [loop: fast_retrieval | fallback: mcp_tool_call] → generation → verification → end

  Slow path (COMPLEX_SLOW):
    input → semantic_router → slow_retrieval → slow_rerank → generation → verification → end

  Simple path (SIMPLE):
    input → semantic_router → direct_generation → end

All paths share the same GraphState schema (graph/state.py).
Each node is a pure function: state_in → state_out.
All routing decisions are explicit conditional edge functions.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal, Optional

import operator
from langgraph.graph import END, StateGraph

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_EVIDENCE_RETRIES = 2  # Circuit breaker for fast-path evidence loop


# ---------------------------------------------------------------------------
# Conditional Edge Functions
# ---------------------------------------------------------------------------

def route_after_semantic_router(state: GraphState) -> str:
    """
    Conditional edge: decide next node based on semantic router output.

    SIMPLE  → direct_generation (no retrieval)
    SLOW    → slow_retrieval (full retrieval, no planning)
    FAST    → task_decomposition (full fast path)
    """
    route = state.get("route")
    if route == Route.SIMPLE:
        return "direct_generation"
    elif route == Route.COMPLEX_SLOW:
        return "slow_retrieval"
    return "task_decomposition"


def route_after_evidence_check(state: GraphState) -> str:
    """
    Conditional edge: fast path — loop back or trigger fallback.

    If evidence is insufficient AND attempts < MAX_EVIDENCE_RETRIES → loop to fast_retrieval
    If evidence is insufficient AND attempts >= MAX → mcp_tool_call
    If evidence is sufficient → mcp_tool_call (tools may still be needed for calc)
    """
    sufficient = state.get("evidence_score", 0.0) >= 0.7
    attempts = state.get("fast_retrieval_attempts", 0)

    if not sufficient and attempts < MAX_EVIDENCE_RETRIES:
        return "fast_retrieval"  # Loop back

    return "mcp_tool_call"


def route_after_mcp_tools(state: GraphState) -> str:
    """
    Conditional edge: after MCP tools, route back to generation.

    After tool execution, always go to generation.
    The circuit breaker is handled in route_after_evidence_check.
    """
    return "generation"


# ---------------------------------------------------------------------------
# Node: Input
# ---------------------------------------------------------------------------

class InputNode:
    """
    Initializes the GraphState from a raw user question.

    This is the canonical entry point — every query first passes through here
    before reaching the semantic router.
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or str(uuid.uuid4())

    def run(self, question: str) -> GraphState:
        return GraphState(
            question=question,
            session_id=self.session_id,
            original_question=question,
            query_rewritten=None,
            route=None,
            routing_reasoning=None,
            task_type=None,
            sub_tasks=[],
            current_task_index=0,
            retrieval_queries=[],
            retrieval_strategy=RetrievalStrategy.ALL,
            slow_retrieved_docs=[],
            slow_reranked_docs=[],
            fast_retrieved_docs=[],
            fast_reranked_docs=[],
            fast_retrieval_attempts=0,
            evidence_snippets=[],
            evidence_score=None,
            evidence_reasoning=None,
            fallback_triggered=FallbackReason.NONE,
            fallback_count=0,
            candidate_tools=[],
            tool_call_results=[],
            answer_draft=None,
            answer_final=None,
            citations=[],
            answer_groundedness_score=None,
            error_message=None,
            node_errors={},
            short_term_context=[],
            memory_summary=None,
            total_tokens_used=0,
        )


# ---------------------------------------------------------------------------
# Node: Semantic Router
# ---------------------------------------------------------------------------

class SemanticRouterNode:
    """
    Classifies the question into SIMPLE / SLOW / FAST using Qwen3-0.6B.

    Also sets the TaskType based on the question's analytical intent.
    """

    ROUTER_PROMPT = """You are a semantic query router for a financial RAG system.

Classify the question into exactly ONE of three categories:

  simple  — Single-hop factual question. No sub-task planning needed.
             Example: "What was Apple's revenue in FY2024?"

  slow    — Multi-hop or comparative question requiring full retrieval
             and reranking, but does NOT need iterative planning loops.
             Examples: "Compare Apple's revenue vs. Microsoft over 3 years",
                       "What is the average ROE across all quarters in 2024?"

  fast    — Complex question requiring task decomposition, per-modality
             retrieval planning, and evidence sufficiency verification.
             Examples: "Summarize risk factors and calculate D/E ratio",
                       "Find all cross-page tables about revenue and compute CAGR"

Reply with ONLY one word: simple | slow | fast"""  # noqa: E501

    TASK_TYPE_PROMPT = """Given the user's question, classify its analytical intent
into one of: factual_query | comparative | trend | risk | financial_calc | cross_table | unknown

Examples:
  "What was revenue?" → factual_query
  "Compare A vs B" → comparative
  "How has ROE changed over 5 years?" → trend
  "What are the risk factors?" → risk
  "Calculate D/E ratio" → financial_calc
  "Combine revenue and margin data" → cross_table

Reply with ONLY the category name."""

    def __init__(self, router: Any) -> None:
        self.router = router

    def run(self, state: GraphState) -> GraphState:
        question = state["question"]
        messages = [{"role": "system", "content": self.ROUTER_PROMPT},
                    {"role": "user", "content": question}]

        try:
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
            state["route"] = Route.COMPLEX_SLOW
            state["routing_reasoning"] = f"Router error: {exc}"

        # Classify task type
        task_messages = [{"role": "system", "content": self.TASK_TYPE_PROMPT},
                         {"role": "user", "content": question}]
        try:
            task_response = self.router.chat(messages=task_messages)
            raw_type = task_response.content.strip().lower()
            try:
                state["task_type"] = TaskType(raw_type)
            except ValueError:
                state["task_type"] = TaskType.UNKNOWN
        except Exception:  # noqa: BLE001
            state["task_type"] = TaskType.UNKNOWN

        return state


# ---------------------------------------------------------------------------
# Node: Direct Generation (Simple Path)
# ---------------------------------------------------------------------------

class DirectGenerationNode:
    """
    Fast-path generation for simple questions.
    Bypasses retrieval entirely — generates directly from conversation history.
    """

    def __init__(self, generator: Any, memory_manager: Optional[Any] = None) -> None:
        self.generator = generator
        self.memory_manager = memory_manager

    def run(self, state: GraphState) -> GraphState:
        question = state["question"]
        context = state.get("short_term_context", [])
        context.append({"role": "user", "content": question})

        try:
            context_messages = (
                self.memory_manager.get_context_for_prompt(query_tokens=500)
                if self.memory_manager else context
            )
            response = self.generator.chat(messages=context_messages)
            state["answer_final"] = response.content
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

    Each SubTask carries: task_id, description, strategy, key_terms.
    """

    PLANNER_PROMPT = """Break this complex question into N independent sub-tasks.

Output JSON array:
[
  {"task_id": "t1", "description": "...", "strategy": "text|table|figure|all", "key_terms": ["..."]},
  ...
]

Rules:
- strategy: "table" if the answer needs numeric table data
           "text" if qualitative (risk factors, discussion)
           "figure" if charts/graphs are needed
           "all" if no specific modality dominates
- Include all key financial terms, company names, years in key_terms
- Each sub-task should be answerable independently"""

    def __init__(self, generator: Any) -> None:
        self.generator = generator

    def run(self, state: GraphState) -> GraphState:
        import json
        question = state["question"]
        messages = [{"role": "system", "content": self.PLANNLER_PROMPT},
                    {"role": "user", "content": question}]

        try:
            response = self.generator.chat(messages=messages)
            raw = response.content.strip()
            raw_tasks = json.loads(raw) if raw.startswith("[") else []
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
            state["node_errors"]["task_decomposition"] = str(exc)
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
    Analyzes sub-tasks and sets the dominant retrieval strategy.

    Builds per-sub-task retrieval query strings and stores them
    in ``retrieval_queries``.
    """

    def run(self, state: GraphState) -> GraphState:
        sub_tasks = state.get("sub_tasks", [])
        if not sub_tasks:
            state["retrieval_queries"] = [state["question"]]
            return state

        # Count modality distribution
        strategy_counts: dict[RetrievalStrategy, int] = {}
        for task in sub_tasks:
            s = RetrievalStrategy(task.strategy)
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        dominant = max(strategy_counts, key=strategy_counts.get)
        if dominant == RetrievalStrategy.ALL and len(strategy_counts) > 1:
            non_all = {k: v for k, v in strategy_counts.items() if k != RetrievalStrategy.ALL}
            if non_all:
                dominant = max(non_all, key=non_all.get)

        state["retrieval_strategy"] = dominant

        # Build retrieval queries for each sub-task
        queries: list[str] = []
        for task in sub_tasks:
            q = task.description
            if task.key_terms:
                q = f"{q} ({', '.join(task.key_terms)})"
            queries.append(q)

        state["retrieval_queries"] = queries
        return state


# ---------------------------------------------------------------------------
# Node: Slow Retrieval (Slow Path — dual召回)
# ---------------------------------------------------------------------------

class SlowRetrievalNode:
    """
    Hybrid retrieval for the slow complex path (single query, no decomposition).
    """

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def run(self, state: GraphState) -> GraphState:
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
# Node: Slow Rerank
# ---------------------------------------------------------------------------

class SlowRerankNode:
    """Re-ranks slow-path documents using BGE-Reranker."""

    def __init__(self, reranker: Any) -> None:
        self.reranker = reranker

    def run(self, state: GraphState) -> GraphState:
        docs = state.get("slow_retrieved_docs", [])
        if not docs:
            state["slow_reranked_docs"] = []
            return state

        question = state["question"]
        texts = [d.text for d in docs]

        try:
            reranked_idx = self.reranker.rerank(question, texts, top_k=len(texts))
            reranked = []
            for idx, score in reranked_idx:
                d = docs[idx]
                d.rerank_score = float(score)
                reranked.append(d)
            state["slow_reranked_docs"] = reranked
        except Exception as exc:  # noqa: BLE001
            state["slow_reranked_docs"] = docs
            state["node_errors"]["slow_rerank"] = str(exc)
        return state


# ---------------------------------------------------------------------------
# Node: Fast Retrieval (Fast Path — multi-query)
# ---------------------------------------------------------------------------

class FastRetrievalNode:
    """
    Multi-query retrieval for the fast complex path.
    Uses retrieve_multi_query to handle decomposed sub-tasks in parallel.
    """

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def run(self, state: GraphState) -> GraphState:
        queries = state.get("retrieval_queries", [])
        strategy = state.get("retrieval_strategy", RetrievalStrategy.ALL)

        if not queries:
            queries = [state["question"]]

        try:
            chunks = self.retriever.retrieve_multi_query(queries=queries, mode="hybrid")
            docs: list[RetrievedDoc] = [
                RetrievedDoc(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    source_doc=c.source_doc,
                    page_number=c.page_number,
                    retrieval_strategy=strategy,
                    dense_score=getattr(c, "dense_score", 0.0),
                    sparse_score=getattr(c, "sparse_score", 0.0),
                )
                for c in chunks
            ]
            state["fast_retrieved_docs"] = deduplicate(docs)
            state["fast_retrieval_attempts"] = state.get("fast_retrieval_attempts", 0) + 1
        except Exception as exc:  # noqa: BLE001
            state["node_errors"]["fast_retrieval"] = str(exc)
            state["fast_retrieved_docs"] = []
        return state


# ---------------------------------------------------------------------------
# Node: Fast Rerank
# ---------------------------------------------------------------------------

class FastRerankNode:
    """Re-ranks fast-path documents using BGE-Reranker."""

    def __init__(self, reranker: Any) -> None:
        self.reranker = reranker

    def run(self, state: GraphState) -> GraphState:
        docs = state.get("fast_retrieved_docs", [])
        if not docs:
            state["fast_reranked_docs"] = []
            return state

        question = state["question"]
        texts = [d.text for d in docs]

        try:
            reranked_idx = self.reranker.rerank(question, texts, top_k=len(texts))
            reranked = []
            for idx, score in reranked_idx:
                d = docs[idx]
                d.rerank_score = float(score)
                reranked.append(d)
            state["fast_reranked_docs"] = reranked
        except Exception as exc:  # noqa: BLE001
            state["fast_reranked_docs"] = docs
            state["node_errors"]["fast_rerank"] = str(exc)
        return state


# ---------------------------------------------------------------------------
# Node: Evidence Sufficiency Check
# ---------------------------------------------------------------------------

class EvidenceSufficiencyNode:
    """
    Judges whether retrieved documents sufficiently answer the query.

    Sets ``evidence_score`` (0.0–1.0) and ``evidence_reasoning`` in state.
    Used by route_after_evidence_check to decide loop vs. fallback.
    """

    CHECK_PROMPT = """Judge whether the retrieved documents can sufficiently answer the question.

Respond with:
  SUFFICIENT (confidence 0.7-1.0) — Evidence covers the key facts
  INSUFFICIENT (confidence 0.0-0.6) — Key facts are missing

Question: {question}

Evidence:
{evidence}

Confidence score (0.0-1.0) and brief reasoning:"""  # noqa: E501

    def __init__(self, judge_model: Any) -> None:
        self.judge_model = judge_model

    def run(self, state: GraphState) -> GraphState:
        docs = state.get("fast_reranked_docs", [])
        question = state["question"]

        if not docs:
            state["evidence_score"] = 0.0
            state["evidence_reasoning"] = "No documents retrieved."
            return state

        evidence_text = "\n\n".join(
            f"[{i+1}] {d.text[:300]}" for i, d in enumerate(docs[:10])
        )
        prompt = self.CHECK_PROMPT.format(question=question, evidence=evidence_text)

        try:
            response = self.judge_model.chat(
                messages=[{"role": "user", "content": prompt}],
                generation_config=Any(temperature=0.0, top_p=1.0, max_tokens=256),  # type: ignore
            )
            raw = response.content.strip().upper()
            sufficient = "SUFFICIENT" in raw

            # Try to extract a numeric confidence
            import re
            conf_match = re.search(r"0\.\d+", raw)
            confidence = float(conf_match.group()) if conf_match else (0.8 if sufficient else 0.3)

            state["evidence_score"] = confidence if sufficient else min(confidence, 0.6)
            state["evidence_reasoning"] = response.content[:200]
        except Exception as exc:  # noqa: BLE001
            state["evidence_score"] = 0.5
            state["evidence_reasoning"] = f"Check error: {exc}"

        return state


# ---------------------------------------------------------------------------
# Node: MCP Tool Call (Fallback)
# ---------------------------------------------------------------------------

class MCPToolNode:
    """
    Invokes MCP tools when the fast-path evidence loop is exhausted
    or when specific capabilities (table parsing, calculation) are needed.

    Tools are selected based on:
    - evidence_score (if insufficient → retrieval augmentation tools)
    - task_type (if financial_calc → analysis tools)
    - retrieval gaps (if no table docs → deepdoc_tools)
    """

    def __init__(self, mcp_tool_registry: Optional[dict[str, callable]] = None) -> None:
        self.tools = mcp_tool_registry or {}

    def run(self, state: GraphState) -> GraphState:
        docs = list(state.get("fast_reranked_docs", []))
        question = state["question"]
        task_type = state.get("task_type", TaskType.UNKNOWN)
        evidence_score = state.get("evidence_score", 0.0)

        candidates: list[ToolCandidate] = []
        selected_results: list[ToolCandidate] = []

        # Strategy 1: Evidence insufficient → try hybrid retrieval augmentation
        if evidence_score < 0.7:
            candidates.append(
                ToolCandidate(
                    tool_name="retrieval_hybrid_search",
                    reason="Evidence score below threshold",
                    selected=True,
                    arguments={"query": question, "top_k": 20},
                )
            )

        # Strategy 2: Financial calculation required
        if task_type == TaskType.FINANCIAL_CALC:
            table_docs = [d for d in docs if d.retrieval_strategy == RetrievalStrategy.TABLE]
            candidates.append(
                ToolCandidate(
                    tool_name="analysis_calc",
                    reason="Task type is financial calculation",
                    selected=True,
                    arguments={
                        "metric": "roe",
                        "values": {},
                    },
                )
            )

        # Strategy 3: No table results but question asks for numbers
        if task_type in (TaskType.FINANCIAL_CALC, TaskType.COMPARATIVE, TaskType.TREND):
            has_tables = any(d.retrieval_strategy == RetrievalStrategy.TABLE for d in docs)
            if not has_tables:
                candidates.append(
                    ToolCandidate(
                        tool_name="deepdoc_table_parse",
                        reason="No table chunks retrieved but tables needed",
                        selected=True,
                        arguments={"query": question, "page_number": 1},
                    )
                )

        # Strategy 4: Cross-page table suspected
        if evidence_score < 0.4:
            candidates.append(
                ToolCandidate(
                    tool_name="deepdoc_cross_page_merge",
                    reason="Very low evidence score, possible cross-page table",
                    selected=True,
                    arguments={"fragments": []},
                )
            )

        state["candidate_tools"] = candidates

        # Execute selected tools
        for tc in candidates:
            tool_fn = self.tools.get(tc.tool_name)
            if tool_fn is None:
                tc.error = f"Tool '{tc.tool_name}' not registered."
                selected_results.append(tc)
                continue
            try:
                tc.result = tool_fn(**tc.arguments)
            except Exception as exc:  # noqa: BLE001
                tc.error = str(exc)

            selected_results.append(tc)

        state["tool_call_results"] = selected_results
        state["fallback_count"] = state.get("fallback_count", 0) + 1

        # If any tool returned new docs, merge them
        new_docs: list[RetrievedDoc] = list(docs)
        for tc in selected_results:
            if tc.result and isinstance(tc.result, dict):
                # Try to extract new chunks from result
                new_chunks = tc.result.get("results", [])
                for c in new_chunks:
                    if isinstance(c, dict) and "chunk_id" in c:
                        new_docs.append(
                            RetrievedDoc(
                                chunk_id=c["chunk_id"],
                                text=c.get("text", ""),
                                source_doc=c.get("source_doc", ""),
                                page_number=c.get("page_number", 1),
                                retrieval_strategy=RetrievalStrategy.ALL,
                            )
                        )

        state["fast_retrieved_docs"] = deduplicate(new_docs)
        return state


# ---------------------------------------------------------------------------
# Node: Generation (Shared — both paths)
# ---------------------------------------------------------------------------

class GenerationNode:
    """
    Shared answer generation node for both slow and fast complex paths.

    Reads from slow_reranked_docs (slow path) or fast_reranked_docs (fast path).
    Sets answer_draft in state (not answer_final — that's set after verification).
    """

    def __init__(self, generator: Any) -> None:
        self.generator = generator

    def run(self, state: GraphState) -> GraphState:
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
            state["answer_draft"] = response.content

            # Build citation list from evidence chunks
            citations = [
                Citation(
                    source_doc=d.source_doc,
                    page_number=d.page_number,
                    text=d.text[:200],
                    chunk_id=d.chunk_id,
                )
                for d in docs
            ]
            state["citations"] = citations
        except Exception as exc:  # noqa: BLE001
            state["node_errors"]["generation"] = str(exc)
            state["answer_draft"] = (
                "I encountered an error generating the answer. Please try again."
            )
        return state


# ---------------------------------------------------------------------------
# Node: Answer Verification
# ---------------------------------------------------------------------------

class VerificationNode:
    """
    Verifies the generated answer against retrieved evidence.

    Runs:
    1. Citation backtracking — ensure all inlined citations are real
    2. Missing data alert — detect if answer references absent data
    3. Sets answer_final = answer_draft if verified, else flags issues
    """

    def __init__(self, verification_tools: Optional[dict[str, callable]] = None) -> None:
        self.verification_tools = verification_tools or {}

    def run(self, state: GraphState) -> GraphState:
        answer = state.get("answer_draft", "")
        docs = state.get("slow_reranked_docs", []) or state.get("fast_reranked_docs", [])
        question = state["question"]

        if not answer or not docs:
            state["answer_final"] = answer
            return state

        # Citation backtracking
        citation_tool = self.verification_tools.get("verification_citation_backtrack")
        if citation_tool:
            evidence_chunks = [
                {"text": d.text, "source_doc": d.source_doc, "page_number": d.page_number}
                for d in docs
            ]
            try:
                cit_result = citation_tool(answer=answer, evidence_chunks=evidence_chunks)
                state["answer_groundedness_score"] = cit_result.get("num_valid", 0) / max(cit_result.get("num_citations_found", 1), 1)
                if cit_result.get("num_issues", 0) > 0:
                    state["fallback_triggered"] = FallbackReason.EVIDENCE_INSUFFICIENT
            except Exception:  # noqa: BLE001
                pass

        state["answer_final"] = answer
        return state


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def deduplicate(docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
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

    Three paths sharing the same entry point and the same final verification node.

    Parameters
    ----------
    router          : QwenRouter (Qwen3-0.6B)
    generator       : QwenGenerator (Qwen3-8B)
    retriever       : HybridRetriever (BGE-M3 + BM25 + RRRF)
    reranker        : BGEReranker
    judge_model     : Any, optional — LLM for evidence checking (default: router)
    verification_tools : dict[str, callable], optional — MCP verification tool functions
    memory_manager  : MemoryManager, optional
    mcp_tool_registry : dict[str, callable], optional — MCP tool registry
    session_id     : str, optional
    """

    def __init__(
        self,
        router: Any,
        generator: Any,
        retriever: Any,
        reranker: Any,
        judge_model: Optional[Any] = None,
        verification_tools: Optional[dict[str, callable]] = None,
        memory_manager: Optional[Any] = None,
        mcp_tool_registry: Optional[dict[str, callable]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.router = router
        self.generator = generator
        self.retriever = retriever
        self.reranker = reranker
        self.judge_model = judge_model or router
        self.verification_tools = verification_tools or {}
        self.memory_manager = memory_manager
        self.mcp_tool_registry = mcp_tool_registry or {}
        self.session_id = session_id or str(uuid.uuid4())

        # Initialize nodes
        self._input = InputNode(session_id=self.session_id)
        self._semantic_router = SemanticRouterNode(router)
        self._direct_gen = DirectGenerationNode(generator, memory_manager)
        self._task_decomp = TaskDecompositionNode(generator)
        self._retrieval_plan = RetrievalPlanningNode()
        self._slow_retrieval = SlowRetrievalNode(retriever)
        self._slow_rerank = SlowRerankNode(reranker)
        self._fast_retrieval = FastRetrievalNode(retriever)
        self._fast_rerank = FastRerankNode(reranker)
        self._evidence_check = EvidenceSufficiencyNode(self.judge_model)
        self._mcp_tool = MCPToolNode(self.mcp_tool_registry)
        self._generation = GenerationNode(generator)
        self._verification = VerificationNode(self.verification_tools)

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        g = StateGraph(GraphState)

        # ── Nodes ──────────────────────────────────────────────────────────
        g.add_node("input", lambda state: self._input.run(state["question"]))
        g.add_node("semantic_router", self._semantic_router.run)
        g.add_node("direct_generation", self._direct_gen.run)
        g.add_node("task_decomposition", self._task_decomp.run)
        g.add_node("retrieval_planning", self._retrieval_plan.run)
        g.add_node("slow_retrieval", self._slow_retrieval.run)
        g.add_node("slow_rerank", self._slow_rerank.run)
        g.add_node("fast_retrieval", self._fast_retrieval.run)
        g.add_node("fast_rerank", self._fast_rerank.run)
        g.add_node("evidence_check", self._evidence_check.run)
        g.add_node("mcp_tool_call", self._mcp_tool.run)
        g.add_node("generation", self._generation.run)
        g.add_node("verification", self._verification.run)

        # ── Entry ───────────────────────────────────────────────────────────
        g.set_entry_point("input")
        g.add_edge("input", "semantic_router")

        # ── Router → paths ─────────────────────────────────────────────────
        g.add_conditional_edges(
            "semantic_router",
            route_after_semantic_router,
            {
                "direct_generation": "direct_generation",
                "slow_retrieval": "slow_retrieval",
                "task_decomposition": "task_decomposition",
            },
        )

        # ── Simple path ─────────────────────────────────────────────────────
        g.add_edge("direct_generation", END)

        # ═══ SLOW PATH ════════════════════════════════════════════════════════
        # retrieval → rerank → generation → verification → end
        g.add_edge("slow_retrieval", "slow_rerank")
        g.add_edge("slow_rerank", "generation")
        g.add_edge("generation", "verification")
        g.add_edge("verification", END)

        # ═══ FAST PATH ════════════════════════════════════════════════════════
        # decomposition → planning → retrieval → rerank → evidence_check
        # → [loop: retrieval | fallback: mcp_tool] → generation → verification → end
        g.add_edge("task_decomposition", "retrieval_planning")
        g.add_edge("retrieval_planning", "fast_retrieval")
        g.add_edge("fast_retrieval", "fast_rerank")
        g.add_edge("fast_rerank", "evidence_check")

        g.add_conditional_edges(
            "evidence_check",
            route_after_evidence_check,
            {
                "fast_retrieval": "fast_retrieval",  # Loop back
                "mcp_tool_call": "mcp_tool_call",    # Fallback to tools
            },
        )

        g.add_edge("mcp_tool_call", "generation")

        return g.compile()

    def run(self, question: str) -> dict[str, Any]:
        """Execute the full pipeline for a question."""
        initial_state = self._input.run(question)
        final_state = self.graph.invoke(initial_state)

        if self.memory_manager is not None:
            self.memory_manager.add_turn("user", question)
            if final_state.get("answer_final"):
                self.memory_manager.add_turn("assistant", final_state["answer_final"])

        return dict(final_state)
