"""
LangGraph Workflow — Two-Path Financial RAG.

Router 输出只有两个值：fast 或 slow。

  Fast 路径（简单问题，单一指标查询）：
    input → truncation → router(fast)
      → fast_retrieval → fast_rerank → generation_fast → END

  Slow 路径（复杂问题，跨期/对比/推理归因）：
    input → truncation → router(slow)
      → task_decomposition → retrieval_planning
      → slow_retrieval → slow_rerank
      → generation_slow → verification → END

无循环，无 MCP 回退。
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from .state import (
    GraphState,
    Route,
    RetrievalStrategy,
    TaskType,
    RetrievedDoc,
    SubTask,
    Citation,
)
from models.qwen_llm import QwenGeneratorFast, QwenGeneratorSlow
from memory.context_manager import TruncationNode, TokenBudgetManager, TruncationPolicy


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _deduplicate(docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
    seen: set[str] = set()
    out: list[RetrievedDoc] = []
    for d in docs:
        if d.chunk_id not in seen:
            seen.add(d.chunk_id)
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# Conditional edge: router → path
# ---------------------------------------------------------------------------

def _route_after_router(state: GraphState) -> str:
    """Router 输出 fast → Fast 路径，slow → Slow 路径。"""
    return "fast_retrieval" if state.get("route") == Route.FAST else "task_decomposition"


# ---------------------------------------------------------------------------
# Node: Input
# ---------------------------------------------------------------------------

class InputNode:
    def __init__(self, session_id: str | None = None) -> None:
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
            fallback_triggered=None,
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
            total_tokens_in_context=0,
            truncation_applied=False,
            budget_threshold=8192,
        )


# ---------------------------------------------------------------------------
# Node: Semantic Router — Qwen3-0.6B
# ---------------------------------------------------------------------------

class SemanticRouterNode:
    """
    Qwen3-0.6B 二分类：fast 或 slow。

    fast  — 单一指标、单一时期、单一公司，无需对比分析
    slow  — 跨期对比、推理归因、多跳综合分析
    """

    ROUTER_PROMPT = """You are a financial query router.

Classify into EXACTLY ONE of two labels:

  fast  — Single-hop factual query.
           ONE indicator, ONE time period, ONE company.
           No comparison, no trend analysis, no reasoning.
           Examples:
           "What was Apple's revenue in FY2024?"
           "What is Tesla's current P/E ratio?"
           "What does the balance sheet show for total assets?"

  slow  — Multi-hop, comparative, cross-period, or reasoning query.
           Requires evidence from multiple sources or time periods,
           or involves calculation, attribution, comparison.
           Examples:
           "Compare Apple's revenue growth vs Microsoft over 3 years"
           "What drove the ROE change across all quarters in 2024?"
           "Analyze risk factors and calculate debt-to-equity ratio"

Reply with ONLY valid JSON: {"route": "fast" | "slow", "confidence": 0.00}"""

    TASK_TYPE_PROMPT = """Classify intent: factual_query | comparative | trend | risk | financial_calc | cross_table | unknown

Reply with only the category name."""

    def __init__(self, router_llm: Any) -> None:
        self.router = router_llm

    def run(self, state: GraphState) -> GraphState:
        import json

        # ── Route ────────────────────────────────────────────────────
        messages = [
            {"role": "system", "content": self.ROUTER_PROMPT},
            {"role": "user", "content": state["question"]},
        ]
        try:
            resp = self.router.chat(messages=messages)
            result = json.loads(resp.content.strip())
            route_val = result.get("route", "slow")
            state["route"] = Route(route_val)
            state["routing_reasoning"] = resp.content
        except Exception:  # noqa: BLE001
            state["route"] = Route.SLOW
            state["routing_reasoning"] = "Router error, defaulting to slow"

        # ── Task type ─────────────────────────────────────────────────
        task_msgs = [
            {"role": "system", "content": self.TASK_TYPE_PROMPT},
            {"role": "user", "content": state["question"]},
        ]
        try:
            t_resp = self.router.chat(messages=task_msgs)
            try:
                state["task_type"] = TaskType(t_resp.content.strip().lower())
            except ValueError:
                state["task_type"] = TaskType.UNKNOWN
        except Exception:  # noqa: BLE001
            state["task_type"] = TaskType.UNKNOWN

        return state


# ---------------------------------------------------------------------------
# Node: Fast Retrieval
# ---------------------------------------------------------------------------

class FastRetrievalNode:
    """
    Fast 路径轻量检索：单一 query，top_k_dense=15 / top_k_bm25=15 → RRRF → top_k_out=6。
    """

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def run(self, state: GraphState) -> GraphState:
        try:
            chunks = self.retriever.retrieve(
                query=state["question"], mode="hybrid", route="fast"
            )
            state["fast_retrieved_docs"] = [
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
        except Exception as exc:  # noqa: BLE001
            state["node_errors"]["fast_retrieval"] = str(exc)
            state["fast_retrieved_docs"] = []
        return state


# ---------------------------------------------------------------------------
# Node: Fast Rerank
# ---------------------------------------------------------------------------

class FastRerankNode:
    def __init__(self, reranker: Any) -> None:
        self.reranker = reranker

    def run(self, state: GraphState) -> GraphState:
        docs = state.get("fast_retrieved_docs", [])
        if not docs:
            state["fast_reranked_docs"] = []
            return state

        try:
            ranked_idx = self.reranker.rerank(
                state["question"], [d.text for d in docs], top_k=len(docs)
            )
            reranked = []
            for idx, score in ranked_idx:
                d = docs[idx]
                d.rerank_score = float(score)
                reranked.append(d)
            state["fast_reranked_docs"] = reranked
        except Exception as exc:  # noqa: BLE001
            state["fast_reranked_docs"] = docs
            state["node_errors"]["fast_rerank"] = str(exc)
        return state


# ---------------------------------------------------------------------------
# Node: Task Decomposition (Slow)
# ---------------------------------------------------------------------------

class TaskDecompositionNode:
    """复杂问题拆解为 N 个独立子任务，并行检索。"""

    PLANNER_PROMPT = """Break this complex question into independent sub-tasks.

Output a JSON array only:
[
  {"task_id": "t1", "description": "...", "strategy": "text|table|figure|all", "key_terms": ["..."]},
  ...
]

Rules:
- strategy "table" if numeric table data needed
- strategy "text" if qualitative (risk factors, discussion)
- strategy "figure" if chart data
- key_terms: all financial terms, company names, years"""

    def __init__(self, generator: Any) -> None:
        self.generator = generator

    def run(self, state: GraphState) -> GraphState:
        import json

        messages = [
            {"role": "system", "content": self.PLANNER_PROMPT},
            {"role": "user", "content": state["question"]},
        ]
        try:
            resp = self.generator.chat(messages=messages)
            raw = resp.content.strip()
            raw_tasks = json.loads(raw) if raw.startswith("[") else []
            state["sub_tasks"] = [
                SubTask(
                    task_id=t["task_id"],
                    description=t["description"],
                    strategy=RetrievalStrategy(t.get("strategy", "all")),
                    key_terms=t.get("key_terms", []),
                )
                for t in raw_tasks
            ]
        except Exception as exc:  # noqa: BLE001
            state["node_errors"]["task_decomposition"] = str(exc)
            state["sub_tasks"] = [
                SubTask(
                    task_id="t0",
                    description=state["question"],
                    strategy=RetrievalStrategy.ALL,
                    key_terms=[],
                )
            ]
        return state


# ---------------------------------------------------------------------------
# Node: Retrieval Planning (Slow)
# ---------------------------------------------------------------------------

class RetrievalPlanningNode:
    """分析子任务，确定主检索策略，构建每个子任务的 query。"""

    def run(self, state: GraphState) -> GraphState:
        sub_tasks = state.get("sub_tasks", [])
        if not sub_tasks:
            state["retrieval_queries"] = [state["question"]]
            state["retrieval_strategy"] = RetrievalStrategy.ALL
            return state

        # 统计各策略数量
        counts: dict[RetrievalStrategy, int] = {}
        for t in sub_tasks:
            s = RetrievalStrategy(t.strategy)
            counts[s] = counts.get(s, 0) + 1

        # 多数策略为主策略（all 除外）
        dominant = max(
            (s for s in counts if s != RetrievalStrategy.ALL),
            key=counts.get,
            default=RetrievalStrategy.ALL,
        )
        if counts.get(RetrievalStrategy.ALL) and len(counts) == 1:
            dominant = RetrievalStrategy.ALL

        state["retrieval_strategy"] = dominant

        # 构建每个子任务的 query
        queries = []
        for t in sub_tasks:
            q = t.description
            if t.key_terms:
                q = f"{q} ({', '.join(t.key_terms)})"
            queries.append(q)

        state["retrieval_queries"] = queries
        return state


# ---------------------------------------------------------------------------
# Node: Slow Retrieval
# ---------------------------------------------------------------------------

class SlowRetrievalNode:
    """
    Slow 路径多 query 检索：route=slow → final_k=8，更丰富的证据基础。
    """

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def run(self, state: GraphState) -> GraphState:
        queries = state.get("retrieval_queries", [])
        if not queries:
            queries = [state["question"]]

        try:
            chunks = self.retriever.retrieve_multi_query(
                queries=queries, mode="hybrid", route="slow"
            )
            state["slow_retrieved_docs"] = _deduplicate([
                RetrievedDoc(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    source_doc=c.source_doc,
                    page_number=c.page_number,
                    retrieval_strategy=state.get("retrieval_strategy", RetrievalStrategy.ALL),
                    dense_score=getattr(c, "dense_score", 0.0),
                    sparse_score=getattr(c, "sparse_score", 0.0),
                )
                for c in chunks
            ])
        except Exception as exc:  # noqa: BLE001
            state["node_errors"]["slow_retrieval"] = str(exc)
            state["slow_retrieved_docs"] = []
        return state


# ---------------------------------------------------------------------------
# Node: Slow Rerank
# ---------------------------------------------------------------------------

class SlowRerankNode:
    def __init__(self, reranker: Any) -> None:
        self.reranker = reranker

    def run(self, state: GraphState) -> GraphState:
        docs = state.get("slow_retrieved_docs", [])
        if not docs:
            state["slow_reranked_docs"] = []
            return state

        try:
            ranked_idx = self.reranker.rerank(
                state["question"], [d.text for d in docs], top_k=len(docs)
            )
            reranked = []
            for idx, score in ranked_idx:
                d = docs[idx]
                d.rerank_score = float(score)
                reranked.append(d)
            state["slow_reranked_docs"] = reranked
        except Exception as exc:  # noqa: BLE001
            state["slow_reranked_docs"] = docs
            state["node_errors"]["slow_rerank"] = str(exc)
        return state


# ---------------------------------------------------------------------------
# Node: Generation — Fast
# ---------------------------------------------------------------------------

class GenerationNodeFast:
    """
    Fast 路径生成器（Qwen3-8B）：
    enable_thinking=False, temperature=0.1, max_new_tokens=512, chunks=4
    """

    def __init__(self, gen_fast: QwenGeneratorFast) -> None:
        self._gen = gen_fast

    def run(self, state: GraphState) -> GraphState:
        docs = state.get("fast_reranked_docs", [])
        evidence = [
            {"text": d.text, "source": d.source_doc, "page": d.page_number}
            for d in docs
        ]

        try:
            resp = self._gen.generate_answer(
                query=state["question"],
                evidence_chunks=evidence,
            )
            state["answer_final"] = resp.content
            state["citations"] = [
                Citation(
                    source_doc=d.source_doc,
                    page_number=d.page_number,
                    text=d.text[:200],
                    chunk_id=d.chunk_id,
                )
                for d in docs
            ]
        except Exception as exc:  # noqa: BLE001
            state["node_errors"]["generation_fast"] = str(exc)
            state["answer_final"] = "Error generating answer. Please try again."
        return state


# ---------------------------------------------------------------------------
# Node: Generation — Slow
# ---------------------------------------------------------------------------

class GenerationNodeSlow:
    """
    Slow 路径生成器（Qwen3-8B）：
    enable_thinking=True, temperature=0.3, max_new_tokens=1024, chunks=6
    """

    def __init__(self, gen_slow: QwenGeneratorSlow) -> None:
        self._gen = gen_slow

    def run(self, state: GraphState) -> GraphState:
        docs = state.get("slow_reranked_docs", [])
        evidence = [
            {"text": d.text, "source": d.source_doc, "page": d.page_number}
            for d in docs
        ]

        try:
            resp = self._gen.generate_answer(
                query=state["question"],
                evidence_chunks=evidence,
            )
            state["answer_draft"] = resp.content
            state["citations"] = [
                Citation(
                    source_doc=d.source_doc,
                    page_number=d.page_number,
                    text=d.text[:200],
                    chunk_id=d.chunk_id,
                )
                for d in docs
            ]
        except Exception as exc:  # noqa: BLE001
            state["node_errors"]["generation_slow"] = str(exc)
            state["answer_draft"] = "Error generating answer. Please try again."
        return state


# ---------------------------------------------------------------------------
# Node: Verification (Slow)
# ---------------------------------------------------------------------------

class VerificationNode:
    """慢路径答案验据：引用回溯 + 一致性检查。"""

    def __init__(self, verification_tools: dict[str, Any] | None = None) -> None:
        self._tools = verification_tools or {}

    def run(self, state: GraphState) -> GraphState:
        answer = state.get("answer_draft", "")
        docs = state.get("slow_reranked_docs", [])

        if not answer or not docs:
            state["answer_final"] = answer
            return state

        # 引用回溯
        cit_tool = self._tools.get("verification_citation_backtrack")
        if cit_tool:
            evidence = [
                {
                    "text": d.text,
                    "source_doc": d.source_doc,
                    "page_number": d.page_number,
                }
                for d in docs
            ]
            try:
                result = cit_tool(answer=answer, evidence_chunks=evidence)
                n_found = result.get("num_citations_found", 0)
                n_valid = result.get("num_valid", 0)
                if n_found > 0:
                    state["answer_groundedness_score"] = n_valid / n_found
            except Exception:  # noqa: BLE001
                pass

        state["answer_final"] = state.get("answer_draft", answer)
        return state


# ---------------------------------------------------------------------------
# TruncationNode is imported from memory.context_manager
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Workflow Builder
# ---------------------------------------------------------------------------

class FinancialRAGWorkflow:
    """
    LangGraph 两路径 RAG 工作流。

    Fast 路径：input → truncation → router → fast_retrieval → fast_rerank → generation_fast → END
    Slow 路径：input → truncation → router → task_decomp → retrieval_plan
                                → slow_retrieval → slow_rerank
                                → generation_slow → verification → END
    """

    def __init__(
        self,
        router: Any,
        retriever: Any,
        reranker: Any,
        gen_fast: QwenGeneratorFast | None = None,
        gen_slow: QwenGeneratorSlow | None = None,
        verification_tools: dict[str, Any] | None = None,
        budget_manager: TokenBudgetManager | None = None,
        session_id: str | None = None,
    ) -> None:
        self.router = router
        self.retriever = retriever
        self.reranker = reranker
        self.verification_tools = verification_tools or {}
        self.session_id = session_id or str(uuid.uuid4())

        # Token budget manager（所有节点共享同一个实例引用）
        self.budget_manager = budget_manager or TokenBudgetManager(
            policy=TruncationPolicy(budget=8192, k_keep=4, compression_model="qwen2.5-1.5b-instruct")
        )

        # Fast / Slow 生成器
        self._gen_fast = gen_fast or QwenGeneratorFast(config=None)
        self._gen_slow = gen_slow or QwenGeneratorSlow(config=None)

        # 初始化所有节点
        self._input = InputNode(session_id=self.session_id)
        self._truncation = TruncationNode(self.budget_manager)
        self._router = SemanticRouterNode(router)
        self._fast_retrieval = FastRetrievalNode(retriever)
        self._fast_rerank = FastRerankNode(reranker)
        self._gen_fast_node = GenerationNodeFast(self._gen_fast)
        self._task_decomp = TaskDecompositionNode(self._gen_slow)
        self._retrieval_plan = RetrievalPlanningNode()
        self._slow_retrieval = SlowRetrievalNode(retriever)
        self._slow_rerank = SlowRerankNode(reranker)
        self._gen_slow_node = GenerationNodeSlow(self._gen_slow)
        self._verification = VerificationNode(self.verification_tools)

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        g = StateGraph(GraphState)

        # ── 节点 ──────────────────────────────────────────────────
        g.add_node("input", lambda state: self._input.run(state["question"]))
        g.add_node("truncation", self._truncation.run)
        g.add_node("semantic_router", self._router.run)

        # Fast 路径
        g.add_node("fast_retrieval", self._fast_retrieval.run)
        g.add_node("fast_rerank", self._fast_rerank.run)
        g.add_node("generation_fast", self._gen_fast_node.run)

        # Slow 路径
        g.add_node("task_decomposition", self._task_decomp.run)
        g.add_node("retrieval_planning", self._retrieval_plan.run)
        g.add_node("slow_retrieval", self._slow_retrieval.run)
        g.add_node("slow_rerank", self._slow_rerank.run)
        g.add_node("generation_slow", self._gen_slow_node.run)
        g.add_node("verification", self._verification.run)

        # ── 入口 ──────────────────────────────────────────────────
        g.set_entry_point("input")
        g.add_edge("input", "truncation")
        g.add_edge("truncation", "semantic_router")

        # ── 路由分叉 ───────────────────────────────────────────────
        g.add_conditional_edges(
            "semantic_router",
            _route_after_router,
            {
                "fast_retrieval": "fast_retrieval",
                "task_decomposition": "task_decomposition",
            },
        )

        # ═══ Fast 路径 ══════════════════════════════════════════════
        # fast_retrieval → fast_rerank → generation_fast → END
        g.add_edge("fast_retrieval", "fast_rerank")
        g.add_edge("fast_rerank", "generation_fast")
        g.add_edge("generation_fast", END)

        # ═══ Slow 路径 ═════════════════════════════════════════════
        # task_decomp → retrieval_plan → slow_retrieval → slow_rerank
        # → generation_slow → verification → END
        g.add_edge("task_decomposition", "retrieval_planning")
        g.add_edge("retrieval_planning", "slow_retrieval")
        g.add_edge("slow_retrieval", "slow_rerank")
        g.add_edge("slow_rerank", "generation_slow")
        g.add_edge("generation_slow", "verification")
        g.add_edge("verification", END)

        return g.compile()

    def run(self, question: str) -> dict[str, Any]:
        """执行完整流程，返回最终状态。"""
        initial = self._input.run(question)
        self.budget_manager.add_turn("user", question)

        final = self.graph.invoke(initial)

        if final.get("answer_final"):
            self.budget_manager.add_turn("assistant", final["answer_final"])

        msgs, total_tokens = self.budget_manager.get_context_for_prompt()
        final["short_term_context"] = msgs
        final["total_tokens_in_context"] = total_tokens

        return dict(final)
