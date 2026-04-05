"""
run.py — 交互式金融问答 RAG 系统。

支持两种模式：
  1. 单次问答：python run.py "你的问题"
  2. 交互 REPL：python run.py --interactive

两条路径自动选择：
  Fast 路径 — 简单单一指标查询（light retrieval）
  Slow 路径 — 复杂跨期对比 / 推理归因（完整 retrieval pipeline）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

COLLECTION_DIR = os.path.join(PROJECT_ROOT, "data", "collections")


# ---------------------------------------------------------------------------
# Collection Store
# ---------------------------------------------------------------------------

class CollectionStore:
    """
    基于 JSONL 的轻量 Collection 存储。

    每个 collection 对应 data/collections/{name}.jsonl，
    每行一个 JSON chunk。
    """

    def __init__(self, collection_dir: str = COLLECTION_DIR) -> None:
        self.collection_dir = collection_dir

    def get_chunk_count(self, name: str) -> int:
        path = os.path.join(self.collection_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            return 0
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def load_chunks(self, name: str) -> list[dict]:
        path = os.path.join(self.collection_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            return []
        chunks: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    def list_collections(self) -> list[str]:
        if not os.path.exists(self.collection_dir):
            return []
        return [
            Path(f).stem
            for f in os.listdir(self.collection_dir)
            if f.endswith(".jsonl")
        ]


# ---------------------------------------------------------------------------
# Result Printer
# ---------------------------------------------------------------------------

def print_result(
    question: str,
    result: dict,
    show_retrieval: bool = False,
    show_state: bool = False,
) -> None:
    route = result.get("route", "?")
    answer = result.get("answer_final", "(无答案)")
    citations = result.get("citations", [])
    errors = result.get("node_errors", {})

    print(f"\n{'='*60}")
    print(f"[{route.upper()} PATH]")
    print(f"{'='*60}")
    print(f"\nQ: {question}\n")
    print(f"A: {answer}\n")

    if citations:
        print(f"📌 Citations ({len(citations)}):")
        seen = set()
        for cit in citations:
            key = f"{cit.source_doc}:p{cit.page_number}"
            if key in seen:
                continue
            seen.add(key)
            preview = cit.text[:80].replace("\n", " ")
            print(f"  · {cit.source_doc} (p{cit.page_number}): {preview}...")

    if show_retrieval:
        docs = result.get("slow_reranked_docs", result.get("fast_reranked_docs", []))
        if docs:
            print(f"\n📄 Retrieved Chunks ({len(docs)}):")
            for i, doc in enumerate(docs, 1):
                preview = doc.text[:100].replace("\n", " ")
                score = getattr(doc, "rerank_score", None)
                score_str = f" [score={score:.3f}]" if score is not None else ""
                print(f"  {i}. {doc.source_doc} (p{doc.page_number}){score_str}")
                print(f"     {preview}...")

    if show_state:
        print(f"\n🔍 State:")
        print(f"  route               : {route}")
        print(f"  task_type           : {result.get('task_type')}")
        print(f"  total_tokens_in_context: {result.get('total_tokens_in_context', 0)}")
        print(f"  truncation_applied  : {result.get('truncation_applied')}")
        print(f"  routing_reasoning  : {str(result.get('routing_reasoning', ''))[:120]}")

    if errors:
        print(f"\n⚠️  Node errors:")
        for node, err in errors.items():
            print(f"  {node}: {err}")

    print(f"\n{'='*60}\n")


def print_history(history: list[dict]) -> None:
    if not history:
        return
    print(f"\n{'='*60}")
    print(f"Session History ({len(history)} turns)")
    print(f"{'='*60}")
    for i, turn in enumerate(history[-10:], 1):
        print(f"\n[Turn {turn.get('turn', i)}] {turn.get('time', '')}")
        print(f"  Q: {turn.get('question', '')[:80]}")
        ans = turn.get("answer", "")
        print(f"  A: {ans[:120]}{'...' if len(ans) > 120 else ''}")
        print(f"  route: {turn.get('route', '?')}")
    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# RAG Pipeline Init
# ---------------------------------------------------------------------------

def init_rag_pipeline(collection_name: str, config_overrides: dict | None = None):
    """初始化 RAG pipeline（lazy import）。"""
    from config import get_config, override_config
    from models.qwen_llm import QwenRouter, QwenGeneratorFast, QwenGeneratorSlow
    from retrieval.hybrid_retriever import HybridRetriever
    from memory.context_manager import TokenBudgetManager, TruncationPolicy
    from graph.workflow import FinancialRAGWorkflow

    if config_overrides:
        override_config(config_overrides)

    cfg = get_config()

    print(f"[run] Initializing RAG pipeline...")
    print(f"[run] Collection : {collection_name}")
    print(f"[run] Router LLM : {cfg.router.model}")
    print(f"[run] Gen Fast   : {cfg.generator_fast.model} (thinking={cfg.generator_fast.enable_thinking})")
    print(f"[run] Gen Slow   : {cfg.generator_slow.model} (thinking={cfg.generator_slow.enable_thinking})")

    # 检查 collection
    store = CollectionStore()
    chunk_count = store.get_chunk_count(collection_name)
    if chunk_count == 0:
        print(f"[run] ⚠️  Collection '{collection_name}' is empty or not found.")
        print(f"[run]   Run: python data_ingest.py --pdf-dir ./data/raw/ -c {collection_name}")
        collections = store.list_collections()
        if collections:
            print(f"[run]   Available collections: {', '.join(collections)}")
        sys.exit(1)
    print(f"[run] Collection  : {collection_name} ({chunk_count} chunks)")

    # LLM
    router = QwenRouter(config=cfg.router)
    gen_fast = QwenGeneratorFast(config=cfg.generator_fast)
    gen_slow = QwenGeneratorSlow(config=cfg.generator_slow)

    # Retriever
    retriever = HybridRetriever(config=cfg)
    reranker = retriever.reranker  # type: ignore

    # Token budget
    budget_manager = TokenBudgetManager(
        policy=TruncationPolicy(
            budget=cfg.token_budget.budget,
            k_keep=cfg.token_budget.k_keep,
            compression_model=cfg.summarizer.model,
        )
    )

    # Workflow
    workflow = FinancialRAGWorkflow(
        router=router,
        retriever=retriever,
        reranker=reranker,
        gen_fast=gen_fast,
        gen_slow=gen_slow,
        budget_manager=budget_manager,
    )

    return workflow


# ---------------------------------------------------------------------------
# Single Query
# ---------------------------------------------------------------------------

def run_single_query(
    question: str,
    collection_name: str,
    config_overrides: dict | None = None,
    show_retrieval: bool = False,
    show_state: bool = False,
) -> dict:
    workflow = init_rag_pipeline(collection_name, config_overrides)
    result = workflow.run(question)
    print_result(question, result, show_retrieval=show_retrieval, show_state=show_state)
    return result


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def run_interactive(
    collection_name: str,
    config_overrides: dict | None = None,
    show_retrieval: bool = False,
    show_state: bool = False,
) -> None:
    workflow = init_rag_pipeline(collection_name, config_overrides)

    session_id = str(uuid.uuid4())[:8]
    history: list[dict] = []
    turn = 0

    print(f"\n{'#'*60}")
    print(f"#  Financial Multimodal RAG — Interactive Mode")
    print(f"#  Session : {session_id}")
    print(f"#  Collection: {collection_name}")
    print(f"#  Commands: history / state / clear / exit")
    print(f"{'#'*60}")

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # 命令
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "history":
            print_history(history)
            continue
        if user_input.lower() == "clear":
            history.clear()
            print("[history cleared]")
            continue
        if user_input.lower() == "state":
            show_state = not show_state
            print(f"[show_state={'on' if show_state else 'off'}]")
            continue

        turn += 1
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[thinking...]\n")

        try:
            result = workflow.run(user_input)
            print_result(
                user_input, result,
                show_retrieval=show_retrieval,
                show_state=show_state,
            )
            history.append({
                "turn": turn,
                "time": ts,
                "question": user_input,
                "answer": result.get("answer_final", ""),
                "route": str(result.get("route", "?")),
                "citations_count": len(result.get("citations", [])),
            })
        except Exception as exc:  # noqa: BLE001
            print(f"\n[ERROR] {exc}")
            if "API" in str(exc):
                print("可能是 API Key 问题，请检查 DASHSCOPE_API_KEY")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_overrides(overrides: list[str] | None) -> dict:
    """解析 --config-override key=value 列表。"""
    if not overrides:
        return {}
    result = {}
    for item in overrides:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        # 尝试转换为 int / float / bool
        if val in ("true", "True"):
            val = True
        elif val in ("false", "False"):
            val = False
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        result[key.strip()] = val
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="financial-multimodal-rag 交互式问答",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "question", nargs="?", help="要查询的问题（省略则进入 REPL 模式）",
    )
    parser.add_argument(
        "-c", "--collection", default="financial_reports",
        help="Collection 名称 (default: financial_reports)",
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="进入交互式 REPL 模式",
    )
    parser.add_argument(
        "--config-override", nargs="+",
        help="覆盖 config 参数，如 --config-override generator_slow.max_new_tokens=2048",
    )
    parser.add_argument(
        "--show-retrieval", action="store_true",
        help="显示检索到的 chunks",
    )
    parser.add_argument(
        "--show-state", action="store_true",
        help="显示 GraphState 详情（路由/截断/令牌数等）",
    )

    args = parser.parse_args()

    # 解析 config overrides
    overrides = parse_overrides(args.config_override)

    # 模式判断
    if args.interactive or args.question is None:
        run_interactive(
            collection_name=args.collection,
            config_overrides=overrides if overrides else None,
            show_retrieval=args.show_retrieval,
            show_state=args.show_state,
        )
    else:
        run_single_query(
            question=args.question,
            collection_name=args.collection,
            config_overrides=overrides if overrides else None,
            show_retrieval=args.show_retrieval,
            show_state=args.show_state,
        )


if __name__ == "__main__":
    main()
