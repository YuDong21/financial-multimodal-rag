"""
run.py — Main entry point for the financial-multimodal-rag system.

Usage
-----
    # Single question
    python run.py "What was Apple's total revenue in FY2024?"

    # Interactive mode
    python run.py --interactive

    # With a PDF document already ingested
    python run.py "Compare Apple's revenue vs. Microsoft over 3 years" \\
        --collection my_financial_reports

    # With config override
    python run.py "What is the ROE trend?" \\
        --config-override generator_slow.max_new_tokens=2048

Prerequisites
-------------
1. Install dependencies:
    pip install -r requirements.txt

2. Set API keys:
    export DASHSCOPE_API_KEY="your-dashscope-api-key"

3. Ingest PDF documents (see ingest.py):
    python ingest.py --pdf /path/to/annual_report_2024.pdf \\
                     --collection financial_reports \\
                     --embed

4. Run queries:
    python run.py "What was Apple's revenue growth rate?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_environment() -> None:
    """Validate environment and load configuration."""
    missing_keys = []
    if not os.environ.get("DASHSCOPE_API_KEY"):
        missing_keys.append("DASHSCOPE_API_KEY")

    if missing_keys:
        print(
            f"[WARNING] Missing environment variables: {', '.join(missing_keys)}",
            file=sys.stderr,
        )
        print("Set them with: export DASHSCOPE_API_KEY='your-key'", file=sys.stderr)

    # Add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)


# ---------------------------------------------------------------------------
# Collection Store (simple file-based vector store)
# ---------------------------------------------------------------------------

class CollectionStore:
    """
    Simple file-based collection store for document chunks.

    In production, replace with Milvus, Qdrant, or ChromaDB.
    Stores chunks as JSON lines: {chunk_id, text, source_doc, page_number, token_count}
    """

    def __init__(self, store_dir: str = "./data/collections") -> None:
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

    def save(self, collection_name: str, chunks: list[dict]) -> None:
        """Save chunks to a collection file."""
        path = os.path.join(self.store_dir, f"{collection_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"[CollectionStore] Saved {len(chunks)} chunks to {path}")

    def load(self, collection_name: str) -> list[dict]:
        """Load chunks from a collection file."""
        path = os.path.join(self.store_dir, f"{collection_name}.jsonl")
        if not os.path.exists(path):
            return []
        chunks = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        print(f"[CollectionStore] Loaded {len(chunks)} chunks from {path}")
        return chunks

    def exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        path = os.path.join(self.store_dir, f"{collection_name}.jsonl")
        return os.path.exists(path)


# ---------------------------------------------------------------------------
# RAG Pipeline Initialization
# ---------------------------------------------------------------------------

def init_rag_pipeline(collection_name: str) -> tuple[Any, Any, Any, Any]:
    """
    Initialize the full RAG pipeline.

    Returns (workflow, retriever, store)
    """
    from config import get_config
    from models.qwen_llm import QwenRouter, QwenGeneratorFast, QwenGeneratorSlow, QwenSummarizer
    from retrieval.hybrid_retriever import HybridRetriever
    from graph.workflow import FinancialRAGWorkflow
    from memory.context_manager import TokenBudgetManager, TruncationPolicy, SemanticTruncator

    cfg = get_config()

    # Initialize models
    print("[Init] Initializing models...")
    router = QwenRouter(config=cfg.router)
    gen_fast = QwenGeneratorFast(config=cfg.generator_fast)
    gen_slow = QwenGeneratorSlow(config=cfg.generator_slow)
    summarizer = QwenSummarizer(config=cfg.summarizer)

    # Initialize token budget manager
    budget_manager = TokenBudgetManager(
        policy=TruncationPolicy(
            budget=cfg.token_budget.budget,
            k_keep=cfg.token_budget.k_keep,
            compression_model=cfg.token_budget.summarizer_model,
        ),
        truncator=SemanticTruncator(summarizer_api=summarizer),
    )

    # Initialize retriever
    retriever = HybridRetriever(config=cfg)

    # Load collection into retriever
    store = CollectionStore()
    chunks = store.load(collection_name)
    if chunks:
        retriever.load_collection(chunks)
        print(f"[Init] Loaded collection '{collection_name}': {len(chunks)} chunks")
    else:
        print(f"[WARNING] Collection '{collection_name}' not found. Run ingest.py first.")

    # Initialize workflow
    workflow = FinancialRAGWorkflow(
        router=router,
        retriever=retriever,
        reranker=retriever.reranker,  # Reuse reranker from retriever
        budget_manager=budget_manager,
        session_id=f"rag_session_{collection_name}",
    )

    return workflow, retriever, store


# ---------------------------------------------------------------------------
# Single Query Run
# ---------------------------------------------------------------------------

def run_query(question: str, collection_name: str) -> dict[str, Any]:
    """Run a single query through the RAG pipeline."""
    workflow, _, _ = init_rag_pipeline(collection_name)

    print(f"\n[Query] {question}")
    print("-" * 60)

    result = workflow.run(question)

    # Print answer
    print(f"[Answer]\n{result.get('answer_final', result.get('answer_draft', 'No answer'))}")

    # Print route
    route = result.get("route", "unknown")
    print(f"\n[Route] {route}")

    # Print citations
    citations = result.get("citations", [])
    if citations:
        print(f"[Citations] ({len(citations)} sources)")
        for i, cit in enumerate(citations[:5], 1):
            print(f"  [{i}] {cit.get('source_doc', '?')}, page {cit.get('page_number', '?')}")

    # Print token budget status
    total_tokens = result.get("total_tokens_in_context", 0)
    truncation = result.get("truncation_applied", False)
    print(f"\n[Token Budget] total={total_tokens}, truncation_applied={truncation}")

    # Print memory summary if active
    summary = result.get("memory_summary")
    if summary:
        print(f"[Memory Summary] {summary[:100]}...")

    return result


# ---------------------------------------------------------------------------
# Interactive Mode
# ---------------------------------------------------------------------------

def interactive_mode(collection_name: str) -> None:
    """Run an interactive REPL."""
    workflow, _, _ = init_rag_pipeline(collection_name)

    print("\n" + "=" * 60)
    print("  financial-multimodal-rag — Interactive Mode")
    print("  Type 'exit' or 'quit' to stop")
    print("  Type 'reset' to clear conversation history")
    print("  Type 'stats' to see token budget stats")
    print("=" * 60 + "\n")

    session_history: list[dict[str, str]] = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if question.lower() == "reset":
            session_history = []
            print("[Context reset]")
            continue

        if question.lower() == "stats":
            print(f"[Stats] Turns in history: {len(session_history)}")
            continue

        if not question:
            continue

        # Run with history
        result = workflow.run_with_history(question, session_history)

        # Update history
        session_history.append({"role": "user", "content": question})
        if result.get("answer_final"):
            session_history.append({"role": "assistant", "content": result["answer_final"]})

        # Print
        print(f"\nBot: {result.get('answer_final', result.get('answer_draft', '(no answer)'))}")

        route = result.get("route", "?")
        citations = result.get("citations", [])
        print(f"  [route={route}, citations={len(citations)}]\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="financial-multimodal-rag — Query financial reports with RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (omit for interactive mode)",
    )
    parser.add_argument(
        "--collection", "-c",
        default="financial_reports",
        help="Collection name to query (default: financial_reports)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch interactive REPL",
    )
    parser.add_argument(
        "--config-override",
        nargs="+",
        help="Override config values, e.g. --config-override generator_slow.max_new_tokens=2048",
    )
    parser.add_argument(
        "--show-retrieval",
        action="store_true",
        help="Show retrieved chunks in the result",
    )
    parser.add_argument(
        "--show-state",
        action="store_true",
        help="Show full GraphState in the result",
    )

    args = parser.parse_args()

    # Apply config overrides
    if args.config_override:
        from config import override_config
        overrides = {}
        for override in args.config_override:
            section, key_value = override.split(".", 1)
            key, value = key_value.split("=", 1)
            # Try to convert value to int/float/bool
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
            overrides.setdefault(section, {})[key] = value
        override_config(**overrides)
        print(f"[Config] Overrides applied: {overrides}")

    setup_environment()

    if args.interactive or args.question is None:
        interactive_mode(args.collection)
    else:
        result = run_query(args.question, args.collection)

        if args.show_retrieval:
            print("\n[Retrieved Chunks]")
            for i, doc in enumerate(result.get("slow_reranked_docs", result.get("fast_reranked_docs", [])), 1):
                print(f"  [{i}] {doc.text[:200]}...")

        if args.show_state:
            print("\n[Full GraphState]")
            safe_state = {k: v for k, v in result.items() if k != "error_message"}
            print(json.dumps(safe_state, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
