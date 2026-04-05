"""
memory: Context window management with token budget and semantic truncation.

Components
----------
TokenBudgetManager   : Token counter + sliding window + semantic truncation logic
TokenCounter         : Standalone tiktoken-based token counter utility
SemanticTruncator    : Qwen2.5-1.5B-based conversation compressor
TruncationNode       : LangGraph node for pre-router budget enforcement
TruncationPolicy     : Configuration for truncation thresholds and K-keep

Token Budget System
-------------------
TokenBudgetManager runs BEFORE SemanticRouterNode on every request:

  1. Check: total_tokens >= budget_threshold?
  2. If YES → apply sliding window:
       - Keep recent K turns verbatim
       - Older turns → Qwen2.5-1.5B semantic compression
       - Summary stored in GraphState["memory_summary"]
  3. Result: downstream nodes see a bounded, context-rich state

GraphState Global Access Pattern
---------------------------------
In LangGraph, the GraphState dict is the SINGLE source of truth.
It is passed by reference (not copied) to every node in the graph.
All nodes read from and write to the SAME dict object, so writes
in one node are immediately visible to all downstream nodes.

Within a node, access is direct:
    state["total_tokens_in_context"]   # read
    state["memory_summary"] = "..."     # write

Across the graph, state fields accumulate naturally:
    short_term_context    → grows with each turn, truncated when budget exceeded
    memory_summary        → set by TruncationNode, read by all downstream nodes
    total_tokens_in_context → updated by TruncationNode, read by router/generation
"""

from .context_manager import (
    TokenBudgetManager,
    TokenCounter,
    SemanticTruncator,
    TruncationNode,
    TruncationPolicy,
    count_tokens,
    count_messages_tokens,
)

__all__ = [
    "TokenBudgetManager",
    "TokenCounter",
    "SemanticTruncator",
    "TruncationNode",
    "TruncationPolicy",
    "count_tokens",
    "count_messages_tokens",
]
