"""
Context Manager — Token Budget & Semantic Truncation.

Implements a token budget management system that runs BEFORE the router on every request:

1. Token Budget Tracking
   - A running token counter lives in the GraphState (persists across turns)
   - Before the SemanticRouterNode, a TruncationNode checks: total_tokens >= budget_threshold?
   - If YES → trigger the truncation mechanism

2. Sliding Window + Semantic Truncation
   - When budget is exceeded, apply sliding window strategy:
     - Force KEEP the most recent K rounds of Q&A in FULL (raw text)
     - OLDER history that gets pushed out → NOT discarded
     - Instead: compress via Qwen2.5-1.5B into a "semantic summary"
     - The summary is stored in GraphState["memory_summary"]
   - Result: recent turns are verbatim, older context is preserved in compressed form

3. GraphState as Global Parameter
   - The GraphState dict is the SINGLE source of truth for the entire graph execution
   - It is NOT copied or cloned — LangGraph passes the same dict reference to every node
   - All nodes read from and write to the SAME dict object
   - This means token counts, summaries, and context accumulate naturally across nodes
   - The dict key `total_tokens_in_context` is the live running counter

Design Rationale
----------------
Why not just let the LLM handle context overflow?
  - LLMs suffer from "lost in the middle" — important facts at the edges of
    a long context are frequently ignored
  - Pre-aggregation lets us preserve the full factual record while staying
    within the LLM's effective context window
  - Semantic summaries capture the "gist" of old conversations, preserving
    context that would otherwise be permanently lost

Why Qwen2.5-1.5B for compression?
  - Small enough to be fast and cheap
  - Large enough to preserve factual relationships in compression
  - Can be run locally via Ollama or similar

Usage
-----
    >>> from memory.context_manager import TokenBudgetManager, TruncationNode
    >>> manager = TokenBudgetManager(
    ...     budget=8192,
    ...     k_keep=4,               # keep last 4 turns fully
    ...     summarizer_model="qwen2.5-1.5b",
    ... )
    >>> truncation_node = TruncationNode(manager)
    >>> # In workflow: input → truncation → semantic_router → ...
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import tiktoken

# ---------------------------------------------------------------------------
# Token Counter
# ---------------------------------------------------------------------------

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in a text string using tiktoken.

    Uses cl100k_base (GPT-4 / Qwen training encoding) by default.
    """
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def count_messages_tokens(messages: list[dict[str, str]], encoding_name: str = "cl100k_base") -> int:
    """
    Count total tokens in a list of {role, content} messages.

    Accounts for role-prefix overhead (≈4 tokens per message).
    """
    enc = tiktoken.get_encoding(encoding_name)
    total = 0
    for msg in messages:
        # Role prefix overhead
        total += 4
        total += len(enc.encode(msg.get("content", "")))
    return total


# ---------------------------------------------------------------------------
# Truncation Policy
# ---------------------------------------------------------------------------

@dataclass
class TruncationPolicy:
    """
    Configuration for the truncation mechanism.

    Attributes
    ----------
    budget : int, default 8192
        Total token budget for conversation context.
        When total_tokens >= budget, truncation activates.
    k_keep : int, default 4
        Number of RECENT full turns to keep verbatim after truncation.
        These are the newest user/assistant exchanges, never compressed.
    compression_model : str, default "qwen2.5-1.5b-instruct"
        Model name for semantic summarization.
        Must be available via DashScope API or local Ollama.
    compress_when_ratio : float, default 0.85
        Trigger compression early when context reaches this fraction of budget.
        Avoids waiting until the budget is fully exhausted.
    """

    budget: int = 8192
    k_keep: int = 4
    compression_model: str = "qwen2.5-1.5b-instruct"
    compress_when_ratio: float = 0.85

    @property
    def soft_threshold(self) -> int:
        """Early warning threshold."""
        return int(self.budget * self.compress_when_ratio)


# ---------------------------------------------------------------------------
# Semantic Truncator
# ---------------------------------------------------------------------------

class SemanticTruncator:
    """
    Compresses older conversation turns into a dense semantic summary
    using Qwen2.5-1.5B.

    Preserves:
    - All factual claims and numbers
    - The overall topic and intent
    - Any unresolved sub-questions / pending tasks
    - Key entities (company names, metrics, time periods)

    Drops:
    - Conversational filler and pleasantries
    - Repetitive confirmations
    - Out-of-scope tangents
    """

    SUMMARY_PROMPT = """You are a context compression assistant for a financial AI system.

Given the following conversation history, produce a CONCENTRATED SEMANTIC SUMMARY.

Your summary MUST retain:
  • All factual claims, numbers, and metrics (revenue, ROE, debt, etc.)
  • Company names, report names, fiscal years, and dates
  • The user's analytical intent (what they were investigating)
  • Any unresolved questions or follow-up items
  • Exact figures cited — do NOT round or paraphrase numbers

Your summary MUST drop:
  • Conversational filler ("Sure!", "Thanks!", "Could you please...")
  • Repeated confirmations or acknowledgments
  • Tangents unrelated to the financial analysis task

Format:
  [Summary]
  <concentrated summary text, ~15-20% of original length>

  [Retained Facts]
  <bullet list of specific facts, figures, and entities>

  [Pending]
  <any unresolved questions or follow-up items>

Begin:"""

    def __init__(
        self,
        summarizer_api: Optional[Any] = None,
        model: str = "qwen2.5-1.5b-instruct",
    ) -> None:
        """
        Parameters
        ----------
        summarizer_api : Any, optional
            Initialized summarizer (e.g., QwenSummarizer from models.qwen_llm).
            If None, uses the DashScope API directly.
        model : str
            Model name for compression.
        """
        self.summarizer_api = summarizer_api
        self.model = model

    def compress(
        self,
        turns: list[dict[str, str]],
    ) -> str:
        """
        Compress a list of conversation turns into a semantic summary.

        Parameters
        ----------
        turns : list of {"role": str, "content": str}
            Chronological conversation turns (oldest first).

        Returns
        -------
        str — concentrated semantic summary
        """
        if not turns:
            return ""

        # Format turns for the prompt
        turns_block = "\n".join(
            f"[{t['role'].upper()}]\n{t['content']}"
            for t in turns
        )

        prompt = f"{self.SUMMARY_PROMPT}\n{turns_block}\n\n[Summary]"

        if self.summarizer_api is not None:
            # Use the provided summarizer API
            response = self.summarizer_api.chat(
                messages=[{"role": "user", "content": prompt}],
                generation_config=Any(temperature=0.3, top_p=0.9, max_tokens=512),  # type: ignore
            )
            return response.content.strip()
        else:
            # Direct DashScope API call
            import os, requests

            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            if not api_key:
                return self._fallback_compress(turns)

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "input": {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ]
                },
                "parameters": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": 512,
                },
            }

            try:
                resp = requests.post(
                    "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/chat",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    choices = result.get("output", {}).get("choices", [{}])
                    return choices[0].get("message", {}).get("content", "") if choices else ""
            except Exception:  # noqa: BLE001
                pass

            return self._fallback_compress(turns)

    @staticmethod
    def _fallback_compress(turns: list[dict[str, str]]) -> str:
        """Fallback compression when no LLM is available."""
        total_text = " ".join(t["content"] for t in turns)
        # Very rough: just truncate to 20% of original tokens
        words = total_text.split()
        keep = words[: len(words) // 5]
        return " ".join(keep) + " [... compressed]"


# ---------------------------------------------------------------------------
# Token Budget Manager
# ---------------------------------------------------------------------------

class TokenBudgetManager:
    """
    Manages the conversation token budget with sliding window + semantic truncation.

    This is the core class that tracks token usage and decides when to truncate.

    Usage
    -----
    >>> manager = TokenBudgetManager(
    ...     budget=8192,
    ...     k_keep=4,
    ... )
    >>> manager.add_turn("user", "What was Apple's revenue?")
    >>> manager.add_turn("assistant", "Apple revenue was $391B...")
    >>> state = {}  # GraphState
    >>> manager.apply_to_state(state)   # writes total_tokens_in_context, short_term_context, memory_summary
    >>> if state.get("memory_summary"):
    ...     print("Summary active:", state["memory_summary"][:50])
    """

    def __init__(
        self,
        policy: Optional[TruncationPolicy] = None,
        truncator: Optional[SemanticTruncator] = None,
    ) -> None:
        self.policy = policy or TruncationPolicy()
        self.truncator = truncator or SemanticTruncator()

        # Internal state
        self._all_turns: list[dict[str, Any]] = []  # full history
        self._summary: Optional[str] = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def add_turn(self, role: str, content: str, token_count: Optional[int] = None) -> None:
        """
        Append a new conversation turn to the history.

        Parameters
        ----------
        role : "user" | "assistant" | "system"
        content : str
        token_count : int, optional
            Pre-computed token count. If omitted, computed automatically.
        """
        if token_count is None:
            token_count = count_tokens(content)

        self._all_turns.append({
            "role": role,
            "content": content,
            "token_count": token_count,
            "timestamp": time.time(),
            "is_summary": False,
        })

    def get_total_tokens(self) -> int:
        """Current total token count of all stored turns."""
        return sum(t["token_count"] for t in self._all_turns if not t.get("is_summary"))

    def should_truncate(self) -> bool:
        """Check if truncation should be triggered."""
        return self.get_total_tokens() >= self.policy.soft_threshold

    def is_budget_exceeded(self) -> bool:
        """Check if hard budget limit is exceeded."""
        return self.get_total_tokens() >= self.policy.budget

    def apply_truncation(self) -> None:
        """
        Execute the truncation strategy.

        1. Identify K most recent full turns (k_keep)
        2. Compress everything older than K into a semantic summary
        3. Replace older turns with the summary
        """
        k = self.policy.k_keep
        n = len(self._all_turns)

        if n <= k:
            return  # Nothing to truncate

        # Recent K turns to keep verbatim
        recent_turns = self._all_turns[-k:]
        older_turns = self._all_turns[:-k]

        if not older_turns:
            return

        # Compress older turns
        older_messages = [
            {"role": t["role"], "content": t["content"]}
            for t in older_turns
        ]
        summary_text = self.truncator.compress(older_messages)

        summary_token_count = count_tokens(summary_text)
        self._summary = summary_text

        # Replace older turns with the summary turn
        self._all_turns = (
            [
                {
                    "role": "system",
                    "content": f"[Prior conversation summary — do not assume any details not mentioned here]\n{summary_text}",
                    "token_count": summary_token_count,
                    "timestamp": time.time(),
                    "is_summary": True,
                }
            ]
            + recent_turns
        )

    def get_context_for_prompt(
        self,
        query_tokens: int = 0,
    ) -> tuple[list[dict[str, str]], int]:
        """
        Return the conversation context suitable for an LLM prompt.

        Returns
        -------
        (messages, total_tokens) — messages ready for LLM chat API,
            and the total token count for record-keeping.
        """
        available = self.policy.budget - query_tokens
        messages: list[dict[str, str]] = []
        total = 0

        for turn in self._all_turns:
            if total + turn["token_count"] <= available:
                messages.append({"role": turn["role"], "content": turn["content"]})
                total += turn["token_count"]

        return messages, total

    def get_full_history(self) -> list[dict[str, Any]]:
        """Return all stored turns."""
        return list(self._all_turns)

    @property
    def summary(self) -> Optional[str]:
        """Current semantic summary, if any."""
        return self._summary

    def get_token_count(self) -> int:
        """Alias for get_total_tokens."""
        return self.get_total_tokens()

    def reset(self) -> None:
        """Clear all history and summary."""
        self._all_turns.clear()
        self._summary = None


# ---------------------------------------------------------------------------
# Truncation Node (LangGraph Integration)
# ---------------------------------------------------------------------------

class TruncationNode:
    """
    LangGraph node that enforces token budget before routing.

    This node runs BEFORE SemanticRouterNode on every request.
    It checks if the token budget is exceeded and triggers truncation
    if needed, writing the compressed summary back to GraphState.

    Because GraphState is a shared global dict in LangGraph (the same
    dict reference is passed to every node), this node's writes are
    immediately visible to all downstream nodes.

    Integration
    -----------
    In the workflow graph:
        input → truncation → semantic_router → ...

    The TruncationNode reads from and writes to GraphState["short_term_context"]
    and GraphState["memory_summary"]. These fields accumulate across all
    node executions within a session.
    """

    def __init__(
        self,
        budget_manager: TokenBudgetManager,
    ) -> None:
        """
        Parameters
        ----------
        budget_manager : TokenBudgetManager
            The token budget manager instance for this session.
        """
        self._manager = budget_manager

    def run(self, state: dict) -> dict:
        """
        Execute the truncation check and apply if needed.

        This is a pure function: reads from state, writes updates to state.

        Reads from GraphState:
          - short_term_context : list of {role, content} conversation turns
          - total_tokens_in_context : int (running token counter)

        Writes to GraphState:
          - short_term_context : updated list (truncated if needed)
          - memory_summary : semantic summary string (if truncation occurred)
          - total_tokens_in_context : updated token count
        """
        # Get current conversation turns from state
        short_term_context: list[dict[str, str]] = state.get("short_term_context", [])
        existing_total = state.get("total_tokens_in_context", 0)

        # Sync turns into the budget manager
        # (Only add turns that aren't already tracked)
        current_tracked = len(self._manager.get_full_history())
        for turn in short_term_context[current_tracked:]:
            self._manager.add_turn(
                role=turn.get("role", "user"),
                content=turn.get("content", ""),
            )

        # Check if budget is exceeded
        if self._manager.is_budget_exceeded():
            # Apply sliding window + semantic truncation
            self._manager.apply_truncation()

            # Get the updated context
            messages, total_tokens = self._manager.get_context_for_prompt()

            # Write back to GraphState
            state["short_term_context"] = messages
            state["memory_summary"] = self._manager.summary
            state["total_tokens_in_context"] = total_tokens
            state["truncation_applied"] = True
        else:
            # Just update the token count
            total_tokens = self._manager.get_total_tokens()
            state["total_tokens_in_context"] = total_tokens
            state["truncation_applied"] = False

        return state


# ---------------------------------------------------------------------------
# Helper: inject token counter into any text
# ---------------------------------------------------------------------------

class TokenCounter:
    """
    Standalone token counter utility.

    Usage:
        >>> counter = TokenCounter()
        >>> n = counter.count("Hello world")
        >>> n = counter.count_messages([
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ... ])
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Count tokens in a single text string."""
        return len(self._encoding.encode(text))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """Count tokens across multiple messages."""
        return count_messages_tokens(messages, self.encoding_name)
