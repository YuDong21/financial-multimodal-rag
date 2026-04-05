"""
Context Window Management — Sliding Window + Semantic Truncation.

Implements two complementary strategies to keep conversation history within
the LLM context window:

1. **Sliding Window** — A fixed-capacity FIFO queue that discards the oldest
   turns once the window is full. Token budget is enforced per-window.

2. **Semantic Truncation** — When the window overflows, instead of blind
   FIFO eviction, we use a Qwen2.5 summarizer to semantically compress the
   oldest N turns into a dense summary, preserving factual content while
   reducing token count.

Design
------
``MemoryManager`` holds the full conversation and exposes a
``get_context_for_prompt()`` method that returns the optimal context slice
(either raw recent turns or a compressed summary + recent turns) based on
the current token count.

Usage
-----
    >>> from models.qwen_llm import QwenSummarizer
    >>> summarizer = QwenSummarizer()
    >>> manager = MemoryManager(
    ...     max_window_tokens=8192,
    ...     summarizer=summarizer,
    ...     compression_threshold=0.85,  # compress when 85% full
    ... )
    >>> manager.add_turn("user", "What was Apple's revenue in FY2024?")
    >>> manager.add_turn("assistant", "Apple revenue was $391.0B in FY2024...")
    >>> ctx = manager.get_context_for_prompt(query_tokens=500)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import tiktoken

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ConversationTurn:
    """
    A single conversational exchange unit.

    Attributes
    ----------
    role : "user" | "assistant" | "system"
    content : str
        The raw text of this turn.
    token_count : int
        Pre-computed token count using cl100k_base encoding.
    timestamp : float
        Unix timestamp when this turn was added.
    is_summary : bool
        True if this turn is a semantically compressed summary of prior turns.
    """

    role: str
    content: str
    token_count: int
    timestamp: float
    is_summary: bool = False


# ---------------------------------------------------------------------------
# Token Counter Utility
# ---------------------------------------------------------------------------

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in a text string using tiktoken.

    Uses cl100k_base (GPT-4 / Qwen training encoding) by default.
    """
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# Context Window
# ---------------------------------------------------------------------------


class ContextWindow:
    """
    A fixed-capacity sliding window over conversation turns.

    Parameters
    ----------
    max_tokens : int, default 8192
        Maximum total token budget for all turns in the window.
    """

    def __init__(self, max_tokens: int = 8192) -> None:
        self.max_tokens = max_tokens
        self._turns: list[ConversationTurn] = []

    @property
    def turns(self) -> list[ConversationTurn]:
        """Current turns in the window (oldest first)."""
        return list(self._turns)

    @property
    def total_tokens(self) -> int:
        """Total token count of all turns in the window."""
        return sum(t.token_count for t in self._turns)

    def add(self, turn: ConversationTurn) -> list[ConversationTurn]:
        """
        Add a turn, evicting oldest turns until the window fits.

        Parameters
        ----------
        turn : ConversationTurn

        Returns
        -------
        list of ConversationTurn
            The turns that were evicted (for potential summarization).
        """
        evicted: list[ConversationTurn] = []
        self._turns.append(turn)

        while self.total_tokens > self.max_tokens and len(self._turns) > 1:
            oldest = self._turns.pop(0)
            evicted.append(oldest)

        return evicted

    def clear(self) -> None:
        """Remove all turns from the window."""
        self._turns.clear()

    def to_messages(self) -> list[dict[str, str]]:
        """Serialize turns to a message list compatible with LLM chat APIs."""
        return [{"role": t.role, "content": t.content} for t in self._turns]


# ---------------------------------------------------------------------------
# Memory Manager — Sliding Window + Semantic Truncation
# ---------------------------------------------------------------------------


class MemoryManager:
    """
    Manages conversation history with adaptive compression.

    Parameters
    ----------
    max_window_tokens : int, default 8192
        Hard token limit for the context passed to the LLM.
    summarizer : QwenSummarizer, optional
        Semantic summarizer (Qwen2.5). If None, falls back to plain FIFO eviction.
    compression_threshold : float, default 0.85
        Trigger compression when window reaches this fraction of max_window_tokens.
    full_history_tokens : int, default 32768
        Hard limit on total stored history (before forced truncation).
    """

    def __init__(
        self,
        max_window_tokens: int = 8192,
        summarizer: Optional[Any] = None,
        compression_threshold: float = 0.85,
        full_history_tokens: int = 32768,
    ) -> None:
        self.max_window_tokens = max_window_tokens
        self.summarizer = summarizer
        self.compression_threshold = compression_threshold
        self.full_history_tokens = full_history_tokens

        self._window = ContextWindow(max_tokens=max_window_tokens)
        self._all_turns: list[ConversationTurn] = []  # full history (for compression)
        self._summary: Optional[str] = None  # compressed summary of older turns

    def add_turn(
        self,
        role: str,
        content: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Append a new turn to the conversation history.

        Parameters
        ----------
        role : "user" | "assistant" | "system"
        content : str
        timestamp : float, optional
            Unix timestamp. Defaults to current time.
        """
        import time

        if timestamp is None:
            timestamp = time.time()

        token_count = count_tokens(content)
        turn = ConversationTurn(
            role=role,
            content=content,
            token_count=token_count,
            timestamp=timestamp,
        )

        self._all_turns.append(turn)
        self._window.add(turn)

        # Check if compression is needed
        if (
            self.summarizer is not None
            and self._window.total_tokens >= self.max_window_tokens * self.compression_threshold
        ):
            self._compress_oldest_turns()

    def _compress_oldest_turns(self) -> None:
        """
        Compress the oldest N turns into a semantic summary using Qwen2.5.

        The summary replaces the evicted turns in the context window while
        the full history is preserved separately for audit purposes.
        """
        if self.summarizer is None:
            return

        # How many turns to compress — aim to recover ~40% of the window
        target_recovery = int(self.max_window_tokens * 0.4)
        accumulated = 0
        to_compress: list[ConversationTurn] = []

        for turn in self._all_turns:
            if turn.is_summary:
                continue
            to_compress.append(turn)
            accumulated += turn.token_count
            if accumulated >= target_recovery:
                break

        if len(to_compress) < 2:
            return  # Not enough turns to compress

        # Summarize
        turns_to_summarize = [
            {"role": t.role, "content": t.content} for t in to_compress
        ]
        result = self.summarizer.summarize_turns(turns_to_summarize)
        summary_text = result.content

        # Insert the summary as a special "summary" turn at the boundary
        summary_turn = ConversationTurn(
            role="system",
            content=f"[Prior conversation summary]\n{summary_text}",
            token_count=count_tokens(summary_text),
            timestamp=to_compress[-1].timestamp,
            is_summary=True,
        )

        # Replace compressed turns in window
        self._window.clear()
        self._window.add(summary_turn)
        self._summary = summary_text

    def get_context_for_prompt(
        self,
        query_tokens: int = 0,
    ) -> list[dict[str, str]]:
        """
        Return the conversation context suitable for an LLM prompt.

        The context includes:
        1. The semantic summary (if any) of older turns
        2. Recent raw turns that fit within the remaining token budget

        Parameters
        ----------
        query_tokens : int, default 0
            Estimated token count of the current query (reserved from window).

        Returns
        -------
        list of {"role": str, "content": str}
        """
        available = self.max_window_tokens - query_tokens

        # If we have a summary, start with it
        messages: list[dict[str, str]] = []
        if self._summary is not None:
            messages.append(
                {"role": "system", "content": f"[Prior conversation summary]\n{self._summary}"}
            )

        # Fill remaining budget with recent turns
        remaining = available - sum(m["content"].count(" ") for m in messages)  # rough
        for turn in reversed(self._window.turns):
            if turn.is_summary:
                continue
            if self._window.total_tokens <= remaining:
                messages.insert(
                    0 if not messages else len(messages),
                    {"role": turn.role, "content": turn.content},
                )
        # Actually use exact token counting
        messages = []
        current_tokens = sum(count_tokens(m["content"]) for m in messages)
        for turn in reversed(self._window.turns):
            if turn.is_summary:
                continue
            if current_tokens + turn.token_count <= available:
                messages.insert(0, {"role": turn.role, "content": turn.content})
                current_tokens += turn.token_count

        return messages

    def get_full_history(self) -> list[ConversationTurn]:
        """Return all stored conversation turns (including compressed ones)."""
        return list(self._all_turns)

    @property
    def summary(self) -> Optional[str]:
        """Current semantic summary of older conversation history."""
        return self._summary
