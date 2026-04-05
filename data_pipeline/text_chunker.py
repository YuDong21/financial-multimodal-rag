"""
Text Chunker — Hierarchical Markdown-aware Text Chunking.

Implements the DeepDoc text chunking strategy:

1. Structural chunking: split by Markdown heading hierarchy first
2. Within each heading section: split by natural paragraph boundaries
3. Length control: enforce max_tokens_per_chunk threshold
4. Parent heading preservation: attach the full heading path to each chunk,
   so retrieval always has section context

Output: list of :class:`TextChunk` with ``text`` (full Markdown with heading
context), ``heading_path``, ``chunk_index``, and metadata.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .text_extractor import MarkdownBlock

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    """
    A text chunk with hierarchical heading context preserved.

    Attributes
    ----------
    chunk_id : str
    text : str
        Full chunk text including heading markers and parent heading path.
    heading_path : list[str]
        Full path of parent headings, e.g. ["1 Introduction", "1.2 Related Work"].
    heading_level : int or None
        Heading level of this chunk's own heading (None for body paragraphs).
    chunk_index : int
        Position of this chunk within its parent heading section.
    source_block_ids : list[str]
        IDs of source MarkdownBlocks that contributed to this chunk.
    page_number : int
    token_count : int
    metadata : dict
        Additional context (source_doc, section_type, etc.).
    """

    chunk_id: str
    text: str
    heading_path: list[str]
    heading_level: Optional[int]
    chunk_index: int
    source_block_ids: list[str]
    page_number: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Text Chunker
# ---------------------------------------------------------------------------

class TextChunker:
    """
    Hierarchical Markdown-aware text chunker.

    Parameters
    ----------
    max_tokens : int, default 512
        Maximum tokens per chunk. Chunks will be split to stay near this limit.
    overlap_tokens : int, default 64
        Token overlap between adjacent chunks in the same section.
        Helps preserve context across chunk boundaries.
    min_chunk_tokens : int, default 32
        Minimum chunk size. Smaller fragments are merged with the next chunk.
    split_by_heading : bool, default True
        If True, never split across heading boundaries.
        If False, split purely by length (aggressive mode).
    preserve_formatting : bool, default True
        Preserve bold/italic Markdown markers in output text.
    """

    MIN_CHUNK_TOKENS = 32
    OVERLAP_TOKENS = 64
    MAX_TOKENS = 512

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        min_chunk_tokens: int = 32,
        split_by_heading: bool = True,
        preserve_formatting: bool = True,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.split_by_heading = split_by_heading
        self.preserve_formatting = preserve_formatting

    def chunk_blocks(
        self,
        blocks: list[MarkdownBlock],
        source_doc: Optional[str] = None,
    ) -> list[TextChunk]:
        """
        Chunk a list of MarkdownBlocks into retrieval-ready TextChunks.

        Parameters
        ----------
        blocks : list of MarkdownBlock
            Extracted blocks from TextExtractor (must be in reading order).
        source_doc : str, optional
            Source document name (e.g. "apple_2024_annual_report.pdf").

        Returns
        -------
        list of TextChunk, sorted in reading order.
        """
        chunks: list[TextChunk] = []
        current_section_blocks: list[tuple[MarkdownBlock, str]] = (
            []
        )  # (block, full_markdown_text)
        current_heading_path: list[str] = []
        current_heading_level: Optional[int] = None
        chunk_index = 0

        def flush_section() -> list[TextChunk]:
            """Flush current section blocks as chunks, respecting max_tokens."""
            nonlocal chunk_index
            if not current_section_blocks:
                return []

            section_chunks: list[TextChunk] = []
            current_text_parts: list[str] = []
            current_block_ids: list[str] = []

            def emit_chunk(text_parts: list[str], block_ids: list[str]) -> TextChunk:
                nonlocal chunk_index
                heading_prefix = (
                    "\n".join(f"## {h}" for h in current_heading_path) + "\n"
                    if current_heading_path
                    else ""
                )
                body_text = "\n\n".join(text_parts)
                full_text = (heading_prefix + body_text).strip()
                token_count = self._count_tokens(full_text)

                block_ids_out = [bid for _, bid in block_ids]
                return TextChunk(
                    chunk_id=str(uuid.uuid4())[:8],
                    text=full_text,
                    heading_path=list(current_heading_path),
                    heading_level=current_heading_level,
                    chunk_index=chunk_index,
                    source_block_ids=block_ids_out,
                    page_number=block_ids[0][0].page_number if block_ids else 1,
                    token_count=token_count,
                    metadata={
                        "source_doc": source_doc or "",
                        "section_type": "heading"
                        if current_heading_level is not None
                        else "body",
                        "num_blocks": len(block_ids),
                    },
                )

            for block, block_text in current_section_blocks:
                block_tokens = self._count_tokens(block_text)

                # If adding this block exceeds max_tokens, flush and start new chunk
                current_tokens = self._count_tokens("\n\n".join(current_text_parts + [block_text]))
                if current_tokens > self.max_tokens and current_text_parts:
                    # Emit current chunk
                    emitted = emit_chunk(current_text_parts, current_block_ids)
                    section_chunks.append(emitted)
                    chunk_index += 1

                    # Start new chunk with overlap
                    if self.overlap_tokens > 0 and current_text_parts:
                        # Include last part as overlap
                        overlap_text = current_text_parts[-1]
                        current_text_parts = [overlap_text]
                        current_block_ids = [current_block_ids[-1]]
                    else:
                        current_text_parts = []
                        current_block_ids = []

                current_text_parts.append(block_text)
                current_block_ids.append((block, block.block_id))

            # Emit remaining
            if current_text_parts:
                emitted = emit_chunk(current_text_parts, current_block_ids)
                section_chunks.append(emitted)
                chunk_index += 1

            return section_chunks

        for block in blocks:
            if block.heading_level is not None:
                # Heading block — flush current section
                if current_section_blocks:
                    chunks.extend(flush_section())
                    current_section_blocks = []

                # Update heading path
                current_heading_path = block.heading_path
                current_heading_level = block.heading_level

                # A heading itself can be a chunk if it's long enough
                if block.token_count >= self.min_chunk_tokens:
                    chunks.append(
                        TextChunk(
                            chunk_id=str(uuid.uuid4())[:8],
                            text=block.raw_markdown,
                            heading_path=list(current_heading_path),
                            heading_level=block.heading_level,
                            chunk_index=chunk_index,
                            source_block_ids=[block.block_id],
                            page_number=block.page_number,
                            token_count=block.token_count,
                            metadata={
                                "source_doc": source_doc or "",
                                "section_type": "heading",
                                "num_blocks": 1,
                            },
                        )
                    )
                    chunk_index += 1
                current_section_blocks = []
            else:
                # Body paragraph block
                current_section_blocks.append((block, block.raw_markdown))

        # Flush final section
        if current_section_blocks:
            chunks.extend(flush_section())

        return chunks

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Rough word-based token count."""
        # Simple tokenizer: split on whitespace and punctuation
        words = re.findall(r"[\w\.\-\+\%]+", text)
        return len(words)
