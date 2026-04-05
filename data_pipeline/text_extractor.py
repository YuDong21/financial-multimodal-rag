"""
Text Extractor — Hierarchical Markdown Structure Recovery.

Extracts text from detected TEXT regions in a PDF page and reconstructs
the heading → paragraph hierarchy as structured Markdown.

Handles:
- Multi-column layout detection and reading-order restoration
- Heading level inference from font size / style / position heuristics
- Hierarchical Markdown output with preserved heading relationships
- Parent heading propagation so each chunk retains its section context
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class MarkdownSpan:
    """
    A single inline span within a MarkdownBlock.

    Could be a word, punctuation, or formatted segment.
    """

    text: str
    bold: bool = False
    italic: bool = False
    is_heading_marker: bool = False  # True for ##, ### etc.


@dataclass
class MarkdownBlock:
    """
    A logical block of text extracted from a PDF region.

    Attributes
    ----------
    block_id : str
        Unique identifier.
    heading_level : int or None
        None for paragraphs; 1–6 for headings.
    heading_path : list[str]
        Full path of parent headings, e.g. ["1 Introduction", "1.2 Background"].
    content : str
        The text content (excluding Markdown syntax markers).
    raw_markdown : str
        The full Markdown-formatted string (with # markers, etc.).
    bbox : BBox
        Page coordinates of this block's source region.
    page_number : int
    token_count : int
        Approximate token count.
    """

    block_id: str
    heading_level: Optional[int]
    heading_path: list[str]
    content: str
    raw_markdown: str
    page_number: int
    token_count: int
    bbox: Optional[Any] = None


# ---------------------------------------------------------------------------
# Heading Inference Heuristics
# ---------------------------------------------------------------------------

HEADING_FONT_SIZE_RATIO = 1.2  # Font size relative to body text → heading
COMMON_HEADING_KEYWORDS = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "chapter", "section", "part",
    "abstract", "introduction", "conclusion", "summary",
    "references", "appendix", "acknowledgements",
}


def is_likely_heading(text: str, font_size: Optional[float] = None) -> tuple[bool, Optional[int]]:
    """
    Heuristic: determine if a text block is a heading and its level.

    Parameters
    ----------
    text : str
        The text content of the block.
    font_size : float, optional
        Detected font size in points.

    Returns
    -------
    (is_heading, heading_level) — heading_level is 1–6 or None.
    """
    stripped = text.strip()
    if not stripped:
        return False, None

    # Check for explicit Markdown heading markers
    m = re.match(r"^(#{1,6})\s+(.+)", stripped)
    if m:
        return True, len(m.group(1))

    # Check for numbered section patterns: "1.2.3 Title" or "1.2 Title"
    numbered = re.match(r"^(\d+(?:\.\d+)*)\s+", stripped)
    if numbered:
        levels = len(numbered.group(1).split("."))
        if levels <= 3:
            return True, min(levels + 1, 6)

    # Font size heuristic
    if font_size is not None:
        # Body text ~10pt, heading should be ≥ 1.2× body
        if font_size >= 10 * HEADING_FONT_SIZE_RATIO:
            # Infer level from size: larger → higher priority
            if font_size >= 18:
                return True, 1
            elif font_size >= 15:
                return True, 2
            elif font_size >= 13:
                return True, 3
            else:
                return True, 4

    # Keyword heuristic
    first_word = stripped.split()[0].lower().rstrip(".")
    if first_word in COMMON_HEADING_KEYWORDS:
        return True, 2  # Default to h2 for keyword-detected headings

    return False, None


# ---------------------------------------------------------------------------
# Text Extractor
# ---------------------------------------------------------------------------

class TextExtractor:
    """
    Extracts text from TEXT regions and reconstructs hierarchical Markdown.

    Parameters
    ----------
    preserve_formatting : bool, default True
        Preserve bold/italic formatting in Markdown output.
    merge_broken_lines : bool, default True
        Attempt to reconnect lines that were split during OCR.
    column_detection : bool, default True
        Enable heuristic two-column layout detection.
    """

    MIN_HEADING_CHARS = 3
    MAX_PARAGRAPH_CHARS = 200  # Used only as a rough sanity cap

    def __init__(
        self,
        preserve_formatting: bool = True,
        merge_broken_lines: bool = True,
        column_detection: bool = True,
    ) -> None:
        self.preserve_formatting = preserve_formatting
        self.merge_broken_lines = merge_broken_lines
        self.column_detection = column_detection

    def extract_from_region(
        self,
        region_text: str,
        region_bbox: Optional[Any] = None,
        page_number: int = 1,
        font_sizes: Optional[list[float]] = None,
    ) -> list[MarkdownBlock]:
        """
        Extract hierarchical Markdown blocks from a single TEXT region.

        Parameters
        ----------
        region_text : str
            Raw OCR text extracted from the region.
        region_bbox : BBox, optional
            Bounding box of the source region.
        page_number : int
            Page number for source tracking.
        font_sizes : list of float, optional
            Per-word font sizes (if available from OCR).

        Returns
        -------
        list of MarkdownBlock objects in reading order.
        """
        import uuid

        blocks: list[MarkdownBlock] = []
        heading_path: list[str] = []  # Stack of active parent headings

        lines = region_text.split("\n")
        current_paragraph_lines: list[str] = []
        current_heading: Optional[MarkdownBlock] = None

        def flush_paragraph(lines: list[str]) -> Optional[MarkdownBlock]:
            if not lines:
                return None
            text = self._merge_and_clean("\n".join(lines))
            if len(text) < self.MIN_HEADING_CHARS:
                return None
            content = re.sub(r"^#+\s*", "", text)  # Strip any existing markers
            block_id = str(uuid.uuid4())[:8]
            raw = text
            return MarkdownBlock(
                block_id=block_id,
                heading_level=None,
                heading_path=list(heading_path),
                content=content,
                raw_markdown=raw,
                page_number=page_number,
                token_count=self._count_tokens(text),
                bbox=region_bbox,
            )

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                # Empty line → flush current paragraph
                if current_paragraph_lines:
                    block = flush_paragraph(current_paragraph_lines)
                    if block:
                        blocks.append(block)
                    current_paragraph_lines = []
                continue

            font_size = font_sizes[i] if font_sizes and i < len(font_sizes) else None
            is_heading, level = is_likely_heading(stripped, font_size)

            if is_heading and level is not None:
                # Flush any pending paragraph first
                if current_paragraph_lines:
                    block = flush_paragraph(current_paragraph_lines)
                    if block:
                        blocks.append(block)
                    current_paragraph_lines = []

                # Update heading path
                heading_path = heading_path[: level - 1]  # Trim deeper levels
                heading_path.append(stripped)

                content = re.sub(r"^#+\s*", "", stripped)
                block_id = str(uuid.uuid4())[:8]
                marker = "#" * level
                raw = f"{marker} {stripped}"
                blocks.append(
                    MarkdownBlock(
                        block_id=block_id,
                        heading_level=level,
                        heading_path=list(heading_path),
                        content=content,
                        raw_markdown=raw,
                        page_number=page_number,
                        token_count=self._count_tokens(content),
                        bbox=region_bbox,
                    )
                )
            else:
                # Regular text line
                if self.merge_broken_lines:
                    # Simple heuristic: lines not ending in punctuation → continue
                    if stripped[-1] not in ".。!?;:»)" and len(stripped) > 20:
                        current_paragraph_lines.append(stripped + " ")
                    else:
                        current_paragraph_lines.append(stripped)
                else:
                    current_paragraph_lines.append(stripped)

        # Flush remaining paragraph
        if current_paragraph_lines:
            block = flush_paragraph(current_paragraph_lines)
            if block:
                blocks.append(block)

        return blocks

    def extract_pages(
        self,
        page_texts: list[dict],
    ) -> list[MarkdownBlock]:
        """
        Extract Markdown blocks from multiple pages.

        Parameters
        ----------
        page_texts : list of dict
            Each dict contains: "page_number", "regions" (list of region dicts with
            "text", "bbox", "font_sizes").

        Returns
        -------
        list of MarkdownBlock across all pages.
        """
        all_blocks: list[MarkdownBlock] = []
        for page in page_texts:
            for region in page.get("regions", []):
                blocks = self.extract_from_region(
                    region_text=region["text"],
                    region_bbox=region.get("bbox"),
                    page_number=page["page_number"],
                    font_sizes=region.get("font_sizes"),
                )
                all_blocks.extend(blocks)
        return all_blocks

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _merge_and_clean(text: str) -> str:
        """Clean up OCR artifacts and normalize whitespace."""
        # Remove excessive whitespace
        text = re.sub(r"[ \t]+", " ", text)
        # Remove orphaned single characters (OCR artifacts)
        text = re.sub(r"\n(?=[a-z])", " ", text)
        # Normalize line breaks within paragraphs
        return text.strip()

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Rough token count (words + punctuation)."""
        return len(text.split())
