"""
DeepDoc Layout Analysis Interface.

Provides a Python client wrapper around the RAGFlow DeepDoc service
(https://github.com/infiniflow/ragflow) for:
  - PDF / Office document layout analysis
  - Cross-page table detection and structure recovery
  - OCR correction for garbled financial table cells
  - Markdown conversion with table / image / text segmentation

This module is a STUB until the RAGFlow DeepDoc HTTP API is deployed.
All methods raise NotImplementedError with instructions for production wiring.
"""

from __future__ import annotations

import io
from typing import Any, BinaryIO, Optional

import pydantic


class Document(pydantic.BaseModel):
    """Represents a parsed document with its layout segments."""

    doc_id: str
    file_name: str
    total_pages: int
    segments: list[Segment]
    tables: list[TableSegment]
    figures: list[FigureSegment]


class Segment(pydantic.BaseModel):
    """A text or mixed-content segment produced by DeepDoc layout analysis."""

    segment_id: str
    page_number: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in points
    content: str  # Markdown-formatted text
    token_count: int


class TableSegment(pydantic.BaseModel):
    """A detected table with its grid structure and cell content."""

    table_id: str
    page_numbers: list[int]
    header_row: list[str]
    body_rows: list[list[str]]
    bbox: tuple[float, float, float, float]
    is_cross_page: bool


class FigureSegment(pydantic.BaseModel):
    """A detected figure / chart with its caption and page location."""

    figure_id: str
    page_number: int
    caption: str
    bbox: tuple[float, float, float, float]


class DeepDocClient:
    """
    Client for the RAGFlow DeepDoc layout analysis service.

    Parameters
    ----------
    base_url : str
        Base URL of the DeepDoc HTTP API.
        e.g. "http://localhost:9380/api/v1/deepdoc"
    api_key : str, optional
        API key for authentication. Reads from DEEPDOC_API_KEY env var if omitted.
    timeout : int, default 60
        Request timeout in seconds for large PDF uploads.

    Examples
    --------
    >>> client = DeepDocClient(
    ...     base_url="http://ragflow-backend:9380/api/v1/deepdoc",
    ...     api_key="sk-xxxx"
    ... )
    >>> doc = client.analyze_pdf("/data/annual_report_2024.pdf")
    >>> print(f"Pages: {doc.total_pages}, Tables: {len(doc.tables)}")
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.base_url = (base_url or "http://localhost:9380/api/v1/deepdoc").rstrip("/")
        self.api_key = api_key or self._load_api_key()
        self.timeout = timeout

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def analyze_pdf(self, file_path: str) -> Document:
        """
        Upload a PDF file and run full layout analysis via DeepDoc.

        Parameters
        ----------
        file_path : str
            Path to the PDF file on the local filesystem.

        Returns
        -------
        Document
            Parsed document with text segments, tables, and figures.

        Raises
        ------
        NotImplementedError
            Stub — replace with actual HTTP call to DeepDoc when deployed.
        """
        raise NotImplementedError(
            "DeepDoc is not yet deployed. "
            "To enable layout analysis:\n"
            "  1. Deploy RAGFlow (https://github.com/infiniflow/ragflow)\n"
            "  2. Set base_url to your DeepDoc service endpoint\n"
            "  3. Implement the HTTP POST /analyze endpoint below"
        )

    def analyze_pdf_stream(
        self, file_bytes: BinaryIO, file_name: str
    ) -> Document:
        """
        Upload a PDF from a byte stream and run layout analysis.

        Parameters
        ----------
        file_bytes : BinaryIO
            File-like byte stream of the PDF content.
        file_name : str
            Original file name (used for extension detection).

        Returns
        -------
        Document

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "DeepDoc stream analysis not yet implemented. "
            "Use file-path based analysis until the API is deployed."
        )

    def extract_cross_page_tables(self, doc: Document) -> list[TableSegment]:
        """
        Given a Document, return only the TableSegments that span multiple pages.

        Parameters
        ----------
        doc : Document

        Returns
        -------
        list[TableSegment]
            Tables with ``is_cross_page == True``.
        """
        return [t for t in doc.tables if t.is_cross_page]

    def to_markdown(self, doc: Document) -> str:
        """
        Serialize a Document back to a Markdown string.

        Preserves table structure using GitHub-Flavored Markdown table syntax.

        Parameters
        ----------
        doc : Document

        Returns
        -------
        str
            Markdown representation.
        """
        lines: list[str] = [f"# {doc.file_name}\n"]

        for seg in doc.segments:
            if seg.content.strip():
                lines.append(seg.content)
                lines.append("")  # blank line between sections

        if doc.tables:
            lines.append("\n## Tables\n")
            for tbl in doc.tables:
                lines.append(self._table_to_md(tbl))
                lines.append("")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_api_key() -> str:
        import os

        key = os.environ.get("DEEPDOC_API_KEY", "")
        if not key:
            # Silently allow empty key for local/dev deployments
            pass
        return key

    @staticmethod
    def _table_to_md(table: TableSegment) -> str:
        """Render a TableSegment as a Markdown table."""
        lines = []
        header = "| " + " | ".join(table.header_row) + " |"
        separator = "| " + " | ".join(["---"] * len(table.header_row)) + " |"
        lines.append(f"Table (pages {table.page_numbers}): {table.table_id}")
        lines.append(header)
        lines.append(separator)
        for row in table.body_rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)
