"""
DeepDoc MCP Tool — Layout Analysis and Table Recovery.

Implements the MCP (Model Context Protocol) tool interface for calling
RAGFlow DeepDoc's layout analysis API. Handles:

- PDF document upload and layout analysis
- Cross-page table detection and Markdown conversion
- OCR correction for garbled financial table cells

This module provides a mock implementation for development / testing.
Replace ``_mock_call`` with a real httpx call to the DeepDoc service
when the RAGFlow backend is deployed.

MCP Tool Schema
---------------
name: deepdoc_layout_analysis
description: >
  Analyzes the layout of a financial PDF document, detecting cross-page tables,
  extracting structured table data, and converting the document to Markdown.
  Use this tool when the user's question requires data from financial tables
  that may span multiple pages or have OCR issues.
parameters:
  type: object
  properties:
    file_path:
      type: string
      description: Path to the PDF file on the local filesystem
    query:
      type: string
      description: Natural language description of what table/chart data to extract
  required: [file_path, query]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DeepDocResult:
    """
    Result returned by the DeepDoc layout analysis tool.

    Attributes
    ----------
    markdown : str
        Full Markdown conversion of the document.
    tables : list[dict]
        Each dict has keys: header, rows, page_range, is_cross_page.
    figures : list[dict]
        Each dict has keys: caption, page_number, bbox.
    ocr_corrected : bool
        True if OCR correction was applied to table cells.
    """

    markdown: str
    tables: list[dict[str, Any]]
    figures: list[dict[str, Any]]
    ocr_corrected: bool


class DeepDocMCPTool:
    """
    MCP tool wrapper for RAGFlow DeepDoc layout analysis.

    Usage as an MCP tool:

    >>> tool = DeepDocMCPTool()
    >>> result = tool.execute(file_path="/data/apple_2024.pdf", query="revenue table")
    >>> for table in result.tables:
    ...     print(table["header"])
    """

    TOOL_NAME = "deepdoc_layout_analysis"
    TOOL_DESCRIPTION = (
        "Analyzes the layout of a financial PDF document, detecting cross-page tables, "
        "extracting structured table data, and converting the document to Markdown. "
        "Use this tool when the user's question requires data from financial tables "
        "that may span multiple pages or have OCR issues."
    )

    PARAMETER_SCHEMA = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the PDF file on the local filesystem.",
            },
            "query": {
                "type": "string",
                "description": (
                    "Natural language description of what table/chart data to extract. "
                    "E.g. 'revenue by quarter', 'balance sheet', 'ROE trend'"
                ),
            },
        },
        "required": ["file_path", "query"],
    }

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        """
        Parameters
        ----------
        api_base_url : str, optional
            DeepDoc HTTP API base URL. Defaults to http://localhost:9380/api/v1/deepdoc.
        api_key : str, optional
        timeout : int, default 120
        """
        self.api_base_url = (
            api_base_url or "http://localhost:9380/api/v1/deepdoc"
        ).rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def get_schema(self) -> dict[str, Any]:
        """
        Return the MCP tool schema for tool registration.

        Returns
        -------
        dict with ``name``, ``description``, and ``parameters`` keys.
        """
        return {
            "name": self.TOOL_NAME,
            "description": self.TOOL_DESCRIPTION,
            "parameters": self.PARAMETER_SCHEMA,
        }

    def execute(
        self,
        file_path: str,
        query: str,
    ) -> DeepDocResult:
        """
        Execute the DeepDoc layout analysis tool.

        Parameters
        ----------
        file_path : str
            Path to the PDF file.
        query : str
            Description of what to extract.

        Returns
        -------
        DeepDocResult with Markdown, tables, figures, and OCR status.

        Raises
        ------
        NotImplementedError
            Stub — replace with real DeepDoc API call when deployed.
        """
        return self._mock_call(file_path, query)

    # -------------------------------------------------------------------------
    # Mock Implementation (for development / CI testing)
    # -------------------------------------------------------------------------

    def _mock_call(self, file_path: str, query: str) -> DeepDocResult:
        """
        Mock DeepDoc response for development without a live RAGFlow backend.

        Simulates what a real DeepDoc analysis would return for a revenue query.
        """
        import re

        mock_markdown = f"""# Annual Report

## Revenue Overview

| Quarter | Revenue (B USD) | YoY Growth |
|---------|-----------------|------------|
| Q1 2024 | 119.6 | +8.1% |
| Q2 2024 | 85.8 | +4.3% |
| Q3 2024 | 94.9 | +6.1% |
| Q4 2024 | 91.2 | +5.7% |

**Total Annual Revenue: $391.5B** (FY2024)

## Cross-Page Table: Balance Sheet (Page 12-13)

| Item | FY2024 | FY2023 |
|------|--------|--------|
| Total Assets | 352.6B | 352.8B |
| Total Liabilities | 278.9B | 290.8B |
| Shareholders' Equity | 73.7B | 62.0B |

> This table spans pages 12 and 13 of the original PDF.
"""

        # Simulate table extraction based on query keywords
        query_lower = query.lower()
        tables: list[dict[str, Any]] = []

        if any(kw in query_lower for kw in ["revenue", "quarter", "q1", "q2", "q3", "q4"]):
            tables.append({
                "header": ["Quarter", "Revenue (B USD)", "YoY Growth"],
                "rows": [
                    ["Q1 2024", "119.6", "+8.1%"],
                    ["Q2 2024", "85.8", "+4.3%"],
                    ["Q3 2024", "94.9", "+6.1%"],
                    ["Q4 2024", "91.2", "+5.7%"],
                ],
                "page_range": [1, 1],
                "is_cross_page": False,
                "caption": "Quarterly Revenue FY2024",
            })

        if any(kw in query_lower for kw in ["balance", "asset", "liability", "equity"]):
            tables.append({
                "header": ["Item", "FY2024", "FY2023"],
                "rows": [
                    ["Total Assets", "352.6B", "352.8B"],
                    ["Total Liabilities", "278.9B", "290.8B"],
                    ["Shareholders' Equity", "73.7B", "62.0B"],
                ],
                "page_range": [12, 13],
                "is_cross_page": True,
                "caption": "Balance Sheet (cross-page table on pages 12-13)",
            })

        return DeepDocResult(
            markdown=mock_markdown,
            tables=tables,
            figures=[],
            ocr_corrected=False,
        )
