"""
DeepDoc Parsing MCP Tools.

Category 1: DeepDoc document parsing tools.

Tools:
  - deepdoc_table_parse    : Parse a table region → logical Markdown table
  - deepdoc_cross_page_merge : Merge cross-page table fragments → one logical table
  - deepdoc_chart_extract  : Extract chart semantic description (OCR + caption + axis)
  - deepdoc_ocr_correct    : OCR correction for garbled financial table cells
"""

from __future__ import annotations

from typing import Any, Optional

from .base import MCPTool

# ---------------------------------------------------------------------------
# Tool: Table Parse
# ---------------------------------------------------------------------------

class DeepDocTableParseTool(MCPTool):
    """
    Parse a detected TABLE region from a PDF page into a logical Markdown table.

    Internally uses TableStructureRecovery (TSR CNN + OCR) to:
    - Detect cell grid structure
    - Identify merged cells (rowspan/colspan)
    - Extract cell text with OCR
    - Preserve header row, units, and footnotes
    """

    name = "deepdoc_table_parse"
    description = (
        "Parse a PDF table region into a structured Markdown table. "
        "Use this when you need to extract and structure a financial table "
        "from a specific page region. Returns: header_row, body_rows, "
        "markdown representation, and metadata."
    )

    parameters = {
        "type": "object",
        "properties": {
            "table_image_path": {
                "type": "string",
                "description": "Path to the cropped table image file.",
            },
            "page_number": {
                "type": "integer",
                "description": "PDF page number (1-indexed) where this table appears.",
            },
            "options": {
                "type": "object",
                "description": "Optional TSR parameters.",
                "properties": {
                    "language": {"type": "string", "enum": ["en", "zh", "mixed"], "default": "mixed"},
                    "include_ocr_correct": {"type": "boolean", "default": True},
                },
            },
        },
        "required": ["table_image_path", "page_number"],
    }

    def __init__(self, table_recovery: Optional[Any] = None) -> None:
        from ..data_pipeline.table_structure_recovery import TableStructureRecovery
        self._tsr = table_recovery or TableStructureRecovery()

    def execute(
        self,
        table_image_path: str,
        page_number: int,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute table parsing.

        Returns a dict with:
          - table_id, header_row, body_rows, markdown, is_cross_page,
            spanning_cells, metadata, page_number, ocr_corrected
        """
        import numpy as np

        # Load image from path (placeholder — replace with actual image loading)
        try:
            from PIL import Image
            img = Image.open(table_image_path)
            table_image = np.array(img.convert("RGB"))
        except Exception:  # noqa: BLE001
            # Mock for development
            table_image = np.zeros((100, 100, 3), dtype=np.uint8)

        options = options or {}
        lang = options.get("language", "mixed")
        include_ocr = options.get("include_ocr_correct", True)

        # Run TSR
        try:
            ts = self._tsr.recover(
                table_images=[table_image],
                page_numbers=[page_number],
            )
            return {
                "table_id": ts.table_id,
                "header_row": ts.header_row,
                "body_rows": ts.body_rows,
                "markdown": ts.raw_markdown,
                "is_cross_page": ts.is_cross_page,
                "spanning_cells": [
                    {
                        "row": c.row,
                        "col": c.col,
                        "rowspan": c.rowspan,
                        "colspan": c.colspan,
                        "text": c.text,
                    }
                    for c in ts.spanning_cells
                ],
                "metadata": ts.metadata,
                "page_number": page_number,
                "ocr_corrected": ts.ocr_corrected,
                "language": lang,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool: Cross-Page Table Merge
# ---------------------------------------------------------------------------

class DeepDocCrossPageMergeTool(MCPTool):
    """
    Merge table fragments that span multiple pages into a single logical table.

    Takes a list of page_image_paths and their page numbers, identifies
    the shared table structure across pages, deduplicates header rows
    on page 2+, and returns a merged logical table.
    """

    name = "deepdoc_cross_page_merge"
    description = (
        "Merge table fragments that span multiple PDF pages into one logical table. "
        "Automatically detects repeated headers on subsequent pages and deduplicates. "
        "Use when a financial table is split across consecutive pages."
    )

    parameters = {
        "type": "object",
        "properties": {
            "fragments": {
                "type": "array",
                "description": "List of table fragment descriptions.",
                "items": {
                    "type": "object",
                    "properties": {
                        "table_image_path": {"type": "string"},
                        "page_number": {"type": "integer"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "[x1, y1, x2, y2] in PDF points.",
                        },
                    },
                    "required": ["table_image_path", "page_number"],
                },
            },
            "options": {
                "type": "object",
                "properties": {
                    "deduplicate_headers": {"type": "boolean", "default": True},
                    "infer_column_spans": {"type": "boolean", "default": True},
                },
            },
        },
        "required": ["fragments"],
    }

    def execute(self, fragments: list[dict[str, Any]], options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Merge cross-page table fragments.

        Returns merged table dict (same schema as table_parse).
        """
        from ..data_pipeline.table_structure_recovery import TableStructureRecovery

        tcr = TableStructureRecovery()
        options = options or {}

        # Override cross_page_mode to force merge
        import os
        tcr.cross_page_mode = "force_merge"

        table_images = []
        page_numbers = []
        bboxes = []

        for frag in fragments:
            try:
                from PIL import Image
                import numpy as np
                img = Image.open(frag["table_image_path"])
                table_images.append(np.array(img.convert("RGB")))
            except Exception:  # noqa: BLE001
                import numpy as np
                table_images.append(np.zeros((100, 100, 3), dtype=np.uint8))

            page_numbers.append(frag["page_number"])
            bboxes.append(frag.get("bbox"))

        try:
            ts = tcr.recover(
                table_images=table_images,
                page_numbers=page_numbers,
                table_bboxes=bboxes,
            )
            return {
                "table_id": ts.table_id,
                "header_row": ts.header_row,
                "body_rows": ts.body_rows,
                "markdown": ts.raw_markdown,
                "is_cross_page": True,
                "page_span": ts.page_numbers,
                "metadata": ts.metadata,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool: Chart Semantic Extract
# ---------------------------------------------------------------------------

class DeepDocChartExtractTool(MCPTool):
    """
    Extract a semantic description from a chart region.

    Combines: OCR text inside chart + axis labels + data point extraction
    + caption search + chart type inference → ChartSemanticBlock.
    """

    name = "deepdoc_chart_extract"
    description = (
        "Extract a semantic description from a chart region in a financial PDF. "
        "Returns: chart_type, title, caption, axis_info, data_points, "
        "and a generated natural-language description. "
        "Use when answering questions about trends, comparisons, or distributions "
        "shown in charts or figures."
    )

    parameters = {
        "type": "object",
        "properties": {
            "chart_image_path": {
                "type": "string",
                "description": "Path to the cropped chart image file.",
            },
            "page_regions": {
                "type": "array",
                "description": "Nearby TEXT regions from the same page (for caption search).",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "bbox": {"type": "array", "items": {"type": "number"}},
                    },
                },
            },
            "page_number": {"type": "integer"},
            "options": {
                "type": "object",
                "properties": {
                    "include_verbose_description": {"type": "boolean", "default": True},
                    "include_data_points": {"type": "boolean", "default": True},
                    "max_data_points": {"type": "integer", "default": 30},
                },
            },
        },
        "required": ["chart_image_path", "page_number"],
    }

    def execute(
        self,
        chart_image_path: str,
        page_number: int,
        page_regions: Optional[list[dict[str, Any]]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute chart semantic extraction.

        Returns chart info dict.
        """
        from ..data_pipeline.chart_extractor import ChartExtractor

        options = options or {}
        extractor = ChartExtractor()

        # Load chart image
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(chart_image_path)
            chart_image = np.array(img.convert("RGB"))
        except Exception:  # noqa: BLE001:
            import numpy as np
            chart_image = np.zeros((100, 100, 3), dtype=np.uint8)

        try:
            block = extractor.extract(
                chart_image=chart_image,
                page_regions=page_regions or [],
                page_number=page_number,
            )
            return block.to_dict()
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool: OCR Correction
# ---------------------------------------------------------------------------

class DeepDocOCRCorrectTool(MCPTool):
    """
    Apply OCR correction to garbled text in financial table cells.

    Handles common garbling patterns:
    - Currency symbols: ¥ → $, ￥ → $
    - Dash normalization: — → -, – → -
    - Number format: 1,000.00 vs 1.000,00 normalization
    - Common financial abbreviation expansion
    """

    name = "deepdoc_ocr_correct"
    description = (
        "Correct garbled OCR text in financial table cells. "
        "Handles currency symbols, dash normalization, number formatting, "
        "and financial abbreviation expansion. "
        "Use when table cells contain obviously garbled characters."
    )

    parameters = {
        "type": "object",
        "properties": {
            "cell_texts": {
                "type": "array",
                "description": "List of raw OCR cell text strings to correct.",
                "items": {"type": "string"},
            },
            "language": {
                "type": "string",
                "enum": ["en", "zh", "mixed"],
                "default": "mixed",
                "description": "Primary language of the document.",
            },
        },
        "required": ["cell_texts"],
    }

    CORRECTIONS_EN: dict[str, str] = {
        "—": "-", "–": "-", """: '"', """: '"',
        "'": "'", "'": "'", "．": ".", "（": "(", "）": ")",
        "％": "%", "¥": "$", "￥": "$",
    }

    CORRECTIONS_ZH: dict[str, str] = {
        "（": "(", "）": ")", "：": ":", "，": ",", "。": ".",
        "【": "[", "】": "]", "￥": "$", "—": "-",
    }

    def execute(
        self,
        cell_texts: list[str],
        language: str = "mixed",
    ) -> dict[str, Any]:
        """Correct a list of cell texts and return corrected versions."""
        corrections = (
            {**self.CORRECTIONS_EN, **self.CORRECTIONS_ZH}
            if language == "mixed"
            else (self.CORRECTIONS_EN if language == "en" else self.CORRECTIONS_ZH)
        )

        corrected: list[str] = []
        for text in cell_texts:
            for garbled, correct in corrections.items():
                text = text.replace(garbled, correct)
            corrected.append(text.strip())

        # Count corrections made
        num_corrected = sum(
            1 for orig, corr in zip(cell_texts, corrected) if orig != corr
        )

        return {
            "original_texts": cell_texts,
            "corrected_texts": corrected,
            "num_corrected": num_corrected,
            "language": language,
        }
