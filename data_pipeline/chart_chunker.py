"""
Chart Chunker — Chart Semantic Description Chunking.

Implements the DeepDoc chart chunking strategy:

1. Each detected chart region becomes exactly ONE retrieval chunk.
2. The chunk text = combination of all interpretable signals:
   caption + title + chart type + axis info + data points + description.
3. The chart's own metadata (figure number, source document, page span,
   footnote) is attached as structured metadata.
4. No sub-chunking: charts are self-contained semantic units by nature.
   They are small enough to serve as a single retrieval unit without splitting.

Output: list of :class:`ChartChunk` ready for embedding and storage.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .chart_extractor import ChartSemanticBlock

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ChartChunk:
    """
    A retrieval-ready chart chunk combining all semantic signals.

    Attributes
    ----------
    chunk_id : str
    chart_id : str
        ID of the source ChartSemanticBlock.
    text : str
        Full semantic description text (the retrieval unit).
    chart_type : str
    title : str
    caption : str
    description : str
        Generated natural-language description.
    data_points : list of dict
        Extracted data points (serializable).
    page_number : int
    bbox : tuple
    metadata : dict
        figure_number, source_doc, page_span, footnote, footnote_id, etc.
    token_count : int
    """

    chunk_id: str
    chart_id: str
    text: str
    chart_type: str
    title: str
    caption: str
    description: str
    data_points: list[dict[str, Any]]
    page_number: int
    bbox: tuple[float, float, float, float]
    metadata: dict[str, Any]
    token_count: int


# ---------------------------------------------------------------------------
# Chart Chunker
# ---------------------------------------------------------------------------

class ChartChunker:
    """
    Chart semantic description chunker.

    Each chart → one ChartChunk. No sub-chunking needed since a chart
    is already a self-contained semantic unit.

    Parameters
    ----------
    include_verbose_description : bool, default True
        Include the full generated natural-language description.
    include_data_points : bool, default True
        Include extracted numeric data points in the chunk text.
    include_axis_info : bool, default True
        Include axis labels and ranges in the chunk text.
    max_data_points : int, default 30
        Maximum number of data points to include in the chunk text.
    """

    MAX_DATA_POINTS = 30

    def __init__(
        self,
        include_verbose_description: bool = True,
        include_data_points: bool = True,
        include_axis_info: bool = True,
        max_data_points: int = 30,
    ) -> None:
        self.include_verbose_description = include_verbose_description
        self.include_data_points = include_data_points
        self.include_axis_info = include_axis_info
        self.max_data_points = max_data_points

    def chunk_charts(
        self,
        charts: list[ChartSemanticBlock],
        source_doc: Optional[str] = None,
    ) -> list[ChartChunk]:
        """
        Chunk one or more charts into retrieval-ready ChartChunks.

        Parameters
        ----------
        charts : list of ChartSemanticBlock
            Extracted chart blocks from ChartExtractor.
        source_doc : str, optional
            Source document name (e.g. "apple_2024_annual_report.pdf").

        Returns
        -------
        list of ChartChunk, one per input chart.
        """
        chunks: list[ChartChunk] = []

        for chart in charts:
            chunk = self._chunk_single_chart(chart, source_doc)
            chunks.append(chunk)

        return chunks

    def _chunk_single_chart(
        self,
        chart: ChartSemanticBlock,
        source_doc: Optional[str],
    ) -> ChartChunk:
        """Convert a ChartSemanticBlock into a ChartChunk."""
        chunk_id = str(uuid.uuid4())[:8]

        # Build the full retrieval text
        text_parts: list[str] = []

        # 1: Title
        if chart.title:
            text_parts.append(f"Chart Title: {chart.title}")

        # 2: Type
        text_parts.append(f"Chart Type: {chart.chart_type.value}")

        # 3: Caption
        if chart.caption:
            text_parts.append(f"Caption: {chart.caption}")

        # 4: Axis information
        if self.include_axis_info:
            if chart.x_axis:
                x_vals = ", ".join(str(v) for v in chart.x_axis.values[:10])
                text_parts.append(f"X-Axis ({chart.x_axis.label}): {x_vals}")
            if chart.y_axis:
                y_range = ""
                if chart.y_axis.numeric_range:
                    y_range = f" range [{chart.y_axis.numeric_range[0]}–{chart.y_axis.numeric_range[1]}]"
                y_unit = f" [{chart.y_axis.unit}]" if chart.y_axis.unit else ""
                text_parts.append(f"Y-Axis ({chart.y_axis.label}{y_unit}){y_range}")

        # 5: Data points
        if self.include_data_points and chart.data_points:
            max_dp = min(len(chart.data_points), self.max_data_points)
            dp_strs: list[str] = []
            for dp in chart.data_points[:max_dp]:
                unit = chart.y_axis.unit if chart.y_axis else ""
                val_str = f"{dp.value} {unit}" if dp.value is not None and unit else str(dp.value or dp.label)
                dp_strs.append(f"{dp.label}: {val_str}")
            text_parts.append(f"Data Points: {'; '.join(dp_strs)}")

        # 6: Natural-language description
        if self.include_verbose_description and chart.description:
            text_parts.append(f"Description: {chart.description}")

        # 7: Cross-reference to page
        text_parts.append(f"Source: page {chart.page_number}")

        text = "\n".join(text_parts)
        token_count = self._count_tokens(text)

        # Build metadata
        meta: dict[str, Any] = dict(chart.metadata)
        meta["source_doc"] = source_doc or ""
        meta["figure_number"] = meta.get("figure_number", "")
        meta["footnote"] = meta.get("footnote", "")
        meta["internal_ocr_text"] = meta.get("internal_ocr_text", "")

        # Serialize data points for storage
        data_points_dicts = [
            {
                "label": dp.label,
                "series_name": dp.series_name,
                "value": dp.value,
            }
            for dp in chart.data_points
        ]

        return ChartChunk(
            chunk_id=chunk_id,
            chart_id=chart.block_id,
            text=text,
            chart_type=chart.chart_type.value,
            title=chart.title,
            caption=chart.caption,
            description=chart.description,
            data_points=data_points_dicts,
            page_number=chart.page_number,
            bbox=chart.bbox,
            metadata=meta,
            token_count=token_count,
        )

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Rough word-based token count."""
        import re
        words = re.findall(r"[\w\.\-\+\%]+", text)
        return len(words)
