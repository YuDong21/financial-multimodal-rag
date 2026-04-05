"""
Chart Extractor — Semantic Description from Chart Regions.

Extracts semantic information from detected CHART regions in financial PDFs:
  1. OCR text inside the chart (data labels, axis values, legend text)
  2. Axis information (labels, ranges, tick values)
  3. Associated Caption text (searched in surrounding page regions)
  4. Chart type inference (bar, line, pie, scatter)
  5. Combines all into a single semantic description block with metadata.

Output is a structured :class:`ChartSemanticBlock` that can be chunked
and retrieved as a standalone semantic unit.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ChartType(str, Enum):
    """Inferred chart type."""

    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class AxisInfo:
    """
    Information about an axis in a chart.
    """

    label: str
    values: list[str]  # Tick labels as strings
    numeric_range: Optional[tuple[float, float]] = None  # (min, max)
    unit: Optional[str] = None  # e.g. "B USD", "%", "x"


@dataclass
class DataPoint:
    """
    A single data point extracted from a chart (OCR + axis reading).
    """

    label: str  # e.g. "Q1 2024" or "Apple"
    series_name: Optional[str] = None  # e.g. "Revenue" for multi-series charts
    value: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None


@dataclass
class ChartSemanticBlock:
    """
    Full semantic description of a chart region.

    This is the retrieval unit for chart information — not just the image,
    but a complete textual representation combining all interpretable signals.

    Attributes
    ----------
    block_id : str
        Unique identifier.
    chart_type : ChartType
        Inferred chart type.
    title : str
        Chart title (from caption or axis label).
    caption : str
        Surrounding explanatory text.
    description : str
        Generated natural-language description of what the chart shows.
    data_points : list[DataPoint]
        Extracted numeric data points.
    x_axis : AxisInfo
        X-axis metadata.
    y_axis : AxisInfo
        Y-axis metadata.
    page_number : int
        Source page.
    bbox : tuple[float, float, float, float]
        Chart bounding box in PDF points.
    metadata : dict
        Additional info (source document name, figure number, footnote, etc.).
    token_count : int
        Approximate token count.
    """

    block_id: str
    chart_type: ChartType
    title: str
    caption: str
    description: str
    data_points: list[DataPoint] = field(default_factory=list)
    x_axis: Optional[AxisInfo] = None
    y_axis: Optional[AxisInfo] = None
    page_number: int = 1
    bbox: tuple[float, float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0, 0.0)
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    def to_text(self) -> str:
        """
        Render as a single text string for embedding / retrieval.

        This is what gets stored in the vector database.
        """
        parts: list[str] = []

        if self.title:
            parts.append(f"Chart Title: {self.title}")

        if self.caption:
            parts.append(f"Caption: {self.caption}")

        parts.append(f"Chart Type: {self.chart_type.value}")

        if self.x_axis:
            x_label = self.x_axis.label
            x_values = ", ".join(str(v) for v in self.x_axis.values[:10])
            parts.append(f"X-Axis ({x_label}): {x_values}")

        if self.y_axis:
            y_label = self.y_axis.label
            y_unit = f" [{self.y_axis.unit}]" if self.y_axis.unit else ""
            parts.append(f"Y-Axis ({y_label}{y_unit}): {self.y_axis.numeric_range}")

        if self.data_points:
            points_str = "; ".join(
                f"{dp.label}={dp.value}{self._unit_str(dp.series_name)}"
                for dp in self.data_points[:20]
            )
            parts.append(f"Data Points: {points_str}")

        parts.append(f"Description: {self.description}")

        return "\n".join(parts)

    def _unit_str(self, series_name: Optional[str]) -> str:
        if self.y_axis and self.y_axis.unit:
            return f" {self.y_axis.unit}"
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "chart_type": self.chart_type.value,
            "title": self.title,
            "caption": self.caption,
            "description": self.description,
            "page_number": self.page_number,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Chart Type Classifier
# ---------------------------------------------------------------------------

class ChartTypeClassifier:
    """
    Heuristic chart type classifier based on visual shape analysis.

    Analyzes the layout of bars, lines, pie slices, or dots in the chart
    image to determine its type. Replace with a fine-tuned CNN classifier
    in production.
    """

    def classify(self, chart_image: np.ndarray) -> ChartType:
        """
        Classify the chart type from its image.

        Heuristics:
        - If height >> width per element → likely bar chart
        - If connected trend elements → likely line chart
        - If radial layout → pie chart
        - Otherwise → unknown
        """
        raise NotImplementedError(
            "ChartTypeClassifier requires a vision model.\n"
            "Options:\n"
            "  1. Fine-tune a ResNet/EfficientNet on chart type classification\n"
            "  2. Use OpenCV contour analysis for shape detection\n"
            "  3. Use a VLM (vision language model) for zero-shot classification\n"
            "For financial charts, bar/line/pie are most common."
        )


# ---------------------------------------------------------------------------
# Chart Extractor
# ---------------------------------------------------------------------------

class ChartExtractor:
    """
    Extracts semantic information from a chart region.

    Combines:
    - Internal OCR: data labels, axis ticks, legend text
    - Axis parsing: label, range, unit
    - Caption search: nearby text regions on the same page
    - Chart type inference
    - Description generation

    Parameters
    ----------
    ocr_engine : callable, optional
        OCR function: ``ocr_engine(image) -> str``.
    caption_search_radius : float, default 20.0
        Max points to search above the chart bbox for a caption.
    """

    COMMON_CHART_TITLE_KEYWORDS = {
        "figure", "fig.", "fig", "chart", "exhibit",
        "图", "图表", "示意图", "折线图", "柱状图", "饼图",
    }

    def __init__(
        self,
        ocr_engine: Optional[Any] = None,
        caption_search_radius: float = 20.0,
    ) -> None:
        self.ocr_engine = ocr_engine
        self.caption_search_radius = caption_search_radius

    def extract(
        self,
        chart_image: np.ndarray,
        page_regions: Optional[list[dict]] = None,
        bbox: Optional[tuple[float, float, float, float]] = None,
        page_number: int = 1,
    ) -> ChartSemanticBlock:
        """
        Extract semantic content from a chart image.

        Parameters
        ----------
        chart_image : np.ndarray
            Cropped RGB image of the chart region.
        page_regions : list of dict, optional
            Nearby TEXT regions from the same page (for caption search).
            Each dict: {"text": str, "bbox": tuple, "type": str}
        bbox : tuple, optional
            PDF coordinates of the chart region.
        page_number : int

        Returns
        -------
        ChartSemanticBlock
        """
        import uuid as uuid_lib

        block_id = str(uuid_lib.uuid4())[:8]

        # 1: Run OCR on the chart image
        internal_text = self._ocr_chart_image(chart_image)

        # 2: Parse axis information
        x_axis, y_axis = self._parse_axes(internal_text)

        # 3: Extract data points (if numeric data is visible)
        data_points = self._extract_data_points(internal_text, y_axis)

        # 4: Infer chart type
        try:
            chart_type = self._classify_chart_type(chart_image, internal_text)
        except Exception:  # noqa: BLE001
            chart_type = ChartType.UNKNOWN

        # 5: Find associated caption
        caption = self._find_caption(page_regions, bbox) if page_regions else ""

        # 6: Extract title (often in caption or axis label)
        title = self._extract_title(internal_text, caption)

        # 7: Generate natural-language description
        description = self._generate_description(chart_type, title, data_points, x_axis, y_axis)

        block = ChartSemanticBlock(
            block_id=block_id,
            chart_type=chart_type,
            title=title,
            caption=caption,
            description=description,
            data_points=data_points,
            x_axis=x_axis,
            y_axis=y_axis,
            page_number=page_number,
            bbox=bbox or (0, 0, 0, 0),
            metadata={"internal_ocr_text": internal_text},
            token_count=0,
        )
        block.token_count = self._count_tokens(block.to_text())

        return block

    def _ocr_chart_image(self, chart_image: np.ndarray) -> str:
        """Run OCR on the chart image to extract internal text."""
        if self.ocr_engine is None:
            # Return empty — production should provide an OCR engine
            return ""
        return self.ocr_engine(chart_image)

    def _parse_axes(self, text: str) -> tuple[Optional[AxisInfo], Optional[AxisInfo]]:
        """
        Parse axis labels and tick values from the OCR text.

        Heuristic: look for patterns like "X-Axis: Revenue [B USD]"
        or "Y: 0  100  200  300".
        """
        import re

        x_axis: Optional[AxisInfo] = None
        y_axis: Optional[AxisInfo] = None

        # Look for axis label patterns
        x_match = re.search(r"(?i)x[- ]axis[:\s]+([^\nY]+)", text)
        y_match = re.search(r"(?i)y[- ]axis[:\s]+([^\nX]+)", text)

        # Look for numeric sequences that look like tick values
        numbers = re.findall(r"\b[\d,]+\.?\d*\b", text)
        numeric_values = []
        for n in numbers:
            try:
                numeric_values.append(float(n.replace(",", "")))
            except ValueError:
                continue

        unit_match = re.search(r"\[([^\]]+)\]", text)
        unit = unit_match.group(1) if unit_match else None

        if numeric_values:
            y_axis = AxisInfo(
                label="Value",
                values=[str(v) for v in numeric_values],
                numeric_range=(min(numeric_values), max(numeric_values)),
                unit=unit,
            )

        x_axis = AxisInfo(label="Category", values=[], unit=None)

        return x_axis, y_axis

    def _extract_data_points(
        self, text: str, y_axis: Optional[AxisInfo]
    ) -> list[DataPoint]:
        """Extract labeled data points from OCR text."""
        import re

        data_points: list[DataPoint] = []
        # Pattern: "Label: value" or "Label value"
        patterns = [
            r"([A-Za-z0-9\u4e00-\u9fa5]+\s*)[:\s]+([\d,]+\.?\d*)\s*([A-Z]{2,})?",
        ]

        for pat in patterns:
            matches = re.finditer(pat, text)
            for m in matches:
                label = m.group(1).strip()
                try:
                    value = float(m.group(2).replace(",", ""))
                except ValueError:
                    continue
                unit = m.group(3) if m.lastindex >= 3 else None

                dp = DataPoint(
                    label=label,
                    value=value,
                    series_name=unit,
                )
                data_points.append(dp)

        return data_points[:30]  # Cap at 30 points

    def _classify_chart_type(
        self, chart_image: np.ndarray, ocr_text: str
    ) -> ChartType:
        """Classify chart type from image + OCR text cues."""
        import re

        text_lower = ocr_text.lower()

        # Keyword-based heuristic
        if any(k in text_lower for k in ["bar", "柱状", "柱", "销售额", "revenue", "profit"]):
            return ChartType.BAR
        if any(k in text_lower for k in ["line", "折线", "趋势", "trend", "growth"]):
            return ChartType.LINE
        if any(k in text_lower for k in ["pie", "饼", "比例", "占比", "share", "%"]):
            return ChartType.PIE
        if any(k in text_lower for k in ["scatter", "散点", "correlation"]):
            return ChartType.SCATTER

        return ChartType.UNKNOWN

    def _find_caption(
        self,
        page_regions: list[dict],
        chart_bbox: Optional[tuple[float, float, float, float]],
    ) -> str:
        """Search nearby TEXT regions for a caption associated with this chart."""
        if chart_bbox is None:
            return ""

        chart_top = chart_bbox[1]  # y1 (top of chart)
        search_top = max(0.0, chart_top - self.caption_search_radius)

        candidates: list[tuple[float, str]] = []
        for region in page_regions:
            if region.get("type") != "text":
                continue
            bbox = region.get("bbox")
            if bbox is None:
                continue
            # Check if region is just above the chart
            if bbox[2] > search_top and bbox[3] < chart_top:
                candidates.append((bbox[2], region["text"]))

        if not candidates:
            return ""

        # Return the closest region to the chart (highest y2 value)
        closest = max(candidates, key=lambda x: x[0])
        return closest[1]

    def _extract_title(self, internal_text: str, caption: str) -> str:
        """Extract the chart title from caption or internal text."""
        combined = f"{caption}\n{internal_text}"
        lines = combined.split("\n")

        for line in lines[:5]:  # Check first few lines
            line = line.strip()
            if len(line) < 3 or len(line) > 200:
                continue
            first_word_lower = line.lower().split()[0] if line.split() else ""
            if any(k in line.lower() for k in self.COMMON_CHART_TITLE_KEYWORDS):
                # Clean up title (remove "Figure 1:" prefix)
                import re
                cleaned = re.sub(r"^(figure|fig\.|fig|chart|exhibit)\s*\d*[:\.\s]*", "", line, flags=re.IGNORECASE)
                return cleaned.strip()
        return ""

    def _generate_description(
        self,
        chart_type: ChartType,
        title: str,
        data_points: list[DataPoint],
        x_axis: Optional[AxisInfo],
        y_axis: Optional[AxisInfo],
    ) -> str:
        """Generate a natural-language description of the chart."""
        parts: list[str] = []

        if title:
            parts.append(f"This {chart_type.value} chart titled '{title}'")

        y_label = y_axis.label if y_axis else "value"
        y_unit = f" in {y_axis.unit}" if y_axis and y_axis.unit else ""

        if data_points:
            if chart_type == ChartType.BAR:
                vals = [f"{dp.label}: {dp.value}{y_unit}" for dp in data_points[:5]]
                parts.append(f"shows bars with values: {', '.join(vals)}")
            elif chart_type == ChartType.LINE:
                parts.append(f"shows a line trend for {y_label}{y_unit} across: {', '.join(dp.label for dp in data_points[:5])}")
            elif chart_type == ChartType.PIE:
                parts.append(f"shows proportional distribution: {', '.join(f'{dp.label}={dp.value}' for dp in data_points[:5])}")
            else:
                parts.append(f"contains {len(data_points)} data points for {y_label}{y_unit}")

        if x_axis and x_axis.label and x_axis.label != "Category":
            parts.append(f"X-axis: {x_axis.label}")
        if y_axis and y_axis.label and y_axis.label != "Value":
            parts.append(f"Y-axis: {y_axis.label}{y_unit}")

        return ". ".join(parts) if parts else f"This is a {chart_type.value} chart."

    @staticmethod
    def _count_tokens(text: str) -> int:
        return len(text.split())
