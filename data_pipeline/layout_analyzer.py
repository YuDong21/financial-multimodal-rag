"""
Layout Analyzer — YoLo v8-based Page Layout Detection.

Detects semantic regions in a financial PDF page using a YoLo v8 model:
  - TEXT    : Body text, headings, paragraphs
  - TABLE   : Table regions (with cross-page detection)
  - CHART   : Chart / figure regions
  - IMAGE   : Pure images (ignored for RAG)
  - HEADER/FOOTER : Page headers/footers (filtered out)

Each detected region is returned as a :class:`BBox` with category, confidence,
and page coordinates.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LayoutType(str, Enum):
    """Semantic region types detected by the layout analyzer."""

    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """
    Axis-aligned bounding box for a detected layout region.

    Coordinates are in PDF points (1/72 inch), origin at top-left.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    label: LayoutType
    confidence: float
    page_number: int

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def iou(self, other: BBox) -> float:
        """
        Compute Intersection over Union with another BBox.

        Returns 0 if boxes do not overlap.
        """
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)

        inter_w = max(0.0, xi2 - xi1)
        inter_h = max(0.0, yi2 - yi1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def contains(self, other: BBox) -> bool:
        """Check if this BBox fully contains another."""
        return (
            self.x1 <= other.x1
            and self.y1 <= other.y1
            and self.x2 >= other.x2
            and self.y2 >= other.y2
        )

    def overlaps_page_boundary(self, page_height: float, page_width: float, margin: float = 5.0) -> bool:
        """Detect if this region touches the top or bottom page boundary (potential cross-page table)."""
        return self.y1 < margin or self.y2 > page_height - margin


@dataclass
class LayoutPage:
    """
    Full layout analysis result for a single PDF page.
    """

    page_number: int
    page_width: float
    page_height: float
    bboxes: list[BBox] = field(default_factory=list)
    is_cross_page_table_suspected: bool = False

    def regions_of(self, *labels: LayoutType) -> list[BBox]:
        """Return all regions matching the given label(s)."""
        return [b for b in self.bboxes if b.label in labels]

    def text_regions(self) -> list[BBox]:
        return self.regions_of(LayoutType.TEXT)

    def table_regions(self) -> list[BBox]:
        return self.regions_of(LayoutType.TABLE)

    def chart_regions(self) -> list[BBox]:
        return self.regions_of(LayoutType.CHART)


# ---------------------------------------------------------------------------
# YoLo v8 Layout Analyzer
# ---------------------------------------------------------------------------

class LayoutAnalyzer:
    """
    YoLo v8-based page layout analyzer.

    Detects TEXT / TABLE / CHART / IMAGE / HEADER / FOOTER regions
    in each page of a financial PDF.

    Parameters
    ----------
    model_path : str, optional
        Path to a fine-tuned YoLo v8 weights file.
        Defaults to ``"yolov8-layout.pt"`` (to be provided by user).
    confidence_threshold : float, default 0.5
        Minimum confidence score to accept a detection.
    nms_iou_threshold : float, default 0.4
        IoU threshold for Non-Maximum Suppression.
    device : str, optional
        "cuda" or "cpu". Auto-detected if omitted.
    """

    # YoLo v8 class index → LayoutType mapping
    # (User should replace these with their fine-tuned model class indices)
    CLASS_MAP: dict[int, LayoutType] = {
        0: LayoutType.TEXT,
        1: LayoutType.TABLE,
        2: LayoutType.CHART,
        3: LayoutType.IMAGE,
        4: LayoutType.HEADER,
        5: LayoutType.FOOTER,
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.4,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path or "yolov8-layout.pt"
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self._model: Optional[Any] = None

    def _load_model(self) -> Any:
        """
        Load the YoLo v8 model.

        Raises NotImplementedError with instructions since we cannot
        import ultralytics without it being installed.
        """
        raise NotImplementedError(
            "LayoutAnalyzer requires the ultralytics package.\n"
            "  pip install ultralytics\n"
            "\n"
            "Place your fine-tuned YoLo v8 weights at the path specified "
            "in model_path (default: yolov8-layout.pt).\n"
            "Training instructions for financial document layout detection:\n"
            "  https://docs.ultralytics.com/tasks/detect/"
        )

    def analyze_page(self, page_image: "np.ndarray") -> LayoutPage:
        """
        Analyze a single page image and return detected layout regions.

        Parameters
        ----------
        page_image : np.ndarray
            RGB image of the PDF page (H × W × 3), e.g. from pdf2image.

        Returns
        -------
        LayoutPage with detected BBoxes.
        """
        raise NotImplementedError(
            "LayoutAnalyzer.analyze_page requires the ultralytics package. "
            "Install with: pip install ultralytics"
        )

    def analyze_pages(
        self, page_images: list["np.ndarray"]
    ) -> list[LayoutPage]:
        """
        Batch-analyze multiple page images.

        Parameters
        ----------
        page_images : list of np.ndarray

        Returns
        -------
        list of LayoutPage, one per input image.
        """
        return [self.analyze_page(img) for img in page_images]

    def detect_cross_page_tables(
        self, pages: list[LayoutPage]
    ) -> list[tuple[BBox, BBox]]:
        """
        Identify pairs of table regions that likely belong to the same
        logical table split across adjacent pages.

        Heuristic: a table is "cross-page" if its bottom region
        touches the page boundary and the next page's top table region
        has similar column structure.

        Parameters
        ----------
        pages : list of LayoutPage

        Returns
        -------
        list of (bottom_region, top_region) tuples for suspected cross-page tables.
        """
        pairs: list[tuple[BBox, BBox]] = []
        MARGIN = 5.0  # points

        for i in range(len(pages) - 1):
            curr_page = pages[i]
            next_page = pages[i + 1]

            for curr_tbl in curr_page.table_regions():
                if not curr_tbl.overlaps_page_boundary(
                    curr_page.page_height, curr_page.page_width, MARGIN
                ):
                    continue

                # Find table at top of next page
                for next_tbl in next_page.table_regions():
                    if next_tbl.y1 > MARGIN:
                        continue
                    # Heuristic: similar width ratio → same table
                    if abs(curr_tbl.width - next_tbl.width) / max(curr_tbl.width, 1) < 0.1:
                        pairs.append((curr_tbl, next_tbl))
                        break

        return pairs


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
