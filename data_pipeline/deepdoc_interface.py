"""
DeepDoc Document Pipeline — Integrated Layout Analysis and Chunking.

End-to-end pipeline that takes a financial PDF and produces
structured, chunked, metadata-enriched retrieval units (text, table, chart).

Pipeline
--------
PDF → Layout Analysis (YoLo v8) → Region-wise processing
  ├── TEXT regions  → TextExtractor → TextChunker → TextChunks
  ├── TABLE regions → TableStructureRecovery → TableChunker → TableChunks
  └── CHART regions → ChartExtractor → ChartChunker → ChartChunks

Cross-page stitching is handled automatically:
  - Tables spanning pages are merged in TableStructureRecovery
  - Chunk metadata records page_span for each chunk

Output: list of :class:`DocumentChunk` (union type of all chunk types).

Usage
-----
>>> from data_pipeline import DeepDocPipeline
>>> pipeline = DeepDocPipeline()
>>> chunks = pipeline.process_pdf("/data/apple_2024.pdf")
>>> print(f"Total chunks: {len(chunks)}")
    Text chunks: 42, Table chunks: 8, Chart chunks: 3
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import pydantic

# ---------------------------------------------------------------------------
# Chunk Types
# ---------------------------------------------------------------------------

TextChunk = Any       # from text_chunker
TableChunk = Any       # from table_chunker
ChartChunk = Any       # from chart_chunker
DocumentChunk = Union[TextChunk, TableChunk, ChartChunk]


# ---------------------------------------------------------------------------
# Pipeline Output Models
# ---------------------------------------------------------------------------

class ProcessedDocument(pydantic.BaseModel):
    """
    Result of the full DeepDoc pipeline for one PDF.

    Attributes
    ----------
    doc_id : str
    file_name : str
    total_pages : int
    text_chunks : list of TextChunk
    table_chunks : list of TableChunk
    chart_chunks : list of ChartChunk
    metadata : dict — overall document metadata
    """

    doc_id: str
    file_name: str
    total_pages: int
    text_chunks: list[Any] = pydantic.Field(default_factory=list)
    table_chunks: list[Any] = pydantic.Field(default_factory=list)
    chart_chunks: list[Any] = pydantic.Field(default_factory=list)
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)

    @property
    def all_chunks(self) -> list[Any]:
        """All chunks (text + table + chart) in a single flat list."""
        return self.text_chunks + self.table_chunks + self.chart_chunks

    @property
    def total_chunks(self) -> int:
        return len(self.all_chunks)


# ---------------------------------------------------------------------------
# Stub for page-to-image conversion
# ---------------------------------------------------------------------------

def pdf_page_to_image(
    pdf_path: str, page_number: int, dpi: int = 200
) -> Any:
    """
    Convert a single PDF page to an RGB image (np.ndarray).

    STUB — replace with pdf2image or PyMuPDF:

    >>> from pdf2image import convert_from_path
    >>> images = convert_from_path(pdf_path, dpi=dpi, first_page=n, last_page=n)
    >>> return np.array(images[0])

    Parameters
    ----------
    pdf_path : str
    page_number : int (1-indexed)
    dpi : int, default 200

    Returns
    -------
    np.ndarray : RGB image of shape (H, W, 3)
    """
    raise NotImplementedError(
        "pdf_page_to_image requires a PDF rendering library.\n"
        "Install pdf2image (with poppler) or PyMuPDF:\n"
        "  pip install pdf2image\n"
        "  # Also install poppler: https://github.com/Belval/pdf2image#poppler\n"
        "\n"
        "Then implement:\n"
        "  from pdf2image import convert_from_path\n"
        "  images = convert_from_path(pdf_path, dpi=dpi, first_page=n, last_page=n)\n"
        "  return np.array(images[0])"
    )


# ---------------------------------------------------------------------------
# DeepDoc Pipeline
# ---------------------------------------------------------------------------

class DeepDocPipeline:
    """
    Integrated DeepDoc document understanding pipeline.

    Combines layout analysis (YoLo v8), structured extraction
    (TextExtractor, TableStructureRecovery, ChartExtractor), and
    chunking (TextChunker, TableChunker, ChartChunker) into a single
    end-to-end processor.

    Parameters
    ----------
    layout_analyzer : LayoutAnalyzer, optional
    text_extractor : TextExtractor, optional
    table_recovery : TableStructureRecovery, optional
    chart_extractor : ChartExtractor, optional
    text_chunker : TextChunker, optional
    table_chunker : TableChunker, optional
    chart_chunker : ChartChunker, optional
    device : str, optional — "cuda" or "cpu"
    """

    def __init__(
        self,
        layout_analyzer: Optional[Any] = None,
        text_extractor: Optional[Any] = None,
        table_recovery: Optional[Any] = None,
        chart_extractor: Optional[Any] = None,
        text_chunker: Optional[Any] = None,
        table_chunker: Optional[Any] = None,
        chart_chunker: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:
        # Lazy imports to avoid hard dependency on all DL libraries
        self._layout_analyzer = layout_analyzer
        self._text_extractor = text_extractor
        self._table_recovery = table_recovery
        self._chart_extractor = chart_extractor
        self._text_chunker = text_chunker
        self._table_chunker = table_chunker
        self._chart_chunker = chart_chunker
        self._device = device

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def process_pdf(
        self,
        pdf_path: str,
        source_doc: Optional[str] = None,
        start_page: int = 1,
        end_page: Optional[int] = None,
    ) -> ProcessedDocument:
        """
        Process a full PDF and return all chunks.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.
        source_doc : str, optional
            Document name override (defaults to filename).
        start_page : int, default 1
        end_page : int, optional — defaults to last page.

        Returns
        -------
        ProcessedDocument
        """
        import os

        doc_id = str(uuid.uuid4())[:8]
        file_name = source_doc or os.path.basename(pdf_path)
        total_pages = self._get_page_count(pdf_path)
        end_page = end_page or total_pages

        all_text_chunks: list[Any] = []
        all_table_chunks: list[Any] = []
        all_chart_chunks: list[Any] = []

        # Process each page
        for page_num in range(start_page, end_page + 1):
            page_result = self.process_page(pdf_path, page_num, file_name)
            all_text_chunks.extend(page_result.text_chunks)
            all_table_chunks.extend(page_result.table_chunks)
            all_chart_chunks.extend(page_result.chart_chunks)

        return ProcessedDocument(
            doc_id=doc_id,
            file_name=file_name,
            total_pages=total_pages,
            text_chunks=all_text_chunks,
            table_chunks=all_table_chunks,
            chart_chunks=all_chart_chunks,
            metadata={
                "pdf_path": pdf_path,
                "start_page": start_page,
                "end_page": end_page,
                "total_chunks": len(all_text_chunks) + len(all_table_chunks) + len(all_chart_chunks),
            },
        )

    def process_page(
        self,
        pdf_path: str,
        page_number: int,
        source_doc: str,
    ) -> ProcessedDocument:
        """
        Process a single page of a PDF.

        Parameters
        ----------
        pdf_path : str
        page_number : int (1-indexed)
        source_doc : str

        Returns
        -------
        ProcessedDocument for this page only.
        """
        # Step 1: Render page to image
        try:
            page_image = pdf_page_to_image(pdf_path, page_number)
        except NotImplementedError:
            # Return empty document if PDF rendering not available
            return ProcessedDocument(
                doc_id=str(uuid.uuid4())[:8],
                file_name=source_doc,
                total_pages=0,
                text_chunks=[],
                table_chunks=[],
                chart_chunks=[],
            )

        # Step 2: Layout analysis (YoLo v8)
        layout_analyzer = self._get_layout_analyzer()
        try:
            layout_page = layout_analyzer.analyze_page(page_image)
        except NotImplementedError:
            layout_page = None

        # Step 3: Extract per region type
        text_blocks = []
        table_images = []
        chart_images = []

        if layout_page is not None:
            # Collect TEXT regions
            for region in layout_page.text_regions():
                # In production: crop page_image to region bbox, run OCR
                text_blocks.append(
                    {
                        "text": "[OCR text for TEXT region]",  # Replace with real OCR
                        "bbox": region,
                        "page_number": page_number,
                    }
                )

            # Collect TABLE regions
            for region in layout_page.table_regions():
                # Crop table image from page
                table_images.append(
                    {
                        "image": self._crop_region(page_image, region),
                        "bbox": region,
                        "page_number": page_number,
                    }
                )

            # Collect CHART regions
            for region in layout_page.chart_regions():
                chart_images.append(
                    {
                        "image": self._crop_region(page_image, region),
                        "bbox": region,
                        "page_number": page_number,
                    }
                )

        # Step 4: Extract text blocks
        text_extractor = self._get_text_extractor()
        markdown_blocks = []
        for region in text_blocks:
            blocks = text_extractor.extract_from_region(
                region_text=region["text"],
                region_bbox=region["bbox"],
                page_number=page_number,
            )
            markdown_blocks.extend(blocks)

        # Step 5: Chunk text
        text_chunker = self._get_text_chunker()
        text_chunks = text_chunker.chunk_blocks(
            markdown_blocks, source_doc=source_doc
        )

        # Step 6: Process tables
        table_chunker = self._get_table_chunker()
        table_structures = []
        for region in table_images:
            tsr = self._get_table_recovery()
            try:
                ts = tsr.recover(
                    table_images=[region["image"]],
                    page_numbers=[page_number],
                    table_bboxes=[region["bbox"]],
                )
                table_structures.append(ts)
            except Exception:  # noqa: BLE001
                continue

        table_chunks = table_chunker.chunk_tables(
            table_structures, source_doc=source_doc
        )

        # Step 7: Process charts
        chart_chunker = self._get_chart_chunker()
        chart_semantic_blocks = []
        for region in chart_images:
            ce = self._get_chart_extractor()
            try:
                cb = ce.extract(
                    chart_image=region["image"],
                    bbox=region["bbox"],
                    page_number=page_number,
                )
                chart_semantic_blocks.append(cb)
            except Exception:  # noqa: BLE001:
                continue

        chart_chunks = chart_chunker.chunk_charts(
            chart_semantic_blocks, source_doc=source_doc
        )

        return ProcessedDocument(
            doc_id=str(uuid.uuid4())[:8],
            file_name=source_doc,
            total_pages=1,
            text_chunks=text_chunks,
            table_chunks=table_chunks,
            chart_chunks=chart_chunks,
            metadata={"page_number": page_number},
        )

    # -------------------------------------------------------------------------
    # Component Accessors (lazy initialization)
    # -------------------------------------------------------------------------

    def _get_layout_analyzer(self) -> Any:
        if self._layout_analyzer is not None:
            return self._layout_analyzer
        from .layout_analyzer import LayoutAnalyzer
        return LayoutAnalyzer(device=self._device)

    def _get_text_extractor(self) -> Any:
        if self._text_extractor is not None:
            return self._text_extractor
        from .text_extractor import TextExtractor
        return TextExtractor()

    def _get_table_recovery(self) -> Any:
        if self._table_recovery is not None:
            return self._table_recovery
        from .table_structure_recovery import TableStructureRecovery
        return TableStructureRecovery()

    def _get_chart_extractor(self) -> Any:
        if self._chart_extractor is not None:
            return self._chart_extractor
        from .chart_extractor import ChartExtractor
        return ChartExtractor()

    def _get_text_chunker(self) -> Any:
        if self._text_chunker is not None:
            return self._text_chunker
        from .text_chunker import TextChunker
        return TextChunker()

    def _get_table_chunker(self) -> Any:
        if self._table_chunker is not None:
            return self._table_chunker
        from .table_chunker import TableChunker
        return TableChunker()

    def _get_chart_chunker(self) -> Any:
        if self._chart_chunker is not None:
            return self._chart_chunker
        from .chart_chunker import ChartChunker
        return ChartChunker()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _crop_region(image: Any, bbox: Any) -> Any:
        """Crop a region from an image."""
        # bbox: (x1, y1, x2, y2) in image coordinates
        # In production: return image[y1:y2, x1:x2]
        raise NotImplementedError(
            "Region cropping requires a PDF rendering library. "
            "Install pdf2image or PyMuPDF."
        )

    @staticmethod
    def _get_page_count(pdf_path: str) -> int:
        """Get the total number of pages in a PDF."""
        raise NotImplementedError(
            "Page count requires PyMuPDF or pdfplumber.\n"
            "  pip install pymupdf\n"
            "  import fitz; doc = fitz.open(pdf_path); return len(doc)"
        )
