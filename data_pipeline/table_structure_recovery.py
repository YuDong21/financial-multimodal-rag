"""
Table Structure Recovery (TSR) — CNN-based Table Structure Recognition.

Recovers the logical structure of a detected table region:
  - Row / column grid detection
  - Merged cell spanning (colspan, rowspan)
  - Header row identification
  - Column header / value type inference
  - Cross-page table stitching

The recovered structure is serialized as a Markdown table with proper
row separators and cell alignment markers.

Architecture
------------
A CNN model (e.g. TableMaster, TableNet, or a fine-tuned ResNet) takes
the cropped table image as input and outputs:
  1. A binary cell grid mask (each cell = 1 region)
  2. Row-separator heatmap
  3. Column-separator heatmap
  4. Cell-header classification (bool per cell)

These outputs are then decoded into a logical table representation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TableCell:
    """
    A single logical cell in a recovered table.
    """

    row: int
    col: int
    text: str
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False
    bbox: Optional[tuple[float, float, float, float]] = None
    confidence: float = 1.0


@dataclass
class TableStructure:
    """
    Fully recovered logical table structure.

    Attributes
    ----------
    table_id : str
        Unique identifier.
    header_row : list[str]
        Text content of the header row cells.
    body_rows : list[list[str]]
        Text content of each body row.
    num_rows : int
    num_cols : int
    page_numbers : list[int]
        Pages where this table appears.
    bbox : tuple[float, float, float, float]
        Bounding box across all pages.
    is_cross_page : bool
        True if this table spans multiple pages.
    spanning_cells : list[TableCell]
        Cells that have rowspan > 1 or colspan > 1.
    metadata : dict
        Additional context: table_caption, unit_row, footnote_text, etc.
    raw_markdown : str
        Rendered Markdown representation.
    ocr_corrected : bool
        True if OCR correction was applied to cell text.
    """

    table_id: str
    header_row: list[str]
    body_rows: list[list[str]]
    num_rows: int
    num_cols: int
    page_numbers: list[int]
    bbox: tuple[float, float, float, float]
    is_cross_page: bool
    spanning_cells: list[TableCell] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_markdown: str = ""
    ocr_corrected: bool = False


# ---------------------------------------------------------------------------
# TSR Model Stub
# ---------------------------------------------------------------------------

class TSRCNNModel:
    """
    CNN model for Table Structure Recognition.

    This is a STUB — replace ``_infer`` with your actual model inference
    when a trained model is available.

    Expected model output shape:
      - cell_mask : (H', W') binary grid — one value per table cell
      - row_seps  : (H',) — probability of row separator after each grid row
      - col_seps  : (W',) — probability of col separator after each grid col
      - header_mask : (H', W') bool — True for header cells

    You can replace this stub with:
      - TableNet (https://github.com/asimshankar/table-net)
      - TableMaster (PaddleOCR)
      - A fine-tuned ResNet/FPN trained on PubTables1M
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or "tsr_model.pth"
        self._model = None  # Loaded lazily

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Placeholder — in production, load your trained model here:
        #   self._model = torch.load(self.model_path, map_location="cpu")
        #   self._model.eval()
        raise NotImplementedError(
            "TSRCNNModel requires a trained TSR model checkpoint.\n"
            "Train a table structure recognition model and set model_path.\n"
            "Suggested architectures:\n"
            "  - TableNet (ResNet50 backbone + decoder)\n"
            "  - TableMaster (PaddleOCR)\n"
            "  - PubTables1M-trained Faster R-CNN\n"
            "See: https://github.com/ibm-aur-nlp/table-net"
        )

    def infer(
        self, table_image: np.ndarray
    ) -> dict[str, Any]:
        """
        Run TSR inference on a cropped table image.

        Parameters
        ----------
        table_image : np.ndarray
            RGB image of the table region (H × W × 3).

        Returns
        -------
        dict with keys: cell_mask, row_seps, col_seps, header_mask
        """
        self._ensure_loaded()
        raise NotImplementedError("TSRCNNModel.infer called before model loading.")

    def detect_merged_cells(
        self, cell_mask: np.ndarray, row_seps: np.ndarray, col_seps: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """
        Decode merged cells (rowspan/colspan) from separator heatmaps.

        Returns
        -------
        list of (row, col, rowspan, colspan) tuples.
        """
        raise NotImplementedError("TSRCNNModel.detect_merged_cells called before model loading.")


# ---------------------------------------------------------------------------
# OCR Cell Text Extractor
# ---------------------------------------------------------------------------

class CellOCRExtractor:
    """
    Extracts text from individual table cells using OCR.

    Applies financial-specific post-processing:
      - Currency symbol normalization (¥, $)
      - Number format standardization (1,000.00 vs 1.000,00)
      - Common financial abbreviation expansion
    """

    # Common financial OCR corrections (garbled → correct)
    OCR_CORRECTIONS: dict[str, str] = {
        "—": "-",  # em dash → hyphen
        "–": "-",  # en dash → hyphen
        """: '"', """: '"',
        "'": "'",  "'": "'",
        "．": ".",  "，": ",",  "（": "(", "）": ")",
        "％": "%",  "¥": "$",  "￥": "$",
        "—": "-",
    }

    def __init__(self, ocr_engine: Optional[Any] = None) -> None:
        """
        Parameters
        ----------
        ocr_engine : callable, optional
            OCR function: ``ocr_engine(cell_image: np.ndarray) -> str``.
            If omitted, uses a placeholder that raises NotImplementedError.
        """
        self.ocr_engine = ocr_engine

    def extract_cell_text(self, cell_image: np.ndarray) -> str:
        """
        Extract text from a single table cell image.

        Parameters
        ----------
        cell_image : np.ndarray
            Cropped image of the cell region.

        Returns
        -------
        str : the recognized text.
        """
        if self.ocr_engine is None:
            raise NotImplementedError(
                "CellOCRExtractor requires an OCR engine.\n"
                "Provide an ocr_engine=callable or install paddleocr/pytesseract.\n"
                "Example:\n"
                "  from paddleocr import PaddleOCR\n"
                "  ocr = PaddleOCR(lang='en')\n"
                "  extractor = CellOCRExtractor(ocr_engine=lambda img: ocr.ocr(img)[0][0][1])"
            )

        raw = self.ocr_engine(cell_image)
        return self._post_process(raw)

    def _post_process(self, text: str) -> str:
        """Apply OCR corrections for common financial garbling."""
        for garbled, correct in self.OCR_CORRECTIONS.items():
            text = text.replace(garbled, correct)

        # Normalize number formats (preserve first encountered convention)
        # Don't auto-convert — leave as-is for downstream parsers to handle
        return text.strip()


# ---------------------------------------------------------------------------
# Table Structure Recovery Pipeline
# ---------------------------------------------------------------------------

class TableStructureRecovery:
    """
    Full TSR pipeline: takes a table-region image → logical Markdown table.

    Pipeline steps:
      1. Cell grid decoding from TSR CNN outputs
      2. OCR text extraction per cell
      3. Merged cell (rowspan/colspan) resolution
      4. Header row identification
      5. Markdown serialization
      6. Cross-page stitching (if needed)

    Parameters
    ----------
    tsr_model : TSRCNNModel, optional
        CNN-based table structure recognition model.
    ocr_extractor : CellOCRExtractor, optional
    cross_page_mode : str, default "infer"
        How to handle cross-page tables: "infer", "force_merge", "split".
    """

    def __init__(
        self,
        tsr_model: Optional[TSRCNNModel] = None,
        ocr_extractor: Optional[CellOCRExtractor] = None,
        cross_page_mode: str = "infer",
    ) -> None:
        self.tsr_model = tsr_model or TSRCNNModel()
        self.ocr = ocr_extractor or CellOCRExtractor()
        self.cross_page_mode = cross_page_mode
        self._tsr_outputs: dict[str, Any] = {}

    def recover(
        self,
        table_images: list[np.ndarray],
        page_numbers: Optional[list[int]] = None,
        table_bboxes: Optional[list[Any]] = None,
    ) -> TableStructure:
        """
        Recover the logical structure of a table from its image(s).

        Parameters
        ----------
        table_images : list of np.ndarray
            Cropped table image(s). If a table spans multiple pages,
            provide images for each page fragment in order.
        page_numbers : list of int, optional
            Corresponding page numbers for each table image fragment.
        table_bboxes : list of BBox, optional
            Bounding boxes for each page fragment.

        Returns
        -------
        TableStructure
        """
        import uuid as uuid_lib

        table_id = str(uuid_lib.uuid4())[:8]
        page_numbers = page_numbers or [1]
        is_cross_page = len(table_images) > 1

        # Step 1: TSR inference (if model is loaded)
        try:
            tsr_outputs = [
                self.tsr_model.infer(img) for img in table_images
            ]
        except NotImplementedError:
            tsr_outputs = [{} for _ in table_images]  # Empty fallback

        # Step 2: Cell grid decoding + OCR
        all_cells: list[list[TableCell]] = []
        num_cols = 0

        for page_idx, img in enumerate(table_images):
            try:
                page_cells, page_cols = self._decode_cells(
                    img, tsr_outputs[page_idx] if page_idx < len(tsr_outputs) else {},
                    page_offset=len(all_cells),
                    page_num=page_numbers[page_idx] if page_idx < len(page_numbers) else 1,
                )
                all_cells.append(page_cells)
                num_cols = max(num_cols, page_cols)
            except Exception:  # noqa: BLE001
                # Fallback: treat as raw text table with no structure
                all_cells.append([])
                num_cols = max(num_cols, 0)

        # Step 3: Merge cross-page tables
        if is_cross_page and self.cross_page_mode in ("infer", "force_merge"):
            merged = self._merge_cross_page_tables(all_cells, num_cols)
            header_row = merged[0] if merged else []
            body_rows = merged[1:] if len(merged) > 1 else []
        else:
            # Flatten all page cells into rows
            flat_cells: list[TableCell] = []
            for page_cells in all_cells:
                flat_cells.extend(page_cells)
            header_row = [cell.text for cell in flat_cells[:num_cols]]
            body_rows = []
            row_start = num_cols
            while row_start + num_cols <= len(flat_cells):
                row = [flat_cells[row_start + c].text for c in range(num_cols)]
                body_rows.append(row)
                row_start += num_cols

        # Step 4: Build metadata
        metadata: dict[str, Any] = {}
        if is_cross_page:
            metadata["cross_page"] = True
            metadata["page_span"] = page_numbers

        # Step 5: Render Markdown
        raw_markdown = self._to_markdown(header_row, body_rows)

        # Step 6: Merge with spanning cells
        spanning_cells = [c for cells in all_cells for c in cells if c.rowspan > 1 or c.colspan > 1]

        total_bbox = self._merge_bboxes([b.bbox for b in table_bboxes if b is not None]) if table_bboxes else (0, 0, 0, 0)

        return TableStructure(
            table_id=table_id,
            header_row=header_row,
            body_rows=body_rows,
            num_rows=len(body_rows) + 1,
            num_cols=num_cols,
            page_numbers=page_numbers,
            bbox=total_bbox,  # type: ignore
            is_cross_page=is_cross_page,
            spanning_cells=spanning_cells,
            metadata=metadata,
            raw_markdown=raw_markdown,
            ocr_corrected=False,
        )

    def _decode_cells(
        self,
        table_image: np.ndarray,
        tsr_output: dict[str, Any],
        page_offset: int,
        page_num: int,
    ) -> tuple[list[TableCell], int]:
        """
        Decode cell grid from TSR outputs and run OCR per cell.

        Returns (cells, num_cols).
        """
        # STUB: Without a real TSR model, use heuristic grid division
        h, w = table_image.shape[:2]

        # Simple heuristic: divide image into a rough N×M grid
        # Replace with actual TSR model decoding in production
        N = 10  # max rows (placeholder)
        M = 8   # max cols (placeholder)

        cell_h = h / N
        cell_w = w / M

        cells: list[TableCell] = []
        num_cols = M

        for row in range(N):
            for col in range(M):
                y1 = int(row * cell_h)
                y2 = int((row + 1) * cell_h)
                x1 = int(col * cell_w)
                x2 = int((col + 1) * cell_w)

                cell_img = table_image[y1:y2, x1:x2]
                if cell_img.size == 0:
                    continue

                try:
                    text = self.ocr.extract_cell_text(cell_img)
                except NotImplementedError:
                    text = ""  # OCR not available

                is_header = (row == 0)
                bbox = (float(x1), float(y1), float(x2), float(y2))

                cells.append(
                    TableCell(
                        row=page_offset + row,
                        col=col,
                        text=text,
                        rowspan=1,
                        colspan=1,
                        is_header=is_header,
                        bbox=bbox,
                        confidence=1.0,
                    )
                )

        return cells, num_cols

    def _merge_cross_page_tables(
        self, page_cells: list[list[TableCell]], num_cols: int
    ) -> list[list[str]]:
        """
        Merge cells from multiple page fragments into a single table.

        Heuristic: deduplicate the header row on page 2+ (since it's repeated
        across pages), then concatenate body rows.
        """
        if not page_cells:
            return []

        # Take header from first page
        first_header = [c.text for c in page_cells[0][:num_cols]]
        all_rows = [first_header]

        for page_idx, cells in enumerate(page_cells[1:], start=2):
            # Skip repeated header row on subsequent pages
            row_data: list[str] = []
            for c in cells[:num_cols]:
                row_data.append(c.text)
            if row_data and row_data != first_header:
                all_rows.append(row_data)
            # Process remaining rows
            for i in range(num_cols, len(cells), num_cols):
                row_data = [cells[i + j].text if i + j < len(cells) else "" for j in range(num_cols)]
                all_rows.append(row_data)

        return all_rows

    @staticmethod
    def _to_markdown(header: list[str], body: list[list[str]]) -> str:
        """Serialize a table as Markdown."""
        lines: list[str] = []

        def fmt_cell(s: str) -> str:
            return s.replace("|", "\\|")

        header_line = "| " + " | ".join(fmt_cell(h) for h in header) + " |"
        sep_line = "| " + " | ".join(["---"] * len(header)) + " |"
        lines.append(header_line)
        lines.append(sep_line)

        for row in body:
            row_line = "| " + " | ".join(fmt_cell(str(c)) for c in row) + " |"
            lines.append(row_line)

        return "\n".join(lines)

    @staticmethod
    def _merge_bboxes(
        bboxes: list[tuple[float, float, float, float]],
    ) -> tuple[float, float, float, float]:
        """Merge multiple bboxes into one covering all."""
        if not bboxes:
            return (0, 0, 0, 0)
        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)
        return (x1, y1, x2, y2)
