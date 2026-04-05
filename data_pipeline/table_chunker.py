"""
Table Chunker — Table-as-Unit Chunking Strategy.

Implements the DeepDoc table chunking strategy:

1. Table-as-unit: NEVER split a table mid-logic. Each logical table
   becomes exactly one retrieval chunk.
2. Metadata enrichment: attach table name, unit row, page span,
   caption, footnote, and column descriptions to every table chunk.
3. Cross-page table: entire table is preserved as one unit even if
   it spans multiple pages (detected during TSR recovery).
4. Large table fallback: for tables with >50 rows, chunk by logical
   sections (e.g., every 20 rows) but always preserve the header row
   and section headers within each sub-chunk.

Output: list of :class:`TableChunk` ready for embedding and storage.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .table_structure_recovery import TableStructure

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TableChunk:
    """
    A retrieval-ready table chunk with full structural and semantic metadata.

    Attributes
    ----------
    chunk_id : str
    table_id : str
        ID of the source TableStructure.
    text : str
        Markdown representation of the full table (or section of a large table).
    header_row : list[str]
        Column headers.
    body_rows : list[list[str]]
        Table data rows.
    page_numbers : list[int]
        Pages where this table appears.
    is_cross_page : bool
    is_partial : bool
        True if this is a sub-section of a large table (header was repeated).
    partial_section_header : str, optional
        For partial chunks: the row label that started this section.
    metadata : dict
        All associated metadata (table_name, units, footnotes, column descriptions).
    token_count : int
    """

    chunk_id: str
    table_id: str
    text: str
    header_row: list[str]
    body_rows: list[list[str]]
    page_numbers: list[int]
    is_cross_page: bool
    is_partial: bool
    partial_section_header: Optional[str]
    metadata: dict[str, Any]
    token_count: int


# ---------------------------------------------------------------------------
# Table Chunker
# ---------------------------------------------------------------------------

class TableChunker:
    """
    Table-as-unit chunker with large-table sub-section support.

    Parameters
    ----------
    max_rows_per_chunk : int, default 50
        If a table has more rows than this, split into sub-sections
        while preserving the header in each sub-section.
    header_repeat_threshold : int, default 20
        When splitting a large table, repeat the header every N rows
        to preserve context in each chunk.
    include_units_in_header : bool, default True
        If True, attach a "(unit: ...)" annotation to column headers
        based on metadata["column_units"].
    include_footnotes : bool, default True
        Include any footnote text in the chunk metadata.
    """

    MAX_ROWS_PER_CHUNK = 50
    HEADER_REPEAT_THRESHOLD = 20

    def __init__(
        self,
        max_rows_per_chunk: int = 50,
        header_repeat_threshold: int = 20,
        include_units_in_header: bool = True,
        include_footnotes: bool = True,
    ) -> None:
        self.max_rows_per_chunk = max_rows_per_chunk
        self.header_repeat_threshold = header_repeat_threshold
        self.include_units_in_header = include_units_in_header
        self.include_footnotes = include_footnotes

    def chunk_tables(
        self,
        tables: list[TableStructure],
        source_doc: Optional[str] = None,
    ) -> list[TableChunk]:
        """
        Chunk one or more tables into retrieval-ready TableChunks.

        Parameters
        ----------
        tables : list of TableStructure
            Recovered table structures from TableStructureRecovery.
        source_doc : str, optional

        Returns
        -------
        list of TableChunk
        """
        chunks: list[TableChunk] = []

        for table in tables:
            table_chunks = self._chunk_single_table(table, source_doc)
            chunks.extend(table_chunks)

        return chunks

    def _chunk_single_table(
        self,
        table: TableStructure,
        source_doc: Optional[str],
    ) -> list[TableChunk]:
        """
        Chunk a single table (or split if too large).
        """
        chunks: list[TableChunk] = []

        # --- Build metadata dict ---
        metadata = dict(table.metadata)
        metadata["source_doc"] = source_doc or ""
        metadata["table_name"] = table.metadata.get("table_name", "")
        metadata["footnote"] = table.metadata.get("footnote", "")
        metadata["column_units"] = table.metadata.get("column_units", {})

        # --- Check if table needs sub-sectioning ---
        num_data_rows = len(table.body_rows)

        if num_data_rows == 0:
            # Empty table — still emit as one chunk
            chunks.append(
                self._make_chunk(
                    table=table,
                    header_row=table.header_row,
                    body_rows=[],
                    is_partial=False,
                    partial_section_header=None,
                    metadata=metadata,
                )
            )
            return chunks

        if num_data_rows <= self.max_rows_per_chunk:
            # Small enough: one chunk for the whole table
            chunks.append(
                self._make_chunk(
                    table=table,
                    header_row=table.header_row,
                    body_rows=table.body_rows,
                    is_partial=False,
                    partial_section_header=None,
                    metadata=metadata,
                )
            )
        else:
            # Large table: split into sections, repeat header each section
            for section_start in range(0, num_data_rows, self.max_rows_per_chunk):
                section_end = min(section_start + self.max_rows_per_chunk, num_data_rows)
                section_rows = table.body_rows[section_start:section_end]

                # Use the first row's label as section header for context
                section_label = (
                    str(section_rows[0][0]) if section_rows else None
                )

                # Repeat header at the top of each section chunk
                enriched_header = self._enrich_header(
                    table.header_row, metadata, section_start > 0
                )

                chunks.append(
                    self._make_chunk(
                        table=table,
                        header_row=enriched_header,
                        body_rows=section_rows,
                        is_partial=True,
                        partial_section_header=section_label,
                        metadata=metadata,
                    )
                )

        return chunks

    def _make_chunk(
        self,
        table: TableStructure,
        header_row: list[str],
        body_rows: list[list[str]],
        is_partial: bool,
        partial_section_header: Optional[str],
        metadata: dict[str, Any],
    ) -> TableChunk:
        """Build a TableChunk from components."""
        chunk_id = str(uuid.uuid4())[:8]

        # Build Markdown text
        markdown_lines: list[str] = []

        # Optional section header
        if partial_section_header:
            markdown_lines.append(f"**Section: {partial_section_header}**\n")

        # Table name / caption as comment
        if metadata.get("table_name"):
            markdown_lines.append(f"<!-- Table: {metadata['table_name']} -->\n")

        lines = []

        def fmt_cell(s: str) -> str:
            return str(s).replace("|", "\\|")

        header_line = "| " + " | ".join(fmt_cell(h) for h in header_row) + " |"
        sep_line = "| " + " | ".join(["---"] * len(header_row)) + " |"
        lines.append(header_line)
        lines.append(sep_line)

        for row in body_rows:
            lines.append("| " + " | ".join(fmt_cell(str(c)) for c in row) + " |")

        markdown_lines.append("\n".join(lines))

        # Add footnote if present
        if self.include_footnotes and metadata.get("footnote"):
            markdown_lines.append(f"\n*Footnote: {metadata['footnote']}*")

        text = "\n".join(markdown_lines)
        token_count = self._count_tokens(text)

        # Add page span to metadata
        meta = dict(metadata)
        meta["is_cross_page"] = table.is_cross_page
        meta["page_span"] = table.page_numbers
        meta["num_rows"] = len(body_rows)
        meta["num_cols"] = table.num_cols
        if partial_section_header:
            meta["partial_section_header"] = partial_section_header

        return TableChunk(
            chunk_id=chunk_id,
            table_id=table.table_id,
            text=text,
            header_row=header_row,
            body_rows=body_rows,
            page_numbers=table.page_numbers,
            is_cross_page=table.is_cross_page,
            is_partial=is_partial,
            partial_section_header=partial_section_header,
            metadata=meta,
            token_count=token_count,
        )

    def _enrich_header(
        self,
        header_row: list[str],
        metadata: dict[str, Any],
        repeat: bool,
    ) -> list[str]:
        """
        Enrich header row with unit annotations if available.

        Parameters
        ----------
        header_row : list of column header names
        metadata : dict with "column_units" key
        repeat : bool — True if this is a repeated header in a sub-section

        Returns
        -------
        list of enriched header strings
        """
        if not self.include_units_in_header:
            return header_row

        column_units: dict[int, str] = metadata.get("column_units", {})
        enriched = []
        for i, col_name in enumerate(header_row):
            unit = column_units.get(i, "")
            if unit:
                enriched.append(f"{col_name} ({unit})")
            else:
                enriched.append(col_name)

        if repeat:
            # Mark as repeated header in sub-section
            enriched = ["[continued]"] + enriched

        return enriched

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Rough word-based token count."""
        import re
        words = re.findall(r"[\w\.\-\+\%]+", text)
        return len(words)
