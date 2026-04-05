"""
data_pipeline: DeepDoc document understanding pipeline.

Modules
-------
layout_analyzer         YoLo v8-based page layout detection
text_extractor         Hierarchical Markdown text extraction
table_structure_recovery  CNN-based table structure recognition (TSR)
chart_extractor        Chart semantic description extraction
text_chunker           Hierarchical Markdown text chunking
table_chunker          Table-as-unit chunking with metadata
chart_chunker          Chart semantic description chunking
deepdoc_interface      End-to-end DeepDocPipeline (all modules integrated)
"""

from .deepdoc_interface import (
    DeepDocPipeline,
    ProcessedDocument,
    DocumentChunk,
)
from .layout_analyzer import (
    LayoutAnalyzer,
    LayoutType,
    LayoutPage,
    BBox,
)
from .text_extractor import TextExtractor, MarkdownBlock, MarkdownSpan
from .table_structure_recovery import (
    TableStructureRecovery,
    TableStructure,
    TableCell,
    TSRCNNModel,
)
from .chart_extractor import ChartExtractor, ChartSemanticBlock, ChartType, AxisInfo, DataPoint
from .text_chunker import TextChunker, TextChunk
from .table_chunker import TableChunker, TableChunk
from .chart_chunker import ChartChunker, ChartChunk

__all__ = [
    # Pipeline
    "DeepDocPipeline",
    "ProcessedDocument",
    "DocumentChunk",
    # Layout
    "LayoutAnalyzer",
    "LayoutType",
    "LayoutPage",
    "BBox",
    # Text extraction
    "TextExtractor",
    "MarkdownBlock",
    "MarkdownSpan",
    # Table
    "TableStructureRecovery",
    "TableStructure",
    "TableCell",
    "TSRCNNModel",
    # Chart
    "ChartExtractor",
    "ChartSemanticBlock",
    "ChartType",
    "AxisInfo",
    "DataPoint",
    # Chunking
    "TextChunker",
    "TextChunk",
    "TableChunker",
    "TableChunk",
    "ChartChunker",
    "ChartChunk",
]
