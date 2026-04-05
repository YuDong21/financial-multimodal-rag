"""
data_pipeline: Document ingestion and layout analysis pipeline.
Interfaces with RAGFlow DeepDoc for cross-page table parsing and Markdown conversion.
"""

from .deepdoc_interface import DeepDocClient

__all__ = ["DeepDocClient"]
