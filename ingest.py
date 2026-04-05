"""
ingest.py — PDF Document Ingestion Pipeline.

Processes financial PDFs through the DeepDoc pipeline and saves chunks
to the collection store for retrieval.

Usage
-----
    # Ingest a single PDF
    python ingest.py --pdf /data/apple_2024_annual_report.pdf \\
                     --collection financial_reports \\
                     --embed

    # Ingest multiple PDFs
    python ingest.py --pdf /data/report1.pdf /data/report2.pdf \\
                     --collection financial_reports \\
                     --embed

    # Ingest all PDFs in a directory
    python ingest.py --pdf-dir /data/financial_reports/ \\
                     --collection financial_reports \\
                     --embed

    # Process and show chunks WITHOUT embedding (dry run)
    python ingest.py --pdf /data/report.pdf --show-chunks

    # DeepDoc only — layout analysis and chunking (no embedding yet)
    python ingest.py --pdf /data/report.pdf \\
                     --collection my_collection \\
                     --deepdoc-only

    # Full pipeline with options
    python ingest.py --pdf /data/report.pdf \\
                     --collection financial_reports \\
                     --embed \\
                     --batch-size 32 \\
                     --chunk-max-tokens 512 \\
                     --overlap 64

Output
------
Chunks are saved to: data/collections/{collection_name}.jsonl

Each chunk JSON line:
{
  "chunk_id": "abc123",
  "text": "## 1 Introduction\n\nApple Inc. reported...",
  "source_doc": "apple_2024_annual_report.pdf",
  "page_number": 5,
  "token_count": 342,
  "chunk_type": "text" | "table" | "chart",
  "metadata": {
    "heading_path": ["1 Introduction", "1.1 Overview"],
    "table_name": "Quarterly Revenue",
    "is_cross_page": false
  }
}

Prerequisites
-------------
1. Install dependencies:
    pip install -r requirements.txt

2. For full DeepDoc processing (YoLo v8 + TSR), install:
    pip install ultralytics paddleocr pdf2image torch

3. Set DeepDoc API key if using remote RAGFlow backend:
    export DEEPDOC_API_KEY="your-key"

Note: Without DeepDoc dependencies, the pipeline runs in mock/stub mode
and will output placeholder chunks. Install the above packages for
full functionality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_environment() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    os.makedirs("./data/collections", exist_ok=True)
    os.makedirs("./data/raw", exist_ok=True)


# ---------------------------------------------------------------------------
# DeepDoc Pipeline Wrapper
# ---------------------------------------------------------------------------

def process_pdf_deepdoc(
    pdf_path: str,
    source_doc: Optional[str] = None,
    show_chunks: bool = False,
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Process a PDF through the full DeepDoc pipeline.

    Returns list of chunk dicts ready for embedding + storage.
    """
    from config import get_config
    from data_pipeline import (
        DeepDocPipeline,
        LayoutAnalyzer,
        TextExtractor,
        TableStructureRecovery,
        ChartExtractor,
        TextChunker,
        TableChunker,
        ChartChunker,
    )

    cfg = get_config()
    source_doc = source_doc or os.path.basename(pdf_path)

    print(f"\n[ingest] Processing: {pdf_path}")
    print(f"[ingest] Source doc: {source_doc}")

    # Initialize DeepDoc pipeline components
    try:
        layout_analyzer = LayoutAnalyzer(
            device=kwargs.get("device"),
        )
        text_extractor = TextExtractor()
        table_recovery = TableStructureRecovery()
        chart_extractor = ChartExtractor()
        text_chunker = TextChunker(
            max_tokens=kwargs.get("chunk_max_tokens", cfg.embedding.chunk_max_tokens),
            overlap_tokens=kwargs.get("overlap", cfg.embedding.chunk_overlap),
        )
        table_chunker = TableChunker()
        chart_chunker = ChartChunker()

        pipeline = DeepDocPipeline(
            layout_analyzer=layout_analyzer,
            text_extractor=text_extractor,
            table_recovery=table_recovery,
            chart_extractor=chart_extractor,
            text_chunker=text_chunker,
            table_chunker=table_chunker,
            chart_chunker=chart_chunker,
            device=kwargs.get("device"),
        )

        print("[ingest] DeepDoc pipeline initialized (full mode)")

    except Exception as exc:  # noqa: BLE001
        print(f"[ingest] WARNING: DeepDoc initialization failed: {exc}")
        print("[ingest] Falling back to mock mode (install ultralytics/paddleocr for full mode)")
        return process_pdf_mock(pdf_path, source_doc, show_chunks)

    # Run the full pipeline
    start = time.time()
    try:
        doc = pipeline.process_pdf(pdf_path, source_doc=source_doc)
        elapsed = time.time() - start

        print(f"[ingest] Processed {doc.total_pages} pages in {elapsed:.1f}s")
        print(f"[ingest]   Text chunks:  {len(doc.text_chunks)}")
        print(f"[ingest]   Table chunks: {len(doc.table_chunks)}")
        print(f"[ingest]   Chart chunks: {len(doc.chart_chunks)}")

    except NotImplementedError as exc:
        print(f"[ingest] DeepDoc processing not available: {exc}")
        print("[ingest] Falling back to mock mode for demonstration")
        return process_pdf_mock(pdf_path, source_doc, show_chunks)

    # Convert to flat chunk list
    chunks: list[dict[str, Any]] = []

    for c in doc.text_chunks:
        chunks.append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source_doc": source_doc,
            "page_number": c.page_number,
            "token_count": c.token_count,
            "chunk_type": "text",
            "metadata": c.metadata,
        })

    for c in doc.table_chunks:
        chunks.append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source_doc": source_doc,
            "page_number": c.page_numbers[0] if c.page_numbers else 1,
            "token_count": c.token_count,
            "chunk_type": "table",
            "metadata": {
                "is_cross_page": c.is_cross_page,
                "table_name": c.metadata.get("table_name", ""),
                "page_span": c.page_numbers,
                "num_rows": c.metadata.get("num_rows", 0),
            },
        })

    for c in doc.chart_chunks:
        chunks.append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source_doc": source_doc,
            "page_number": c.page_number,
            "token_count": c.token_count,
            "chunk_type": "chart",
            "metadata": {
                "chart_type": c.chart_type,
                "title": c.title,
            },
        })

    return chunks


# ---------------------------------------------------------------------------
# Mock Mode (no DeepDoc dependencies)
# ---------------------------------------------------------------------------

def process_pdf_mock(
    pdf_path: str,
    source_doc: Optional[str] = None,
    show_chunks: bool = False,
) -> list[dict[str, Any]]:
    """
    Mock PDF processing for demonstration when DeepDoc is not available.

    Creates placeholder chunks from the PDF filename.
    Replace with real DeepDoc processing for production.
    """
    source_doc = source_doc or os.path.basename(pdf_path)

    print(f"\n[ingest] MOCK MODE: Processing {pdf_path}")
    print("[ingest] NOTE: Install DeepDoc dependencies for real PDF processing:")
    print("           pip install ultralytics paddleocr pdf2image torch")
    print("[ingest] Creating placeholder chunks for demonstration...")

    import uuid
    chunks = []

    # Create 5 mock text chunks as demonstration
    mock_sections = [
        ("Executive Summary", "This section provides an executive overview of the company's performance in the fiscal year, including key highlights of revenue, profit margins, and strategic initiatives undertaken during the reporting period."),
        ("Financial Highlights", "Total revenue reached $391 billion, representing a 7% year-over-year increase. Operating income was $115 billion with a margin of 29.4%. Net income attributable to shareholders was $97 billion, up 8% from the prior year."),
        ("Revenue Analysis", "Revenue by product category: iPhone contributed $200 billion, Services grew to $85 billion, Mac reached $35 billion, and Wearables declined to $30 billion. Geographic breakdown shows Americas at $165 billion, Europe at $95 billion, Greater China at $72 billion."),
        ("Balance Sheet", "Total assets stood at $352 billion. Cash and equivalents were $62 billion. Total liabilities were $279 billion. Shareholders' equity increased to $74 billion, reflecting strong profitability and share buyback programs."),
        ("Risk Factors", "The company faces risks including: global economic uncertainty, supply chain disruptions, foreign exchange fluctuations, competitive pressures in key markets, and regulatory challenges in various jurisdictions affecting digital services and data privacy."),
    ]

    for i, (heading, content) in enumerate(mock_sections, start=1):
        chunk_id = str(uuid.uuid4())[:8]
        full_text = f"## {heading}\n\n{content}"
        chunks.append({
            "chunk_id": chunk_id,
            "text": full_text,
            "source_doc": source_doc,
            "page_number": i,
            "token_count": len(full_text.split()),
            "chunk_type": "text",
            "metadata": {
                "section_type": "body",
                "heading_path": [heading],
            },
        })

    print(f"[ingest] Created {len(chunks)} mock chunks")
    return chunks


# ---------------------------------------------------------------------------
# Embedding (using BGE-M3)
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: list[dict[str, Any]],
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """
    Generate embeddings for chunks using BGE-M3.

    Adds 'embedding' field to each chunk dict (list of floats).
    """
    try:
        from FlagEmbedding import BGEM3FlagEmbedding
        from config import get_config

        cfg = get_config()
        print(f"\n[ingest] Generating embeddings with BGE-M3 (batch_size={batch_size})...")

        model = BGEM3FlagEmbedding(
            model_name_or_path=cfg.embedding.model,
            model_kwargs={"device": "cuda" if _cuda_available() else "cpu"},
            encode_kwargs={"batch_size": batch_size, "max_length": cfg.embedding.max_length},
            use_fp16=_cuda_available(),
        )

        texts = [c["text"] for c in chunks]
        start = time.time()
        embeddings = model.encode(texts)
        elapsed = time.time() - start

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings["dense_vecs"][i].tolist()

        print(f"[ingest] Generated {len(chunks)} embeddings in {elapsed:.1f}s")

    except ImportError:
        print("[ingest] WARNING: FlagEmbedding not installed. Run: pip install FlagEmbedding")
        print("[ingest] Skipping embedding generation. Chunks saved without vectors.")

    return chunks


# ---------------------------------------------------------------------------
# Save to Collection
# ---------------------------------------------------------------------------

def save_collection(
    collection_name: str,
    chunks: list[dict[str, Any]],
    output_dir: str = "./data/collections",
) -> str:
    """Save chunks to a JSONL collection file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{collection_name}.jsonl")

    # Remove embeddings before saving (they're large — store separately in production)
    chunks_to_save = []
    for chunk in chunks:
        c = dict(chunk)
        c.pop("embedding", None)  # Remove large vector
        chunks_to_save.append(c)

    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks_to_save:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[ingest] Saved {len(chunks)} chunks to {path}")
    return path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _find_pdf_files(paths_or_dir: list[str]) -> list[str]:
    """Expand directories into PDF file paths."""
    pdf_files: list[str] = []
    for path in paths_or_dir:
        if os.path.isdir(path):
            for f in Path(path).glob("*.pdf"):
                pdf_files.append(str(f))
            for f in Path(path).glob("*.PDF"):
                pdf_files.append(str(f))
        elif os.path.isfile(path) and path.lower().endswith(".pdf"):
            pdf_files.append(path)
    return pdf_files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest financial PDFs into the RAG collection store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pdf",
        nargs="+",
        help="PDF file path(s) or directory(ies) to ingest",
    )
    parser.add_argument(
        "--pdf-dir",
        help="Directory containing PDFs to ingest (alternative to --pdf)",
    )
    parser.add_argument(
        "--collection", "-c",
        default="financial_reports",
        help="Collection name (default: financial_reports)",
    )
    parser.add_argument(
        "--embed", "-e",
        action="store_true",
        help="Generate BGE-M3 embeddings (requires GPU or CPU time)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    parser.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=None,
        help="Max tokens per text chunk (default from config: 512)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Overlap between chunks in tokens (default from config: 64)",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print first 3 chunks after processing (no save)",
    )
    parser.add_argument(
        "--deepdoc-only",
        action="store_true",
        help="Run DeepDoc processing only, skip embedding",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/collections",
        help="Output directory for collection files",
    )

    args = parser.parse_args()

    setup_environment()

    # Collect PDF files
    pdf_paths: list[str] = []
    if args.pdf:
        pdf_paths = _find_pdf_files(args.pdf)
    if args.pdf_dir:
        pdf_paths.extend(_find_pdf_files([args.pdf_dir]))

    if not pdf_paths:
        parser.print_help()
        print("\n[ERROR] No PDF files specified. Use --pdf or --pdf-dir.")
        sys.exit(1)

    print(f"[ingest] Found {len(pdf_paths)} PDF(s) to ingest")
    print(f"[ingest] Collection: {args.collection}")

    kwargs = {
        "batch_size": args.batch_size,
        "chunk_max_tokens": args.chunk_max_tokens,
        "overlap": args.overlap,
    }

    all_chunks: list[dict[str, Any]] = []

    # Process each PDF
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"[ERROR] File not found: {pdf_path}")
            continue

        # Copy PDF to raw storage
        dest_raw = os.path.join("./data/raw", os.path.basename(pdf_path))
        if not os.path.exists(dest_raw):
            import shutil
            os.makedirs("./data/raw", exist_ok=True)
            shutil.copy2(pdf_path, dest_raw)

        chunks = process_pdf_deepdoc(
            pdf_path=pdf_path,
            source_doc=os.path.basename(pdf_path),
            **kwargs,
        )

        if args.show_chunks:
            print("\n[Sample Chunks]")
            for i, c in enumerate(chunks[:3], 1):
                print(f"\n  --- Chunk {i} ({c['chunk_type']}) ---")
                print(f"  ID:    {c['chunk_id']}")
                print(f"  Page:  {c['page_number']}")
                print(f"  Text:  {c['text'][:200]}...")

        all_chunks.extend(chunks)

    if args.show_chunks and not args.deepdoc_only and not args.embed:
        print(f"\n[ingest] Total chunks: {len(all_chunks)}")
        return

    # Embed if requested
    if args.embed and not args.deepdoc_only:
        all_chunks = embed_chunks(all_chunks, batch_size=args.batch_size)

    # Save to collection
    path = save_collection(args.collection, all_chunks, output_dir=args.output_dir)
    print(f"\n[ingest] DONE. Ingested {len(all_chunks)} chunks from {len(pdf_paths)} PDF(s).")
    print(f"[ingest] Collection: {args.collection}")
    print(f"[ingest] To query: python run.py 'your question' --collection {args.collection}")


if __name__ == "__main__":
    main()
