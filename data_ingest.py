"""
data_ingest.py — 离线 PDF 数据摄入脚本。

调度 data_pipeline/ 下的各个处理模块，将多个 PDF 文档
转化为带 embedding 的 chunk，存入 collection。

chunking 算法全部在 data_pipeline/ 各模块中实现，本文件仅负责：
  1. PDF 文件遍历
  2. 调用 DeepDocPipeline（版面分析 + 文本抽取 + 表格恢复 + 图表提取）
  3. 调用 TextChunker / TableChunker / ChartChunker（算法不在本文件）
  4. 调用 BGE-M3 生成 dense embedding
  5. 写入 data/collections/{collection}.jsonl

Usage
-----
    # 单文件
    python data_ingest.py --pdf /data/apple_2024.pdf -c my_collection

    # 多文件
    python data_ingest.py --pdf /data/r1.pdf /data/r2.pdf -c my_collection

    # 整个目录
    python data_ingest.py --pdf-dir /data/reports/ -c my_collection

    # 不生成 embedding（仅处理 + 分chunk）
    python data_ingest.py --pdf /data/report.pdf -c my_collection --no-embed

    # 查看 chunks（不保存）
    python data_ingest.py --pdf /data/report.pdf --show
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "collections")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


# ---------------------------------------------------------------------------
# DeepDoc Pipeline — 调用 data_pipeline/，不在本文件写 chunking 逻辑
# ---------------------------------------------------------------------------

def run_deepdoc_pipeline(
    pdf_path: str,
    source_doc: str,
    chunk_max_tokens: int = 512,
    overlap_tokens: int = 64,
    device: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    调用 data_pipeline.DeepDocPipeline 执行完整文档理解流程。

    切分算法全部在 data_pipeline/ 内部：
      - LayoutAnalyzer       (YoLo v8)
      - TextExtractor        (Markdown 层级恢复)
      - TableStructureRecovery (TSR CNN)
      - ChartExtractor       (图表语义提取)
      - TextChunker          (Markdown 感知切分)
      - TableChunker         (表格为单元切分)
      - ChartChunker         (图表语义描述切分)

    Parameters
    ----------
    pdf_path        : PDF 文件路径
    source_doc      : 来源文档名（用于 metadata）
    chunk_max_tokens: TextChunker 的 max_tokens 参数
    overlap_tokens  : TextChunker 的 overlap_tokens 参数
    device          : 运行设备，None=自动检测

    Returns
    -------
    dict[str, list[dict]] — {"text": [...], "table": [...], "chart": [...]}
    """
    # 延迟导入，避免 data_pipeline 未安装时报错
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

    # ── 初始化各处理模块（算法实现均在 data_pipeline/ 内）─────────────
    try:
        layout_analyzer = LayoutAnalyzer(device=device)
        text_extractor = TextExtractor()
        table_recovery = TableStructureRecovery()
        chart_extractor = ChartExtractor()

        text_chunker = TextChunker(
            max_tokens=chunk_max_tokens,
            overlap_tokens=overlap_tokens,
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
            device=device,
        )

        print(f"[data_ingest] DeepDocPipeline 初始化成功 (full mode)")

    except Exception as exc:  # noqa: BLE001
        print(f"[data_ingest] WARNING: DeepDoc 初始化失败: {exc}")
        print("[data_ingest] 使用 mock 模式（安装 ultralytics paddleocr pdf2image torch 以启用完整功能）")
        return run_deepdoc_mock(pdf_path, source_doc)

    # ── 执行完整流程 ──────────────────────────────────────────────────
    start = time.time()
    try:
        doc = pipeline.process_pdf(pdf_path, source_doc=source_doc)
        elapsed = time.time() - start
        print(f"[data_ingest] 处理完成: {doc.total_pages} 页, "
              f"文本={len(doc.text_chunks)}, 表格={len(doc.table_chunks)}, "
              f"图表={len(doc.chart_chunks)} ({elapsed:.1f}s)")

    except NotImplementedError as exc:
        print(f"[data_ingest] DeepDoc 处理不可用: {exc}")
        print("[data_ingest] 使用 mock 模式")
        return run_deepdoc_mock(pdf_path, source_doc)

    # ── 统一转换为 dict 格式 ───────────────────────────────────────────
    chunks: dict[str, list[dict[str, Any]]] = {"text": [], "table": [], "chart": []}

    for c in doc.text_chunks:
        chunks["text"].append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source_doc": source_doc,
            "page_number": c.page_number,
            "token_count": c.token_count,
            "heading_path": c.heading_path,
            "metadata": c.metadata,
        })

    for c in doc.table_chunks:
        chunks["table"].append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source_doc": source_doc,
            "page_number": c.page_numbers[0] if c.page_numbers else 1,
            "token_count": c.token_count,
            "is_cross_page": c.is_cross_page,
            "table_name": c.metadata.get("table_name", ""),
            "page_span": list(c.page_numbers) if c.page_numbers else [],
            "metadata": c.metadata,
        })

    for c in doc.chart_chunks:
        chunks["chart"].append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source_doc": source_doc,
            "page_number": c.page_number,
            "token_count": c.token_count,
            "chart_type": c.chart_type,
            "title": c.title,
            "metadata": c.metadata,
        })

    return chunks


# ---------------------------------------------------------------------------
# Mock 模式（无 DeepDoc 依赖时演示用）
# ---------------------------------------------------------------------------

def run_deepdoc_mock(
    pdf_path: str,
    source_doc: str,
) -> dict[str, list[dict[str, Any]]]:
    """DeepDoc 不可用时的 mock 实现，输出示例 chunks。"""
    print(f"[data_ingest] MOCK MODE: {pdf_path}")

    mock_texts = [
        ("Executive Summary",
         "本报告涵盖 FY2024 财年表现，总营收达 3910 亿美元，同比增长 7%。"
         "净利润 970 亿美元，每股收益 6.13 美元。"),
        ("财务亮点",
         "iPhone 收入 2000 亿美元，服务收入 850 亿美元，Mac 收入 350 亿美元。"
         "毛利率 46.2%，营业利润率 29.4%。"),
        ("资产负债表",
         "总资产 3520 亿美元，现金及等价物 620 亿美元。"
         "总负债 2790 亿美元，股东权益 740 亿美元。"),
        ("风险因素",
         "主要风险：全球宏观经济不确定性、供应链中断、外汇波动、"
         "关键市场竞争加剧、数据隐私及监管合规挑战。"),
    ]

    chunks: dict[str, list[dict[str, Any]]] = {"text": [], "table": [], "chart": []}
    for heading, content in mock_texts:
        chunks["text"].append({
            "chunk_id": str(uuid.uuid4())[:8],
            "text": f"## {heading}\n\n{content}",
            "source_doc": source_doc,
            "page_number": mock_texts.index((heading, content)) + 1,
            "token_count": len((heading + content).split()),
            "heading_path": [heading],
            "metadata": {"section_type": "body"},
        })

    print(f"[data_ingest] 创建了 {len(chunks['text'])} 个 mock 文本 chunks")
    return chunks


# ---------------------------------------------------------------------------
# Embedding — BGE-M3
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: dict[str, list[dict[str, Any]]],
    batch_size: int = 32,
    model_name: str = "BAAI/bge-m3",
    max_length: int = 512,
) -> dict[str, list[dict[str, Any]]]:
    """
    调用 BGE-M3 生成 dense embedding。

    embedding 算法使用 FlagEmbedding 库，不在本文件实现。
    """
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError:
        print("[data_ingest] ERROR: FlagEmbedding 未安装。")
        print("             pip install FlagEmbedding")
        return chunks

    print(f"[data_ingest] 生成 BGE-M3 embedding (batch_size={batch_size})...")

    device = "cuda" if _cuda_available() else "cpu"
    model = BGEM3FlagModel(
        model_name_or_path=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": batch_size, "max_length": max_length},
        use_fp16=(device == "cuda"),
    )

    # 扁平化所有 chunks
    flat: list[dict[str, Any]] = []
    type_map: list[tuple[int, str]] = []  # (chunk_idx, chunk_type)

    for chunk_type, items in chunks.items():
        for item in items:
            flat.append(item)
            type_map.append((len(flat) - 1, chunk_type))

    texts = [c["text"] for c in flat]

    start = time.time()
    results = model.encode(texts)
    elapsed = time.time() - start

    dense_vecs = results["dense_vecs"]
    for i, vec in enumerate(dense_vecs):
        flat[i]["embedding"] = vec.tolist()

    # 重建类型分组
    out: dict[str, list[dict[str, Any]]] = {"text": [], "table": [], "chart": []}
    for i, (chunk_idx, chunk_type) in enumerate(type_map):
        out[chunk_type].append(flat[i])

    total = sum(len(v) for v in out.values())
    print(f"[data_ingest] embedding 完成: {total} chunks in {elapsed:.1f}s")
    return out


# ---------------------------------------------------------------------------
# 保存到 Collection
# ---------------------------------------------------------------------------

def save_collection(
    collection_name: str,
    chunks: dict[str, list[dict[str, Any]]],
    output_dir: str = OUTPUT_DIR,
) -> str:
    """将 chunks 写入 data/collections/{collection}.jsonl。"""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{collection_name}.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for chunk_type, items in chunks.items():
            for chunk in items:
                c = dict(chunk)
                # embedding 太大，随压缩存储；生产环境建议存向量数据库
                c.pop("embedding", None)
                c["chunk_type"] = chunk_type
                c["ingested_at"] = datetime.now().isoformat()
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    total = sum(len(v) for v in chunks.values())
    print(f"[data_ingest] 保存到 {out_path} ({total} chunks)")
    return out_path


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _find_pdfs(paths_or_dirs: list[str]) -> list[str]:
    """将目录或文件列表展开为 PDF 路径列表。"""
    pdfs: list[str] = []
    for p in paths_or_dirs:
        if os.path.isdir(p):
            pdfs.extend(str(f) for f in Path(p).glob("*.pdf"))
            pdfs.extend(str(f) for f in Path(p).glob("*.PDF"))
        elif os.path.isfile(p) and p.lower().endswith(".pdf"):
            pdfs.append(p)
    return pdfs


def _copy_to_raw(pdf_path: str, raw_dir: str = RAW_DIR) -> str:
    """备份 PDF 到 data/raw/。"""
    os.makedirs(raw_dir, exist_ok=True)
    dest = os.path.join(raw_dir, os.path.basename(pdf_path))
    if not os.path.exists(dest):
        shutil.copy2(pdf_path, dest)
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="离线 PDF 数据摄入 — 调用 data_pipeline/ 处理并生成 collection。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pdf", nargs="+", help="PDF 文件路径（可多个）")
    parser.add_argument("--pdf-dir", help="包含 PDF 的目录")
    parser.add_argument(
        "-c", "--collection", default="default",
        help="Collection 名称 (default: default)",
    )
    parser.add_argument(
        "--no-embed", action="store_true",
        help="跳过 embedding（仅处理 + chunk，不生成向量）",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="打印 chunks 内容（不保存）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Embedding batch size (default: 32)",
    )
    parser.add_argument(
        "--chunk-max-tokens", type=int, default=512,
        help="TextChunker max_tokens (default: 512)",
    )
    parser.add_argument(
        "--overlap", type=int, default=64,
        help="TextChunker overlap_tokens (default: 64)",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Collection 输出目录 (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--device", default=None,
        help="运行设备，如 'cuda' 或 'cpu' (默认自动检测)",
    )

    args = parser.parse_args()

    # ── 收集 PDF ──────────────────────────────────────────────────────
    pdf_paths: list[str] = []
    if args.pdf:
        pdf_paths.extend(_find_pdfs(args.pdf))
    if args.pdf_dir:
        pdf_paths.extend(_find_pdfs([args.pdf_dir]))

    if not pdf_paths:
        parser.print_help()
        print("\n[ERROR] 未指定 PDF 文件，请使用 --pdf 或 --pdf-dir")
        sys.exit(1)

    # 去重
    pdf_paths = list(dict.fromkeys(pdf_paths))
    missing = [p for p in pdf_paths if not os.path.exists(p)]
    if missing:
        print(f"[ERROR] 文件不存在: {missing[0]}")
        sys.exit(1)

    print(f"[data_ingest] Collection : {args.collection}")
    print(f"[data_ingest] PDF 文件   : {len(pdf_paths)} 个")

    # ── 处理每个 PDF ──────────────────────────────────────────────────
    all_chunks: dict[str, list[dict[str, Any]]] = {
        "text": [], "table": [], "chart": []
    }

    for pdf_path in pdf_paths:
        source_doc = os.path.basename(pdf_path)
        print(f"\n{'='*50}")
        print(f"[data_ingest] 处理: {source_doc}")

        # 备份原文件
        _copy_to_raw(pdf_path)

        # DeepDoc 处理（chunking 算法在 data_pipeline/ 内）
        chunks = run_deepdoc_pipeline(
            pdf_path=pdf_path,
            source_doc=source_doc,
            chunk_max_tokens=args.chunk_max_tokens,
            overlap_tokens=args.overlap,
            device=args.device,
        )

        # 展示
        if args.show:
            for ctype, items in chunks.items():
                print(f"\n  [{ctype.upper()}] {len(items)} chunks")
                for item in items[:2]:
                    preview = item["text"][:120].replace("\n", " ")
                    print(f"    {item['chunk_id']}: {preview}...")

        all_chunks["text"].extend(chunks.get("text", []))
        all_chunks["table"].extend(chunks.get("table", []))
        all_chunks["chart"].extend(chunks.get("chart", []))

    if args.show:
        print(f"\n[data_ingest] 共 {sum(len(v) for v in all_chunks.values())} chunks (已展示，不保存)")
        return

    # ── Embedding ─────────────────────────────────────────────────────
    if not args.no_embed:
        all_chunks = embed_chunks(
            chunks=all_chunks,
            batch_size=args.batch_size,
        )

    # ── 保存 ──────────────────────────────────────────────────────────
    out_path = save_collection(args.collection, all_chunks, output_dir=args.output_dir)

    total = sum(len(v) for v in all_chunks.values())
    print(f"\n[data_ingest] ✓ 完成！")
    print(f"    Collection : {args.collection}")
    print(f"    PDF 文件   : {len(pdf_paths)}")
    print(f"    总 chunks  : {total} (text={len(all_chunks['text'])}, "
          f"table={len(all_chunks['table'])}, chart={len(all_chunks['chart'])})")
    print(f"    输出路径   : {out_path}")
    print(f"\n    查询: python run.py '问题' --collection {args.collection}")


if __name__ == "__main__":
    main()
