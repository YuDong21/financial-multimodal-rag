# financial-multimodal-rag

> **A Multimodal RAG Smart Assistant for Financial Reports based on LangGraph and Hybrid Retrieval.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

---

## 🎯 Project Overview

**`financial-multimodal-rag`** is an intelligent question-answering system designed specifically for Chinese and English financial reports (annual reports, quarterly reports, ESG reports, IPO filings, etc.). It addresses two core pain points in real-world financial document understanding:

1. **Cross-page table parsing & garbled text recovery** — leveraging RAGFlow's DeepDoc layout analysis and Markdown normalization pipeline.
2. **Long-text retrieval & lost-in-the-middle syndrome** — mitigated by a hybrid retrieval strategy (dense + sparse + reranking) combined with a citation-grounded generation architecture.

The system is built on **LangGraph** as the orchestration substrate, giving it transparent, inspectable reasoning graphs where every node (routing, retrieval, generation) is a first-class citizen.

---

## 🏗️ Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────┐
│   Semantic Router           │  Qwen3-0.6B 二分类
│   (Qwen3-0.6B)              │  fast  ──► Fast Path
│                             │  slow  ──► Slow Path
└─────────────────────────────┘
              │
    ┌─────────┴──────────┐
    ▼                  ▼
═════════════════════════════════════
FAST PATH                          SLOW PATH
─────────────────────────────────────
fast_retrieval                     task_decomposition
  (单 query，轻量检索)              (多子任务拆解)
       │                                 │
fast_rerank                        retrieval_planning
  (top_k=6)                        (确定主策略 + 构建 query)
       │                                 │
generation_fast                    slow_retrieval
  (Qwen3-8B, no thinking,         (multi-query, top_k=8)
   temp=0.1, max_tokens=512)             │
       │                           slow_rerank
       │                                 │
   END ─────────────────────────► generation_slow
                                    (Qwen3-8B, thinking=True,
                                     temp=0.3, max_tokens=1024)
                                          │
                                    verification
                                          │
                                     END
═════════════════════════════════════
```

---

## 📂 Project Structure

```
financial-multimodal-rag/
├── config.py                      # ⭐ Central parameter configuration (single source of truth)
├── run.py                         # ⭐ Main entry point — run queries against a collection
├── ingest.py                      # ⭐ PDF ingestion pipeline — process PDFs into chunks
├── data_pipeline/                 # DeepDoc document understanding pipeline
│   ├── __init__.py
│   ├── layout_analyzer.py         # YoLo v8 page layout detection
│   ├── text_extractor.py          # Hierarchical Markdown text extraction
│   ├── table_structure_recovery.py # CNN-based table structure recognition (TSR)
│   ├── chart_extractor.py         # Chart semantic description extraction
│   ├── text_chunker.py            # Markdown-aware text chunking
│   ├── table_chunker.py           # Table-as-unit chunking
│   ├── chart_chunker.py           # Chart semantic description chunking
│   └── deepdoc_interface.py      # Integrated DeepDocPipeline
├── retrieval/                     # Hybrid retrieval engine
│   ├── __init__.py
│   └── hybrid_retriever.py        # BGE-M3 + BM25 + RRRF + BGE-Reranker
├── graph/                          # LangGraph workflow orchestration
│   ├── __init__.py
│   ├── state.py                   # ⭐ GraphState — global shared state schema
│   └── workflow.py                # ⭐ Complete 3-path state machine
├── models/                         # LLM interface wrappers
│   ├── __init__.py
│   └── qwen_llm.py               # QwenRouter / QwenGeneratorFast / QwenGeneratorSlow / QwenSummarizer
├── memory/                         # Token budget & context management
│   ├── __init__.py
│   └── context_manager.py        # ⭐ TokenBudgetManager + SemanticTruncator + TruncationNode
├── mcp_tools/                      # MCP protocol toolchain (4 categories)
│   ├── __init__.py
│   ├── base.py                   # MCPTool abstract base class
│   ├── deepdoc_tools.py          # Category 1: DeepDoc parsing
│   ├── retrieval_tools.py        # Category 2: Retrieval augmentation
│   ├── analysis_tools.py          # Category 3: Financial analysis
│   ├── verification_tools.py      # Category 4: Result verification
│   ├── mcp_server.py            # ⭐ Python MCP Server (stdio, auto-discovers tools)
│   └── mcp_client.py             # ⭐ MCP Client (LangGraph integration)
├── evaluation/                     # Evaluation framework
│   ├── __init__.py
│   └── ragas_eval.py             # RAGAS Faithfulness evaluation
├── requirements.txt
└── README.md
```

---

## 🔑 Core Features

### 🔀 Adaptive Semantic Routing
- **Model**: Qwen3-0.6B (lightweight, low-latency)
- **Logic**: Classifies each query into `simple` or `complex`
  - `simple` → direct generation (bypass retrieval, ~10× faster)
  - `complex` → full LangGraph pipeline with hybrid retrieval + reranking

### 📊 Hybrid Retrieval (RRRF Fusion)
1. **Dense Retrieval** — BGE-M3 embedding (multi-lingual, 1024 dims)
2. **Sparse Retrieval** — BM25 with analyzer tuned for financial terminology
3. **Fusion** — Reciprocal Rank Fusion (RRRF) to combine rank signals
4. **Reranking** — BGE-Reranker v2 to surface the most contextually relevant chunks

### 🧠 Context Memory Management
- **Sliding Window**: Fixes token budget by maintaining a rolling window over conversation history
- **Semantic Truncation**: When the window overflows, semantically compress older turns using Qwen2.5 summarization model

### 🛠️ MCP Tool Integration
- **DeepDoc Tool**: Calls RAGFlow DeepDoc API for layout analysis + Markdown conversion (handles cross-page tables, OCR correction)
- **Financial Calculator Tool**: Evaluates financial formulas (e.g., ROE, EBITDA, debt-to-equity ratios) using parsed table data

### 📐 Graph-State Orchestration (LangGraph)
Every node is an isolated Python function. The state dict carries:
- `question`, `route`, `sub_tasks`, `retrieved_docs`, `reranked_docs`
- `evidence_sufficient`, `tool_calls`, `answer`, `conversation_history`

Conditional edges handle the fast/slow path branching and loop-back on insufficient evidence.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YuDong21/financial-multimodal-rag.git
cd financial-multimodal-rag
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
export DEEPDOC_API_KEY="your-deepdoc-api-key"   # optional
```

### 3. Ingest PDF Documents

```bash
# Ingest a single PDF (mock mode — no DeepDoc deps needed for demo)
python ingest.py --pdf /path/to/annual_report.pdf \
                 --collection financial_reports \
                 --embed

# Ingest multiple PDFs from a directory
python ingest.py --pdf-dir ./data/pdfs/ \
                 --collection financial_reports \
                 --embed

# Dry run — see chunks without saving
python ingest.py --pdf /path/to/report.pdf --show-chunks
```

**Note:** Without DeepDoc dependencies installed, `ingest.py` runs in **mock mode** and generates placeholder chunks. For real PDF processing, install:
```bash
pip install ultralytics paddleocr pdf2image torch
```

### 4. Run the RAG Pipeline

```bash
# Single question
python run.py "What was Apple's total revenue in FY2024?" \
    --collection financial_reports

# Interactive REPL
python run.py --interactive --collection financial_reports

# With retrieval details shown
python run.py "Compare Apple vs Microsoft revenue" \
    --collection financial_reports \
    --show-retrieval
```

**Programmatic usage:**
```python
from run import run_query
result = run_query(
    question="What was Apple's revenue growth in FY2024?",
    collection_name="financial_reports",
)
print(result["answer_final"])
print(result["route"])          # 'simple', 'slow', or 'fast'
print(result["citations"])      # [{source_doc, page_number, text}, ...]
```

### 5. Programmatic API

```python
from config import get_config
from models.qwen_llm import QwenRouter, QwenGeneratorFast, QwenGeneratorSlow
from retrieval.hybrid_retriever import HybridRetriever
from graph.workflow import FinancialRAGWorkflow
from memory.context_manager import TokenBudgetManager

# Initialize from config
cfg = get_config()

router = QwenRouter(config=cfg.router)
gen_fast = QwenGeneratorFast(config=cfg.generator_fast)
gen_slow = QwenGeneratorSlow(config=cfg.generator_slow)
retriever = HybridRetriever(config=cfg)
budget_manager = TokenBudgetManager()

workflow = FinancialRAGWorkflow(
    router=router,
    retriever=retriever,
    reranker=retriever.reranker,
    budget_manager=budget_manager,
)

# Run a query
result = workflow.run("What is Apple's ROE trend over 3 years?")
print(result["answer_final"])
print(result["route"])          # which path was taken
print(result["citations"])       # cited sources
```

---

## 📐 Retrieval Module Details

### Hybrid Retriever (`retrieval/hybrid_retriever.py`)

```python
# Core API
retriever = HybridRetriever(
    dense_model="BAAI/bge-m3",       # BGE-M3 embedding
    sparse_model="bm25",             # BM25 sparse retriever
    reranker_model="BAAI/bge-reranker-v2-m3",  # Cross-encoder reranker
    top_k=20,                         # Retrieve top-20 before reranking
    final_k=5,                        # Return top-5 after reranking
)

results = retriever.retrieve(query, collection_name="financial_reports")
# Returns: List[RetrievedChunk] sorted by fused RRRF + reranker scores
```

### Supported Retrieval Modes

| Mode | Description |
|------|-------------|
| `dense_only` | BGE-M3 embeddings cosine similarity |
| `sparse_only` | BM25 scoring with financial analyzer |
| `hybrid` | RRRF fusion of dense + sparse |
| `rerank` | Hybrid → BGE-Reranker cross-encoder re-scoring |

---

## 📊 Evaluation

Run RAGAS-based faithfulness evaluation:

```bash
python -m evaluation.ragas_eval \
    --answers results/answers.jsonl \
    --ground_truths results/ground_truths.jsonl \
    --contexts results/contexts.jsonl \
    --output results/faithfulness_scores.json
```

---

## 🔬 Design Rationale

### Why LangGraph?
Traditional sequential RAG (retrieve → generate) fails on:
- Multi-hop queries that need iterative retrieval
- Dynamic tool-calling when evidence is insufficient
- Queries that need to fast-path (simple factual questions)

LangGraph's graph model makes the control flow **explicit and debuggable**, with checkpointing for resumable long-running queries.

### Why BGE-M3?
BGE-M3 supports 100+ languages and achieves state-of-the-art performance on multilingual benchmarks. Its `ColBERT-style` late-interaction mode is ideal for financial documents with mixed Chinese-English terminology.

### Why RRRF over simple score combination?
RRRF is rank-based (not score-based), making it robust to calibration differences between dense and sparse retrievers. It does not require learned weights.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Please open an Issue or Submit a Pull Request for:
- Bug reports & feature requests
- New retrieval strategies or embedding models
- Evaluation benchmarks on financial datasets

---

*Maintained by **Yu Dong (@YuDong21)** — AI Algorithm Engineer*
