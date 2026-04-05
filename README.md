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
│   Semantic Routing Node     │  ──► "Simple" → Direct Generation
│   (Qwen3-0.6B)              │  ──► "Complex" → Deep Retrieval Pipeline
└─────────────────────────────┘
            │
            ▼ (complex path)
┌─────────────────────────────┐
│   Task Decomposition Node   │  Splits multi-hop queries into sub-tasks
└─────────────────────────────┘
            │
            ▼
┌─────────────────────────────┐
│   Hybrid Retriever          │  BGE-M3 dense  +  BM25 sparse  →  RRRF fusion
│   (BGE-M3 + BGE-Reranker)  │  Reranked Top-K chunks fed to generator
└─────────────────────────────┘
            │
            ▼
┌─────────────────────────────┐
│   Evidence Sufficiency      │  Check: retrieved docs satisfy the query?
│   Check Node                 │  → No: → MCP Tool call (DeepDoc / Calc)
└─────────────────────────────┘
            │
            ▼
┌─────────────────────────────┐
│   Answer Generation Node     │
│   (Qwen3-8B, citation-aware) │
└─────────────────────────────┘
            │
            ▼
       Grounded Answer
       (with inline citations)
```

---

## 📂 Project Structure

```
financial-multimodal-rag/
├── data_pipeline/                 # Data ingestion & document parsing
│   ├── __init__.py
│   └── deepdoc_interface.py      # RAGFlow DeepDoc layout API placeholder
├── retrieval/                     # Hybrid retrieval engine
│   ├── __init__.py
│   └── hybrid_retriever.py        # BGE-M3 + BM25 + BGE-Reranker
├── graph/                          # LangGraph workflow orchestration
│   ├── __init__.py
│   └── workflow.py                # State machine with conditional routing
├── models/                         # LLM interface wrappers
│   ├── __init__.py
│   └── qwen_llm.py                # Qwen3-8B / Qwen3-0.6B / Qwen2.5 API
├── memory/                         # Context window management
│   ├── __init__.py
│   └── context_manager.py         # Sliding window + semantic truncation
├── mcp_tools/                      # MCP protocol toolchain
│   ├── __init__.py
│   ├── deepdoc_tool.py            # DeepDoc layout analysis tool
│   └── financial_calc_tool.py     # Financial formula calculator
├── evaluation/                     # Evaluation framework
│   ├── __init__.py
│   └── ragas_eval.py              # RAGAS Faithfulness metric
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
# Set your Qwen/DashScope API key
export DASHSCOPE_API_KEY="your-dashscope-api-key"

# Optional: DeepDoc API (RAGFlow backend)
export DEEPDOC_API_KEY="your-deepdoc-api-key"
```

### 3. Run the RAG Pipeline

```python
from graph.workflow import FinancialRAGWorkflow
from models.qwen_llm import QwenRouter, QwenGenerator

router = QwenRouter()          # Qwen3-0.6B for routing
generator = QwenGenerator()    # Qwen3-8B for answer generation

workflow = FinancialRAGWorkflow(
    router=router,
    generator=generator,
    retriever=retriever,        # hybrid_retriever instance
)
answer = workflow.run("What was Apple's total revenue in FY2024?")
print(answer)
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
