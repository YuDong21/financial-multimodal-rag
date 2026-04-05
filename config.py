"""
config.py — Centralized Parameter Configuration for financial-multimodal-rag.

This module is the single source of truth for all model and pipeline parameters.
No hardcoded values should exist in individual modules — all parameters
are loaded from this config.

Parameters are organized into typed dataclasses by role:

    config.router          — Qwen3-0.6B routing model
    config.generator_fast — Qwen3-8B fast-path generator (simple / direct)
    config.generator_slow — Qwen3-8B slow-path generator (complex reasoning)
    config.embedding      — BGE-M3 embedding model
    config.retrieval     — Dual-branch recall (dense + BM25)
    config.reranker      — BGE-Reranker v2-m3
    config.summarizer    — Qwen2.5-1.5B semantic truncation
    config.token_budget  — Context window budget management

Usage
-----
    >>> from config import get_config
    >>> cfg = get_config()
    >>> print(cfg.generator_fast.temperature)
    0.1
    >>> print(cfg.generator_fast.max_new_tokens)
    512
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Router — Qwen3-0.6B
# ---------------------------------------------------------------------------

@dataclass
class RouterConfig:
    """
    Configuration for the Qwen3-0.6B semantic router.

    The router is a LOW-LATENCY classification task.
    It ONLY classifies questions — it does NOT generate answers or explanations.
    Therefore: zero temperature, very short max_tokens, JSON output constraint.

    Model: Qwen3-0.6B
    Context window: 32,768 tokens
    """

    model: str = "qwen3-0.6b"

    # Generation parameters
    enable_thinking: bool = False      # Disabled — pure classification
    temperature: float = 0.0           # Zero temperature for stable classification
    top_p: float = 0.8
    top_k: int = 20
    max_new_tokens: int = 32           # Short — just enough for JSON output
    repetition_penalty: float = 1.05
    presence_penalty: float = 0.0

    # Output constraint
    output_format: str = "json"        # Force JSON: {"route": "fast|slow", "confidence": 0.00}

    # Context
    max_input_tokens: int = 8192       # Budget for conversation context
    conversation_history_tokens: int = 2048  # Reserved for history in router prompt


# ---------------------------------------------------------------------------
# Generator — Fast Path — Qwen3-8B
# ---------------------------------------------------------------------------

@dataclass
class GeneratorFastConfig:
    """
    Configuration for the Qwen3-8B fast-path answer generator.

    Fast path handles: single-hop, factual, low-ambiguity questions.
    Strategy: tight generation parameters for stability and speed.

    Model: Qwen3-8B (8.2B parameters)
    Context window: 32,768 tokens (131,072 with YaRN)
    """

    model: str = "qwen3-8b"

    # Generation parameters
    enable_thinking: bool = False      # Disabled — fast, direct answer
    temperature: float = 0.1           # Very low — factual, non-creative
    top_p: float = 0.8
    top_k: int = 20
    max_new_tokens: int = 512         # Short — factual answers are concise
    repetition_penalty: float = 1.05
    presence_penalty: float = 0.0

    # Context
    max_input_tokens: int = 6000       # Input context budget for fast path
    final_chunks: int = 4             # Number of reranked chunks injected into prompt

    # Slow path: n/a for fast path
    # (fast path bypasses retrieval, so no retrieval-specific settings)


# ---------------------------------------------------------------------------
# Generator — Slow Path — Qwen3-8B
# ---------------------------------------------------------------------------

@dataclass
class GeneratorSlowConfig:
    """
    Configuration for the Qwen3-8B slow-path answer generator.

    Slow path handles: cross-table, cross-period, multi-evidence joint analysis.
    Strategy: enable step-by-step reasoning with controlled creativity.

    Model: Qwen3-8B (8.2B parameters)
    Context window: 32,768 tokens (131,072 with YaRN)
    """

    model: str = "qwen3-8b"

    # Generation parameters
    enable_thinking: bool = True       # Enabled — step-by-step reasoning needed
    temperature: float = 0.3           # Moderate — some creativity for analysis
    top_p: float = 0.9
    top_k: int = 20
    max_new_tokens: int = 1024        # Longer — complex analysis needs space
    repetition_penalty: float = 1.05
    presence_penalty: float = 0.0

    # Context
    max_input_tokens: int = 12000     # Larger budget for complex retrieval
    final_chunks: int = 6             # More evidence for cross-table analysis


# ---------------------------------------------------------------------------
# Embedding — BGE-M3
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    """
    Configuration for the BGE-M3 embedding model.

    BGE-M3 simultaneously:
    - Provides dense vector retrieval
    - Supports multi-lingual (Chinese + English) financial text
    - Handles long documents (up to 8192 tokens per document)

    Model: BAAI/bge-m3
    Dimensions: 1024
    """

    model: str = "BAAI/bge-m3"

    # Batch processing
    embedding_batch_size: int = 32

    # Normalization
    normalize_embeddings: bool = True  # Cosine similarity for dense retrieval

    # Chunking (for text preprocessing before indexing)
    chunk_max_tokens: int = 512        # Per-chunk token limit
    chunk_overlap: int = 64            # Overlap between adjacent chunks

    # Encoding
    max_length: int = 1024             # Model max sequence length


# ---------------------------------------------------------------------------
# Retrieval — Dual-branch Recall
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    """
    Configuration for the hybrid dual-branch retrieval pipeline.

    Strategy: Dense (BGE-M3) + Sparse (BM25) → RRRF Fusion → Rerank (BGE-Reranker)

    RRRF is rank-based (not score-based) — robust to calibration differences
    between dense and sparse retrievers.
    """

    # Dense branch (BGE-M3)
    top_k_dense: int = 15             # Retrieve top-15 from dense branch

    # Sparse branch (BM25)
    top_k_bm25: int = 15             # Retrieve top-15 from BM25 branch

    # RRRF Fusion
    rrf_k: float = 60.0              # RRF damping parameter (higher = more weight to low ranks)

    # Mode options
    default_mode: str = "hybrid"      # "dense_only" | "sparse_only" | "hybrid" | "rerank"


# ---------------------------------------------------------------------------
# Reranker — BGE-Reranker v2-m3
# ---------------------------------------------------------------------------

@dataclass
class RerankerConfig:
    """
    Configuration for the BGE-Reranker v2-m3 cross-encoder.

    The reranker takes the 20 fused candidates and re-scores them
    with a cross-encoder for precision. Different output limits apply
    to fast vs. slow paths.
    """

    model: str = "BAAI/bge-reranker-v2-m3"

    # Input / output
    rerank_top_k_in: int = 20        # How many fused candidates to rerank
    rerank_top_k_out_fast: int = 6   # Final chunks for fast path
    rerank_top_k_out_slow: int = 8   # Final chunks for slow path (more evidence needed)

    # Cross-encoder
    max_length: int = 1024           # Max sequence length per (query, doc) pair
    score_threshold: float = 0.15      # Minimum reranker score to keep a result

    # Device
    device: Optional[str] = None      # Auto-detect (cuda/cpu) if None


# ---------------------------------------------------------------------------
# Semantic Summarizer — Qwen2.5-1.5B
# ---------------------------------------------------------------------------

@dataclass
class SummarizerConfig:
    """
    Configuration for the Qwen2.5-1.5B semantic truncation model.

    Used to compress older conversation turns when the token budget is exceeded.
    Small model (1.5B) — fast, cheap, sufficient for summarization.

    Model: Qwen2.5-1.5B (or any Qwen2.5-instruct variant)
    Context window: 32,768 tokens
    """

    model: str = "qwen2.5-1.5b-instruct"

    # Generation parameters
    enable_thinking: bool = False
    temperature: float = 0.3
    max_new_tokens: int = 512         # Summary is concise
    top_p: float = 0.9
    top_k: int = 20

    # Compression target
    compression_ratio: float = 0.15  # Target: ~15% of original length


# ---------------------------------------------------------------------------
# Token Budget
# ---------------------------------------------------------------------------

@dataclass
class TokenBudgetConfig:
    """
    Configuration for the conversation token budget manager.
    """

    budget: int = 8192                # Hard limit: truncate when reached
    k_keep: int = 4                  # Keep last 4 full turns verbatim
    compress_when_ratio: float = 0.85  # Trigger compression early at 85% of budget
    summarizer_model: str = "qwen2.5-1.5b-instruct"  # Model for compression


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@dataclass
class MCPToolsConfig:
    """Configuration for the MCP tool registry."""

    # Communication
    transport: str = "stdio"          # stdio for same-machine deployment
    server_command: list = field(default_factory=lambda: ["python", "-m", "mcp_tools.mcp_server"])

    # Timeout
    tool_timeout: int = 120           # Seconds per tool call


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Complete configuration for the financial-multimodal-rag system.

    Single source of truth — all modules import from here.
    """

    router: RouterConfig = field(default_factory=RouterConfig)
    generator_fast: GeneratorFastConfig = field(default_factory=GeneratorFastConfig)
    generator_slow: GeneratorSlowConfig = field(default_factory=GeneratorSlowConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    token_budget: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    mcp_tools: MCPToolsConfig = field(default_factory=MCPToolsConfig)


# ---------------------------------------------------------------------------
# Global singleton accessor
# ---------------------------------------------------------------------------

_CONFIG: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global Config singleton.

    Creates the config with default values on first call.
    Replace with a YAML/ENV-based loader in production:

        def get_config() -> Config:
            return ConfigLoader.from_yaml("config.yaml")

    Usage
    -----
        >>> from config import get_config
        >>> cfg = get_config()
        >>> cfg.generator_fast.temperature
        0.1
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG


def override_config(**kwargs) -> Config:
    """
    Override specific config fields.

    Usage
    -----
        >>> cfg = override_config(
        ...     generator_fast={"temperature": 0.2},
        ...     embedding={"embedding_batch_size": 64},
        ... )
    """
    cfg = get_config()
    for section, overrides in kwargs.items():
        section_obj = getattr(cfg, section, None)
        if section_obj is None:
            raise ValueError(f"Unknown config section: {section}")
        for key, value in overrides.items():
            if not hasattr(section_obj, key):
                raise ValueError(f"Unknown field {section}.{key}")
            setattr(section_obj, key, value)
    return cfg
