"""
RAGAS-based Evaluation — Faithfulness Metric.

Implements the RAGAS Faithfulness score using the ragas library.
Faithfulness measures the fraction of claims in the generated answer
that can be attributed to the retrieved evidence chunks.

Score Range
-----------
  0.0 – 1.0  (higher = more faithful)

Interpretation
-------------
  ≥ 0.9  Excellent — almost all claims are supported by evidence
  0.7–0.9  Good — minor hallucinations or unsupported leaps
  0.5–0.7  Moderate — several unsupported claims
  < 0.5   Poor — significant hallucination or off-topic generation

Usage
-----
    >>> from evaluation import RAGASEvaluator
    >>> evaluator = RAGASEvaluator()
    >>> score = evaluator.faithfulness_score(
    ...     answer="Apple revenue was $391B in FY2024 [Source: apple_2024.pdf, page 3]",
    ...     contexts=[
    ...         "Apple reported total revenue of $391.0B in fiscal year 2024.",
    ...         "Revenue grew 7% year-over-year from $365B in FY2023.",
    ...     ],
    ... )
    >>> print(f"Faithfulness: {score:.3f}")
    Faithfulness: 0.950

Batch Evaluation
----------------
    >>> results = evaluator.evaluate_from_file(
    ...     answers_path="results/answers.jsonl",
    ...     contexts_path="results/contexts.jsonl",
    ...     output_path="results/faithfulness_scores.json",
    ... )
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

# ragas imports — handle optional installation gracefully
try:
    from ragas import EvaluationDataset
    from ragas.metrics import Faithfulness
    from ragas.run_config import RunConfig
    from ragas.evaluator import evaluate

    _RAGAS_AVAILABLE = True
except ImportError:  # noqa: E501
    _RAGAS_AVAILABLE = False
    evaluate = None
    Faithfulness = None
    EvaluationDataset = None
    RunConfig = None

# ---------------------------------------------------------------------------
# Result Models
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """
    Single evaluation result for one question–answer pair.

    Attributes
    ----------
    question : str
        The original user question.
    answer : str
        The generated answer.
    faithfulness : float
        Faithfulness score in [0.0, 1.0].
    citations : list[dict]
        Source citations from the answer.
    error : Optional[str]
        Error message if evaluation failed.
    """

    question: str
    answer: str
    faithfulness: float
    citations: list[dict[str, Any]]
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "faithfulness": round(self.faithfulness, 4),
            "citations": self.citations,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Core Faithfulness Implementation
# ---------------------------------------------------------------------------

def faithfulness_score(
    answer: str,
    contexts: list[str],
) -> float:
    """
    Compute the RAGAS Faithfulness score for a single answer.

    Faithfulness is computed by:
      1. Decomposing the answer into individual claims
      2. Checking each claim against the provided contexts
      3. Returning the fraction of claims that are supported

    Parameters
    ----------
    answer : str
        The generated answer text.
    contexts : list of str
        The retrieved evidence chunks used to generate the answer.

    Returns
    -------
    float in [0.0, 1.0]
    """
    if not _RAGAS_AVAILABLE:
        raise ImportError(
            "ragas is not installed. Run: pip install ragas>=0.1.0"
        )

    # Prepare as ragas dataset
    dataset_dict = {
        "user_input": ["placeholder"],  # ragas needs this column
        "reference": [""],             # ground truth (empty for faithfulness-only)
        "response": [answer],
        "contexts": [contexts],
    }
    dataset = EvaluationDataset.from_dict(dataset_dict)

    # Run evaluation with only the Faithfulness metric
    metrics = [Faithfulness()]
    result = evaluate(dataset, metrics=metrics, run_config=RunConfig(timeout=120))

    # Extract the faithfulness score
    scores = result.scores
    if hasattr(scores, "to_dict"):
        scores = scores.to_dict()
    return float(scores[0]["faithfulness"])


# ---------------------------------------------------------------------------
# Batch Evaluator
# ---------------------------------------------------------------------------

class RAGASEvaluator:
    """
    Batch evaluator for RAG pipelines using RAGAS metrics.

    Parameters
    ----------
    api_key : str, optional
        API key for LLM-based evaluation (used by ragas for claim decomposition).
        Falls back to DASHSCOPE_API_KEY environment variable.
    run_config : dict, optional
        Override for ragas RunConfig (timeout, max_retries, etc.).

    Usage
    -----
    >>> evaluator = RAGASEvaluator()
    >>> results = evaluator.evaluate_from_file(
    ...     answers_path="results/answers.jsonl",
    ...     contexts_path="results/contexts.jsonl",
    ...     output_path="results/scores.json",
    ... )
    >>> for r in results:
    ...     print(f"Q: {r.question[:50]} | Faithfulness: {r.faithfulness:.3f}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        run_config: Optional[dict[str, Any]] = None,
    ) -> None:
        if not _RAGAS_AVAILABLE:
            raise ImportError(
                "ragas is not installed. Run: pip install ragas>=0.1.0"
            )

        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.run_config = run_config or {}

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        citations: Optional[list[dict[str, Any]]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single question–answer pair.

        Parameters
        ----------
        question : str
        answer : str
        contexts : list of evidence strings
        citations : list of citation dicts, optional

        Returns
        -------
        EvaluationResult
        """
        try:
            faithfulness = faithfulness_score(answer, contexts)
            return EvaluationResult(
                question=question,
                answer=answer,
                faithfulness=faithfulness,
                citations=citations or [],
            )
        except Exception as exc:  # noqa: BLE001
            return EvaluationResult(
                question=question,
                answer=answer,
                faithfulness=0.0,
                citations=citations or [],
                error=str(exc),
            )

    def evaluate_batch(
        self,
        questions: list[str],
        answers: list[str],
        contexts_list: list[list[str]],
        citations_list: Optional[list[list[dict[str, Any]]]] = None,
        progress_callback: Optional[callable[[int, int], None]] = None,
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple question–answer pairs in batch.

        Parameters
        ----------
        questions : list of str
        answers : list of str
        contexts_list : list of evidence lists (one per question)
        citations_list : optional list of citation lists
        progress_callback : callable(current, total), optional

        Returns
        -------
        list of EvaluationResult
        """
        n = len(questions)
        citations_list = citations_list or [None] * n

        results: list[EvaluationResult] = []
        for i, (q, a, ctxs, cites) in enumerate(
            zip(questions, answers, contexts_list, citations_list)
        ):
            result = self.evaluate_single(q, a, ctxs, cites)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, n)

        return results

    def evaluate_from_file(
        self,
        answers_path: str,
        contexts_path: str,
        output_path: Optional[str] = None,
        questions_path: Optional[str] = None,
    ) -> list[EvaluationResult]:
        """
        Evaluate from newline-delimited JSON (JSONL) files.

        Expected format (one JSON object per line):
            {"answer": "...", "contexts": ["...", "..."], "question": "...", "citations": [...]}

        Parameters
        ----------
        answers_path : str
            Path to answers JSONL file.
        contexts_path : str
            Path to contexts JSONL file (list of context lists).
        questions_path : str, optional
            Path to questions JSONL file (if not embedded in answers_path).
        output_path : str, optional
            If provided, results are written here as JSONL.

        Returns
        -------
        list of EvaluationResult
        """
        answers_data = self._load_jsonl(answers_path)
        contexts_data = self._load_jsonl(contexts_path)
        questions_data = (
            self._load_jsonl(questions_path) if questions_path else [{}] * len(answers_data)
        )

        questions = [d.get("question", "") for d in questions_data]
        answers = [d.get("answer", "") for d in answers_data]
        contexts_list = [d.get("contexts", []) for d in contexts_data]
        citations_list = [d.get("citations", []) for d in answers_data]

        results = self.evaluate_batch(questions, answers, contexts_list, citations_list)

        if output_path:
            self._write_jsonl(results, output_path)

        return results

    # -------------------------------------------------------------------------
    # Report generation
    # -------------------------------------------------------------------------

    def generate_report(
        self, results: list[EvaluationResult]
    ) -> dict[str, Any]:
        """
        Generate a summary statistics report from evaluation results.

        Parameters
        ----------
        results : list of EvaluationResult

        Returns
        -------
        dict with keys: mean_faithfulness, median_faithfulness,
                        std_faithfulness, score_distribution, failures
        """
        scores = [r.faithfulness for r in results if r.error is None]
        failures = [r for r in results if r.error is not None]

        import statistics

        if not scores:
            return {
                "mean_faithfulness": 0.0,
                "median_faithfulness": 0.0,
                "std_faithfulness": 0.0,
                "score_distribution": {},
                "total_evaluated": 0,
                "failures": len(failures),
            }

        return {
            "mean_faithfulness": round(statistics.mean(scores), 4),
            "median_faithfulness": round(statistics.median(scores), 4),
            "std_faithfulness": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
            "score_distribution": {
                "≥ 0.9": sum(1 for s in scores if s >= 0.9),
                "0.7–0.9": sum(1 for s in scores if 0.7 <= s < 0.9),
                "0.5–0.7": sum(1 for s in scores if 0.5 <= s < 0.7),
                "< 0.5": sum(1 for s in scores if s < 0.5),
            },
            "total_evaluated": len(scores),
            "failures": len(failures),
        }

    # -------------------------------------------------------------------------
    # File I/O helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_jsonl(path: str) -> list[dict[str, Any]]:
        """Load a JSONL file into a list of dicts."""
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def _write_jsonl(results: list[EvaluationResult], path: str) -> None:
        """Write evaluation results to a JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
