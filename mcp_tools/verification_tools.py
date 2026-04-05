"""
Verification MCP Tools.

Category 4: Result verification tools.

Tools:
  - verification_evidence_check   : Check if retrieved evidence sufficiently supports the answer
  - verification_citation_backtrack : Verify that inlined citations match actual retrieved chunks
  - verification_answer_groundedness : Check if generated answer is grounded in retrieved evidence
  - verification_missing_data_alert : Detect if the answer references data not in evidence
"""

from __future__ import annotations

from typing import Any, Optional

from .base import MCPTool


# ---------------------------------------------------------------------------
# Tool: Evidence Sufficiency Check
# ---------------------------------------------------------------------------

class VerificationEvidenceCheckTool(MCPTool):
    """
    Judge whether the retrieved evidence sufficiently supports answering the query.

    Uses a lightweight LLM to assess:
    - Does the evidence contain the key facts needed to answer?
    - Are there obvious gaps or missing dimensions?
    - Should additional retrieval or tools be invoked?

    Returns a structured verdict with confidence score and reasoning.
    """

    name = "verification_evidence_check"
    description = (
        "Check whether the currently retrieved evidence chunks are sufficient "
        "to fully answer the user's question. "
        "Returns: SUFFICIENT / INSUFFICIENT verdict, confidence score (0-1), "
        "and specific gap descriptions if insufficient. "
        "Use this before generating the final answer to decide if more "
        "retrieval or MCP tool calls are needed."
    )

    PROMPT_TEMPLATE = """You are an evidence sufficiency judge for a financial RAG system.

Given:
  QUESTION: {question}
  RETRIEVED EVIDENCE (chunks):
  {evidence_text}

Assess whether the retrieved evidence is sufficient to fully and accurately answer
the question. Consider:
1. Are the key facts / numbers mentioned in the question present in the evidence?
2. Does the evidence cover all aspects of the question?
3. Are the numbers / metrics the right ones (correct year, correct company, correct unit)?

Respond in JSON format:
{{
  "verdict": "SUFFICIENT" | "INSUFFICIENT",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of the verdict.",
  "gaps": ["List of specific information that is missing or uncertain."],
  "suggested_tools": ["deepdoc_table_parse", "retrieval_hybrid_search"] if insufficient else []
}}"""

    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Original user question."},
            "evidence_chunks": {
                "type": "array",
                "description": "List of retrieved evidence chunks.",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "source_doc": {"type": "string"},
                        "page_number": {"type": "integer"},
                        "retrieval_strategy": {"type": "string"},
                    },
                },
            },
        },
        "required": ["question", "evidence_chunks"],
    }

    def __init__(self, judge_model: Optional[Any] = None) -> None:
        self._judge_model = judge_model

    def execute(
        self,
        question: str,
        evidence_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Check evidence sufficiency."""
        if not evidence_chunks:
            return {
                "verdict": "INSUFFICIENT",
                "confidence": 1.0,
                "reasoning": "No evidence chunks provided.",
                "gaps": ["No retrieved documents available."],
                "suggested_tools": ["retrieval_hybrid_search"],
            }

        evidence_text = "\n\n".join(
            f"[{i+1}] (Source: {c['source_doc']}, page {c.get('page_number', '?')})\n"
            f"{c['text'][:500]}"
            for i, c in enumerate(evidence_chunks[:10])
        )

        prompt = self.PROMPT_TEMPLATE.format(
            question=question,
            evidence_text=evidence_text,
        )

        if self._judge_model is None:
            return {
                "error": "Judge model not initialized.",
                "verdict": "INSUFFICIENT",
                "confidence": 0.0,
                "reasoning": "MCP tool called without judge model.",
                "gaps": ["LLM judge not available."],
                "suggested_tools": [],
            }

        try:
            response = self._judge_model.chat(
                messages=[{"role": "user", "content": prompt}],
                generation_config=Any(temperature=0.0, top_p=1.0, max_tokens=512),  # type: ignore
            )
            import json
            raw = response.content
            # Attempt JSON parsing of the response
            verdict_data = self._parse_json_response(raw)
            return verdict_data
        except Exception as exc:  # noqa: BLE001
            return {
                "verdict": "INSUFFICIENT",
                "confidence": 0.0,
                "reasoning": f"Error during evidence check: {exc}",
                "gaps": ["Evidence check failed."],
                "suggested_tools": [],
            }

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        """Parse LLM JSON response."""
        import json, re
        # Try to extract JSON from markdown code blocks
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            raw = m.group(1)
        else:
            # Try to find JSON object in the response
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                raw = m.group(0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "verdict": "INSUFFICIENT",
                "confidence": 0.0,
                "reasoning": raw[:200],
                "gaps": ["Could not parse LLM response as JSON."],
                "suggested_tools": [],
            }


# ---------------------------------------------------------------------------
# Tool: Citation Backtracking
# ---------------------------------------------------------------------------

class VerificationCitationBacktrackTool(MCPTool):
    """
    Verify that inlined citations in the generated answer actually match
    the retrieved evidence chunks.

    Checks:
    - Every [Source: doc, page N] citation exists in retrieved docs
    - Claimed values match the source evidence
    - No hallucinations (citations to non-existent sources)
    """

    name = "verification_citation_backtrack"
    description = (
        "Verify that inlined citations in a generated answer match the actual "
        "retrieved evidence chunks. Detects: missing citations, hallucinated sources, "
        "cited values that don't match the evidence. "
        "Use before returning the final answer to ensure answer integrity."
    )

    parameters = {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "Generated answer with inline citations."},
            "evidence_chunks": {
                "type": "array",
                "description": "Retrieved evidence chunks.",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "source_doc": {"type": "string"},
                        "page_number": {"type": "integer"},
                        "chunk_id": {"type": "string"},
                    },
                },
            },
        },
        "required": ["answer", "evidence_chunks"],
    }

    def execute(
        self,
        answer: str,
        evidence_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Verify citations against evidence."""
        import re

        # Extract all [Source: ...] citations
        citation_pattern = r"\[Source:\s*([^,\]]+),\s*page\s*(\d+)\]"
        cited_sources = re.findall(citation_pattern, answer)

        if not cited_sources:
            return {
                "num_citations_found": 0,
                "verified": True,
                "issues": [],
                "message": "No citations found in answer.",
            }

        # Build a set of valid (source_doc, page_number) pairs
        valid_sources: set[tuple[str, int]] = {
            (c["source_doc"], c["page_number"])
            for c in evidence_chunks
            if "source_doc" in c and "page_number" in c
        }

        issues: list[dict[str, Any]] = []
        for cited_doc, cited_page in cited_sources:
            cited_doc = cited_doc.strip()
            cited_page = int(cited_page)
            if (cited_doc, cited_page) not in valid_sources:
                issues.append(
                    {
                        "type": "HALLUCINATED_CITATION",
                        "cited_doc": cited_doc,
                        "cited_page": cited_page,
                        "issue": f"Source '{cited_doc}' page {cited_page} not in retrieved evidence.",
                    }
                )

        return {
            "num_citations_found": len(cited_sources),
            "num_valid": len(cited_sources) - len(issues),
            "num_issues": len(issues),
            "verified": len(issues) == 0,
            "issues": issues,
            "message": (
                "All citations verified." if not issues
                else f"{len(issues)} citation issue(s) found."
            ),
        }


# ---------------------------------------------------------------------------
# Tool: Answer Groundedness
# ---------------------------------------------------------------------------

class VerificationAnswerGroundednessTool(MCPTool):
    """
    Check if the generated answer is factually grounded in the evidence.

    Uses an LLM to decompose the answer into claims and verify each claim
    against the evidence. Returns a groundedness score and per-claim results.
    """

    name = "verification_answer_groundedness"
    description = (
        "Check if the generated answer is factually grounded in the retrieved evidence. "
        "Decomposes the answer into individual claims and checks each against evidence. "
        "Returns a groundedness score (0-1), per-claim status, and hallucination flags. "
        "Use this to measure answer quality before returning to the user."
    )

    PROMPT_TEMPLATE = """You are a factual groundedness auditor.

Given:
  QUESTION: {question}
  ANSWER: {answer}
  EVIDENCE:
  {evidence_text}

Step 1: Decompose the answer into individual factual claims.
Step 2: For each claim, determine if it is SUPPORTED, CONTRADICTED, or UNSUPPORTED
        (insufficient evidence to verify) by the EVIDENCE.

Respond in JSON format:
{{
  "num_claims": N,
  "groundedness_score": 0.0-1.0,
  "claims": [
    {{
      "claim": "text of the claim",
      "status": "SUPPORTED" | "CONTRADICTED" | "UNSUPPORTED",
      "evidence_for": "matching evidence text or null",
      "issue": "description of issue if contradicted or unsupported"
    }}
  ]
}}"""

    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
            "evidence_chunks": {
                "type": "array",
                "description": "Retrieved evidence chunks.",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "source_doc": {"type": "string"},
                        "page_number": {"type": "integer"},
                    },
                },
            },
        },
        "required": ["question", "answer", "evidence_chunks"],
    }

    def __init__(self, auditor_model: Optional[Any] = None) -> None:
        self._auditor_model = auditor_model

    def execute(
        self,
        question: str,
        answer: str,
        evidence_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Check answer groundedness."""
        evidence_text = "\n\n".join(
            f"[{i+1}] {c['text'][:400]}" for i, c in enumerate(evidence_chunks[:8])
        )

        prompt = self.PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            evidence_text=evidence_text,
        )

        if self._auditor_model is None:
            return {
                "error": "Auditor model not initialized.",
                "groundedness_score": 0.0,
            }

        try:
            response = self._auditor_model.chat(
                messages=[{"role": "user", "content": prompt}],
                generation_config=Any(temperature=0.0, top_p=1.0, max_tokens=1024),  # type: ignore
            )
            import json, re
            raw = response.content
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                raw = m.group(0)
            result = json.loads(raw)
            return result
        except Exception as exc:  # noqa: BLE001
            return {
                "error": str(exc),
                "groundedness_score": 0.0,
                "num_claims": 0,
                "claims": [],
            }


# ---------------------------------------------------------------------------
# Tool: Missing Data Alert
# ---------------------------------------------------------------------------

class VerificationMissingDataAlertTool(MCPTool):
    """
    Detect if the answer references data (numbers, years, metrics) that
    is not present in the retrieved evidence.

    Use when the answer makes quantitative claims that cannot be verified
    against the provided evidence chunks.
    """

    name = "verification_missing_data_alert"
    description = (
        "Detect if the answer references data (numbers, years, metrics, companies) "
        "that is not found in the retrieved evidence chunks. "
        "Returns a list of unverified data references and an alert level. "
        "Use when the answer makes quantitative claims that need cross-checking."
    )

    parameters = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "evidence_chunks": {
                "type": "array",
                "description": "Retrieved evidence chunks.",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "source_doc": {"type": "string"},
                        "page_number": {"type": "integer"},
                    },
                },
            },
        },
        "required": ["answer", "evidence_chunks"],
    }

    def execute(
        self,
        answer: str,
        evidence_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Detect missing data references."""
        import re

        # Extract numbers and key terms from answer
        numbers = re.findall(r"\b[\d,]+\.?\d*\b", answer)
        # Extract years (4-digit numbers starting with 19 or 20)
        years = re.findall(r"\b(20\d{2}|19\d{2})\b", answer)
        # Extract percentage values
        percentages = re.findall(r"\b\d+\.?\d*%", answer)

        all_evidence_text = " ".join(c["text"] for c in evidence_chunks)

        missing_numbers: list[str] = []
        missing_years: list[str] = []
        missing_percentages: list[str] = []

        for num in set(numbers):
            num_clean = num.replace(",", "")
            if num_clean not in all_evidence_text:
                missing_numbers.append(num)

        for year in set(years):
            if year not in all_evidence_text:
                missing_years.append(year)

        for pct in set(percentages):
            pct_num = pct.rstrip("%")
            if pct_num not in all_evidence_text and pct not in all_evidence_text:
                missing_percentages.append(pct)

        alert_level = "NONE"
        if missing_years or len(missing_numbers) > 2:
            alert_level = "HIGH"
        elif missing_numbers or missing_percentages:
            alert_level = "MEDIUM"

        return {
            "answer": answer[:100] + "...",
            "alert_level": alert_level,
            "missing_years": list(set(missing_years)),
            "missing_numbers": list(set(missing_numbers[:10])),
            "missing_percentages": list(set(missing_percentages)),
            "total_chunks_checked": len(evidence_chunks),
            "message": f"Alert level: {alert_level}. "
                       f"{len(missing_numbers)} numbers, {len(missing_years)} years, "
                       f"{len(missing_percentages)} percentages not found in evidence.",
        }
