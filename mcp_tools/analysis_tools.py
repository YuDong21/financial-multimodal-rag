"""
Analysis MCP Tools.

Category 3: Financial analysis tools (metric extraction, calculation,
unit normalization, cross-table field mapping).

Tools:
  - analysis_metric_extract  : Extract financial metrics from structured tables
  - analysis_calc            : Calculate financial ratios / formulas
  - analysis_unit_normalize  : Normalize units across table cells
  - analysis_cross_table_map : Map equivalent fields across different tables
                              (e.g., "Total Revenue" in table A vs "Net Revenue" in table B)
  - analysis_cagr            : Compute CAGR over a multi-year series
  - analysis_yoy_growth      : Compute YoY growth between two periods
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .base import MCPTool

# ---------------------------------------------------------------------------
# Metric Extraction
# ---------------------------------------------------------------------------

class AnalysisMetricExtractTool(MCPTool):
    """
    Extract specific financial metrics from structured Markdown tables.

    Given one or more tables and a metric name, extracts the numeric values
    with their row/column context (year, company name, etc.).
    """

    name = "analysis_metric_extract"
    description = (
        "Extract specific financial metrics (e.g., Revenue, ROE, Debt-to-Equity) "
        "from structured Markdown tables. Returns extracted values with "
        "row context (year, company, currency) and metadata. "
        "Use when you need to pull specific numbers from table data."
    )

    METRIC_ALIASES: dict[str, list[str]] = {
        "revenue": ["Revenue", "Total Revenue", "Net Revenue", "Operating Revenue"],
        "net_income": ["Net Income", "Net Profit", "Profit Attributable"],
        "gross_profit": ["Gross Profit", "Gross Margin"],
        "operating_income": ["Operating Income", "EBIT", "Operating Profit"],
        "total_assets": ["Total Assets", "Assets"],
        "total_liabilities": ["Total Liabilities", "Total Debt", "Liabilities"],
        "shareholders_equity": ["Shareholders' Equity", "Equity", "Total Equity"],
        "current_assets": ["Current Assets"],
        "current_liabilities": ["Current Liabilities"],
        "inventory": ["Inventories", "Inventory"],
        "ebitda": ["EBITDA"],
        "eps": ["EPS", "Earnings Per Share", "Diluted EPS"],
        "roe": ["ROE", "Return on Equity"],
        "roa": ["ROA", "Return on Assets"],
    }

    parameters = {
        "type": "object",
        "properties": {
            "tables": {
                "type": "array",
                "description": "List of Markdown table strings.",
                "items": {"type": "string"},
            },
            "metric": {
                "type": "string",
                "description": "Metric to extract (e.g., 'revenue', 'ROE', 'debt_to_equity').",
            },
            "context_fields": {
                "type": "array",
                "description": "Fields to extract alongside the metric (e.g., ['year', 'company']).",
                "items": {"type": "string"},
                "default": ["year"],
            },
        },
        "required": ["tables", "metric"],
    }

    def execute(
        self,
        tables: list[str],
        metric: str,
        context_fields: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Extract metric values from tables."""
        context_fields = context_fields or ["year"]
        metric_lower = metric.lower()
        aliases = self.METRIC_ALIASES.get(metric_lower, [metric])

        extracted: list[dict[str, Any]] = []

        for table_text in tables:
            try:
                headers, rows = self._parse_markdown_table(table_text)
                if not headers or not rows:
                    continue

                # Find the metric column
                metric_col = None
                for col_idx, header in enumerate(headers):
                    header_clean = header.strip().lower()
                    if any(alias.lower() in header_clean for alias in aliases):
                        metric_col = col_idx
                        break

                if metric_col is None:
                    continue

                # Find context columns
                context_cols: dict[str, int] = {}
                for ctx in context_fields:
                    ctx_lower = ctx.lower()
                    for col_idx, header in enumerate(headers):
                        if ctx_lower in header.lower():
                            context_cols[ctx] = col_idx

                # Extract rows
                for row in rows:
                    if metric_col >= len(row):
                        continue
                    val_str = row[metric_col].strip()
                    # Extract numeric value
                    num_match = re.search(r"[\-]?[\d,]+\.?\d*", val_str.replace(",", ""))
                    if not num_match:
                        continue
                    value = float(num_match.group().replace(",", ""))

                    entry: dict[str, Any] = {"metric": metric, "value": value}
                    for ctx_name, col_idx in context_cols.items():
                        if col_idx < len(row):
                            entry[ctx_name] = row[col_idx].strip()
                    extracted.append(entry)

            except Exception:  # noqa: BLE001
                continue

        return {
            "metric": metric,
            "num_values_extracted": len(extracted),
            "values": extracted,
        }

    @staticmethod
    def _parse_markdown_table(table_text: str) -> tuple[list[str], list[list[str]]]:
        """Parse a Markdown table into headers and rows."""
        lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
        headers: list[str] = []
        rows: list[list[str]] = []

        MD_SEP_RE = re.compile(r"^\|[\s\-:|]+\|$")
        MD_ROW_RE = re.compile(r"^\|(.+)\|$")

        for line in lines:
            if MD_SEP_RE.match(line):
                continue
            match = MD_ROW_RE.match(line)
            if not match:
                continue
            cells = [c.strip() for c in match.group(1).split("|")]
            cells = [c for c in cells if c != ""]
            if not headers:
                headers = cells
            else:
                rows.append(cells)
        return headers, rows


# ---------------------------------------------------------------------------
# Financial Calculation
# ---------------------------------------------------------------------------

class AnalysisCalcTool(MCPTool):
    """
    Calculate financial ratios and formulas on extracted metric values.

    Supports: profit margins, ROE, ROA, current ratio, debt-to-equity,
    interest coverage, YoY growth, CAGR, P/E, EV/EBITDA.
    """

    name = "analysis_calc"
    description = (
        "Calculate financial ratios and formulas from extracted metric values. "
        "Supports: Gross/Operating/Net Profit Margin, ROE, ROA, ROIC, "
        "Current/Quick Ratio, Debt-to-Equity, Interest Coverage, "
        "YoY Growth, CAGR, P/E, EV/EBITDA. "
        "Use when the question asks for a computed metric or ratio."
    )

    parameters = {
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "description": "Metric to calculate (e.g., 'roe', 'cagr', 'yoy_growth').",
            },
            "values": {
                "type": "object",
                "description": "Named input values for the formula.",
                "additionalProperties": {"type": "number"},
            },
        },
        "required": ["metric", "values"],
    }

    CALC_MAP: dict[str, str] = {
        "gross_profit_margin": "(gross_profit / revenue) * 100",
        "operating_profit_margin": "(operating_income / revenue) * 100",
        "net_profit_margin": "(net_income / revenue) * 100",
        "roe": "(net_income / shareholders_equity) * 100",
        "roa": "(net_income / total_assets) * 100",
        "current_ratio": "current_assets / current_liabilities",
        "quick_ratio": "(current_assets - inventory) / current_liabilities",
        "debt_to_equity": "total_liabilities / shareholders_equity",
        "debt_to_assets": "total_liabilities / total_assets",
        "interest_coverage": "operating_income / interest_expense",
        "yoy_growth": "((value_current - value_prior) / value_prior) * 100",
        "cagr": "((end_value / start_value) ** (1 / years) - 1) * 100",
        "pe_ratio": "market_cap / net_income",
        "ev_ebitda": "enterprise_value / ebitda",
    }

    def execute(self, metric: str, values: dict[str, float]) -> dict[str, Any]:
        """Calculate a financial metric from input values."""
        metric_lower = metric.lower()
        formula = self.CALC_MAP.get(metric_lower)

        if formula is None:
            return {"error": f"Unknown metric: {metric}", "metric": metric}

        try:
            # Safety: only allow known variable names
            allowed_vars = {
                "revenue", "gross_profit", "operating_income", "net_income",
                "total_assets", "current_assets", "inventory",
                "total_liabilities", "current_liabilities", "shareholders_equity",
                "interest_expense", "ebitda", "market_cap", "enterprise_value",
                "value_current", "value_prior", "start_value", "end_value", "years",
            }
            # Check all values keys are allowed
            unknown = set(values.keys()) - allowed_vars
            if unknown:
                return {"error": f"Unknown variables: {unknown}", "metric": metric}

            result = eval(formula, {"__builtins__": {}}, values)  # noqa: S307
            result = round(float(result), 4)

            return {
                "metric": metric,
                "formula": formula,
                "inputs": values,
                "result": result,
                "unit": "%" if "margin" in metric_lower or "growth" in metric_lower or "coverage" in metric_lower else "",
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "metric": metric, "inputs": values}


# ---------------------------------------------------------------------------
# Unit Normalization
# ---------------------------------------------------------------------------

class AnalysisUnitNormalizeTool(MCPTool):
    """
    Normalize financial numbers with different unit suffixes.

    Handles: B (billions), M (millions), K (thousands), %,
    Chinese units (万, 亿), currency symbols.
    """

    name = "analysis_unit_normalize"
    description = (
        "Normalize financial numbers with different unit suffixes "
        "(B, M, K, %, 万, 亿, etc.) to a common scale. "
        "Use when comparing numbers from different tables that use "
        "different unit notations."
    )

    UNIT_MAP: dict[str, float] = {
        "b": 1e9, "bn": 1e9, "billion": 1e9,
        "m": 1e6, "mm": 1e6, "million": 1e6,
        "k": 1e3, "thousand": 1e3,
        "w": 1e4, "wan": 1e4,
        "y": 1e8, "yi": 1e8,
        "%": 0.01,
    }

    parameters = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "description": "List of strings with unit suffixes to normalize.",
                "items": {"type": "string"},
            },
            "target_unit": {
                "type": "string",
                "default": "absolute",
                "description": "Target unit: 'absolute' (full number), 'B', 'M', 'K', '%'.",
            },
        },
        "required": ["values"],
    }

    def execute(self, values: list[str], target_unit: str = "absolute") -> dict[str, Any]:
        """Normalize a list of unit-suffixed strings."""
        results: list[Optional[float]] = []

        for v in values:
            try:
                v_clean = v.strip().replace(",", "").replace(" ", "")
                m = re.match(r"^([\d.]+)\s*([a-zA-Z%\u4e00-\u9fa5]*)$", v_clean)
                if not m:
                    results.append(None)
                    continue

                num = float(m.group(1))
                unit = m.group(2).lower()

                if target_unit == "absolute":
                    factor = self.UNIT_MAP.get(unit, 1.0)
                    results.append(round(num * factor, 4))
                else:
                    # Convert to target unit
                    src_factor = self.UNIT_MAP.get(unit, 1.0)
                    tgt_factor = self.UNIT_MAP.get(target_unit.lower(), 1.0)
                    results.append(round(num * src_factor / tgt_factor, 4))
            except Exception:  # noqa: BLE001
                results.append(None)

        return {
            "original_values": values,
            "normalized_values": results,
            "target_unit": target_unit,
        }


# ---------------------------------------------------------------------------
# Cross-Table Field Mapping
# ---------------------------------------------------------------------------

class AnalysisCrossTableMapTool(MCPTool):
    """
    Map equivalent field names across different financial tables.

    For example: "Total Revenue" (Table A) → "Net Revenue" (Table B),
    or "FY2023" across different companies. Returns a mapping dict
    that can be used to align data for comparison or computation.
    """

    name = "analysis_cross_table_map"
    description = (
        "Map equivalent fields across different financial tables "
        "(e.g., 'Total Revenue' → 'Net Revenue', '2022' → 'FY2022'). "
        "Use when you need to align data from multiple tables for "
        "cross-company comparison or multi-period analysis."
    )

    STANDARD_NAMES: dict[str, list[str]] = {
        "revenue": ["Revenue", "Total Revenue", "Net Revenue", "Operating Revenue", "Sales"],
        "net_income": ["Net Income", "Net Profit", "Profit", "Net Earnings"],
        "gross_profit": ["Gross Profit", "Gross Margin"],
        "total_assets": ["Total Assets", "Assets"],
        "total_liabilities": ["Total Liabilities", "Total Debt", "Liabilities"],
        "shareholders_equity": ["Shareholders' Equity", "Equity", "Total Equity"],
        "ebitda": ["EBITDA", "EBIT"],
    }

    parameters = {
        "type": "object",
        "properties": {
            "table_headers": {
                "type": "array",
                "description": "List of table header lists (one per table).",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "table_names": {
                "type": "array",
                "description": "Optional names for each table.",
                "items": {"type": "string"},
            },
        },
        "required": ["table_headers"],
    }

    def execute(
        self,
        table_headers: list[list[str]],
        table_names: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Build a cross-table field mapping."""
        if table_names is None:
            table_names = [f"Table_{i}" for i in range(len(table_headers))]

        mappings: dict[str, list[dict[str, Any]]] = {}

        for std_name, aliases in self.STANDARD_NAMES.items():
            alias_map: list[dict[str, Any]] = []
            for t_idx, headers in enumerate(table_headers):
                for h_idx, header in enumerate(headers):
                    if any(alias.lower() in header.lower() for alias in aliases):
                        alias_map.append(
                            {
                                "table": table_names[t_idx],
                                "header": header,
                                "column_index": h_idx,
                            }
                        )
                        break
            if alias_map:
                mappings[std_name] = alias_map

        return {
            "num_tables": len(table_headers),
            "table_names": table_names,
            "mappings": mappings,
        }


# ---------------------------------------------------------------------------
# CAGR
# ---------------------------------------------------------------------------

class AnalysisCAGRTool(MCPTool):
    """
    Calculate Compound Annual Growth Rate (CAGR) over a multi-year series.
    """

    name = "analysis_cagr"
    description = (
        "Calculate Compound Annual Growth Rate (CAGR) from a start value, "
        "end value, and number of years. "
        "CAGR = (EndValue / StartValue)^(1/n) - 1. "
        "Use when analyzing growth rates over multiple periods."
    )

    parameters = {
        "type": "object",
        "properties": {
            "start_value": {"type": "number"},
            "end_value": {"type": "number"},
            "years": {"type": "number"},
        },
        "required": ["start_value", "end_value", "years"],
    }

    def execute(self, start_value: float, end_value: float, years: float) -> dict[str, Any]:
        """Compute CAGR."""
        try:
            if start_value <= 0 or years <= 0:
                return {"error": "start_value and years must be positive."}
            cagr = ((end_value / start_value) ** (1.0 / years) - 1) * 100
            return {
                "start_value": start_value,
                "end_value": end_value,
                "years": years,
                "cagr_percent": round(cagr, 4),
                "formula": f"({end_value}/{start_value})^(1/{years}) - 1",
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# YoY Growth
# ---------------------------------------------------------------------------

class AnalysisYoYGrowthTool(MCPTool):
    """
    Calculate Year-over-Year growth rate between two periods.
    """

    name = "analysis_yoy_growth"
    description = (
        "Calculate Year-over-Year (YoY) growth rate between a current period value "
        "and a prior period value. "
        "Formula: (Current - Prior) / Prior * 100%. "
        "Use when comparing performance between consecutive years."
    )

    parameters = {
        "type": "object",
        "properties": {
            "current_value": {"type": "number"},
            "prior_value": {"type": "number"},
            "period_label": {"type": "string", "description": "e.g. 'FY2024 vs FY2023'."},
        },
        "required": ["current_value", "prior_value"],
    }

    def execute(
        self,
        current_value: float,
        prior_value: float,
        period_label: Optional[str] = None,
    ) -> dict[str, Any]:
        """Compute YoY growth."""
        try:
            if prior_value == 0:
                return {"error": "prior_value cannot be zero."}
            yoy = (current_value - prior_value) / prior_value * 100
            return {
                "current_value": current_value,
                "prior_value": prior_value,
                "period": period_label or "",
                "yoy_growth_percent": round(yoy, 4),
                "absolute_change": round(current_value - prior_value, 4),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
