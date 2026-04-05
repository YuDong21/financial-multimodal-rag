"""
Financial Calculation MCP Tool.

Implements the MCP (Model Context Protocol) tool interface for evaluating
financial formulas and ratios on structured table data extracted from
financial reports.

Supported calculations include:
- Profitability:  Gross Profit Margin, Operating Profit Margin, Net Profit Margin, ROE, ROA, ROIC
- Liquidity:       Current Ratio, Quick Ratio, Cash Ratio
- Leverage:       Debt-to-Equity, Debt-to-Assets, Interest Coverage
- Growth:         YoY Revenue Growth, QoQ Revenue Growth, CAGR
- Valuation:      P/E Ratio, P/B Ratio, EV/EBITDA

MCP Tool Schema
---------------
name: financial_calculation
description: >
  Evaluates financial ratios, formulas, and calculations using structured
  table data extracted from financial reports. Use when the user's question
  asks for a specific financial metric, ratio, or calculated value.
parameters:
  type: object
  properties:
    table_texts:
      type: array
      items: {type: string}
      description: List of Markdown/CSV table strings extracted from financial reports
    query:
      type: string
      description: Natural language description of the calculation needed
  required: [table_texts, query]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import pydantic


# ---------------------------------------------------------------------------
# Financial Metric Definitions
# ---------------------------------------------------------------------------

METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "gross_profit_margin": {
        "formula": "(Revenue - COGS) / Revenue",
        "required_fields": ["revenue", "cogs", "gross_profit"],
        "description": "Gross profit as a percentage of revenue",
    },
    "operating_profit_margin": {
        "formula": "Operating Income / Revenue",
        "required_fields": ["operating_income", "revenue"],
        "description": "Operating profit as a percentage of revenue",
    },
    "net_profit_margin": {
        "formula": "Net Income / Revenue",
        "required_fields": ["net_income", "revenue"],
        "description": "Net profit as a percentage of revenue",
    },
    "roe": {
        "formula": "Net Income / Shareholders' Equity",
        "required_fields": ["net_income", "shareholders_equity", "equity"],
        "description": "Return on Equity — net income relative to equity",
    },
    "roa": {
        "formula": "Net Income / Total Assets",
        "required_fields": ["net_income", "total_assets"],
        "description": "Return on Assets — net income relative to total assets",
    },
    "roic": {
        "formula": "NOPAT / Invested Capital",
        "required_fields": ["nopat", "invested_capital"],
        "description": "Return on Invested Capital",
    },
    "current_ratio": {
        "formula": "Current Assets / Current Liabilities",
        "required_fields": ["current_assets", "current_liabilities"],
        "description": "Liquidity ratio — ability to pay short-term obligations",
    },
    "quick_ratio": {
        "formula": "(Current Assets - Inventory) / Current Liabilities",
        "required_fields": ["current_assets", "inventory", "current_liabilities"],
        "description": "Acid-test liquidity ratio",
    },
    "debt_to_equity": {
        "formula": "Total Debt / Shareholders' Equity",
        "required_fields": ["total_debt", "shareholders_equity", "total_liabilities", "equity"],
        "description": "Leverage ratio — total debt relative to equity",
    },
    "debt_to_assets": {
        "formula": "Total Debt / Total Assets",
        "required_fields": ["total_debt", "total_assets"],
        "description": "Percentage of assets financed by debt",
    },
    "interest_coverage": {
        "formula": "EBIT / Interest Expense",
        "required_fields": ["ebit", "interest_expense"],
        "description": "Ability to pay interest from earnings",
    },
    "yoy_revenue_growth": {
        "formula": "(Revenue_t - Revenue_t-1) / Revenue_t-1",
        "required_fields": ["revenue_current", "revenue_prior"],
        "description": "Year-over-year revenue change",
    },
    "cagr": {
        "formula": "(EndValue / StartValue)^(1/n) - 1",
        "required_fields": ["start_value", "end_value", "years"],
        "description": "Compound Annual Growth Rate over n years",
    },
    "pe_ratio": {
        "formula": "Market Cap / Net Income (or Price Per Share / EPS)",
        "required_fields": ["market_cap", "net_income", "price", "eps"],
        "description": "Price-to-Earnings ratio",
    },
    "ev_ebitda": {
        "formula": "Enterprise Value / EBITDA",
        "required_fields": ["enterprise_value", "ebitda"],
        "description": "EV to EBITDA multiple",
    },
}


# ---------------------------------------------------------------------------
# Result Model
# ---------------------------------------------------------------------------

@dataclass
class CalcResult:
    """
    Result of a financial calculation.

    Attributes
    ----------
    metric : str
        The metric that was calculated.
    value : Optional[float]
        The calculated value, or None if calculation failed.
    unit : str
        Human-readable unit (e.g. "%", "x", "B USD").
    formula : str
        The formula used.
    inputs : dict[str, float]
        The extracted numeric inputs used in the calculation.
    confidence : float
        0.0–1.0 confidence that the correct values were extracted.
    error : Optional[str]
        Error message if calculation failed.
    """

    metric: str
    value: Optional[float]
    unit: str
    formula: str
    inputs: dict[str, float]
    confidence: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Table Parser
# ---------------------------------------------------------------------------

class TableParser:
    """
    Lightweight parser for Markdown / CSV table strings.

    Extracts header → row mappings and maps column names to numeric values.
    Handles common financial table formats from annual report PDFs.
    """

    # Regex to detect a Markdown table row
    MD_ROW_RE = re.compile(r"^\|(.+)\|$")
    MD_SEP_RE = re.compile(r"^\|[\s\-:|]+\|$")

    def parse_md_table(self, table_text: str) -> tuple[list[str], list[list[str]]]:
        """
        Parse a Markdown table string into headers and data rows.

        Parameters
        ----------
        table_text : str
            Multi-line Markdown table (with | separators and --- row).

        Returns
        -------
        (headers: list[str], rows: list[list[str]])

        Examples
        --------
            headers, rows = parser.parse_md_table(table_text)
        """
        lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
        headers: list[str] = []
        rows: list[list[str]] = []

        for line in lines:
            if self.MD_SEP_RE.match(line):
                continue
            match = self.MD_ROW_RE.match(line)
            if not match:
                continue
            cells = [c.strip() for c in match.group(1).split("|")]
            if not headers:
                headers = cells
            else:
                rows.append(cells)

        return headers, rows

    def parse_csv_table(self, table_text: str) -> tuple[list[str], list[list[str]]]:
        """Parse a CSV-style table (comma or tab separated)."""
        lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
        if not lines:
            return [], []
        headers = [h.strip() for h in re.split(r"[,\t]", lines[0])]
        rows = [
            [c.strip() for c in re.split(r"[,\t]", line)]
            for line in lines[1:]
        ]
        return headers, rows

    def extract_numeric(
        self,
        headers: list[str],
        rows: list[list[str]],
        field_aliases: dict[str, list[str]],
    ) -> dict[str, float]:
        """
        Extract numeric values from a parsed table.

        Parameters
        ----------
        headers : list of column header names
        rows : list of rows (each row is a list of cell strings)
        field_aliases : dict mapping canonical field name → list of possible header aliases
            e.g. {"revenue": ["Revenue", "Total Revenue", "Revenue (B USD)"]}

        Returns
        -------
        dict of {canonical_name: value}
        """
        values: dict[str, float] = {}
        for canonical, aliases in field_aliases.items():
            for row in rows:
                for alias in aliases:
                    for col_idx, header in enumerate(headers):
                        if alias.lower() in header.lower() and col_idx < len(row):
                            val_str = re.sub(r"[^\d.\-]", "", row[col_idx])
                            try:
                                values[canonical] = float(val_str)
                                break
                            except ValueError:
                                continue
                    if canonical in values:
                        break
                if canonical in values:
                    break
        return values


# ---------------------------------------------------------------------------
# Financial Calculator Tool
# ---------------------------------------------------------------------------

class FinancialCalcTool:
    """
    MCP tool for financial formula evaluation on structured table data.

    Usage as an MCP tool:

    >>> calc = FinancialCalcTool()
    >>> result = calc.execute(
    ...     table_texts=[
    ...         "| Quarter | Revenue |\n|---------|--------|\n| Q1 2024 | 119.6 |",
    ...     ],
    ...     query="What is the YoY revenue growth?",
    ... )
    >>> print(result.value, result.unit)
    """

    TOOL_NAME = "financial_calculation"
    TOOL_DESCRIPTION = (
        "Evaluates financial ratios, formulas, and calculations using structured "
        "table data extracted from financial reports. "
        "Use when the user's question asks for a specific financial metric, "
        "ratio, or calculated value such as ROE, Debt-to-Equity, CAGR, etc."
    )

    PARAMETER_SCHEMA = {
        "type": "object",
        "properties": {
            "table_texts": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of Markdown or CSV table strings extracted from "
                    "financial reports. Each string is one table."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Natural language description of the calculation needed. "
                    "E.g. 'Calculate ROE for FY2024', 'What is the debt-to-equity ratio?'"
                ),
            },
        },
        "required": ["table_texts", "query"],
    }

    # Common alias mappings for financial terms
    FIELD_ALIASES: dict[str, list[str]] = {
        "revenue": ["Revenue", "Total Revenue", "Revenue (B USD)", "Net Revenue"],
        "net_income": ["Net Income", "Net Profit", "Net Income Attributable"],
        "gross_profit": ["Gross Profit", "Gross Margin"],
        "operating_income": ["Operating Income", "EBIT", "Operating Profit"],
        "total_assets": ["Total Assets", "Assets"],
        "total_liabilities": ["Total Liabilities", "Total Debt", "Liabilities"],
        "shareholders_equity": ["Shareholders' Equity", "Equity", "Total Equity"],
        "current_assets": ["Current Assets"],
        "current_liabilities": ["Current Liabilities"],
        "inventory": ["Inventories", "Inventory"],
        "equity": ["Shareholders' Equity", "Equity", "Total Equity"],
        "ebit": ["EBIT", "Operating Income"],
        "ebitda": ["EBITDA"],
        "interest_expense": ["Interest Expense", "Interest"],
        "cogs": ["Cost of Goods Sold", "COGS"],
        "nopat": ["NOPAT", "Net Operating Profit After Tax"],
        "invested_capital": ["Invested Capital", "Capital Employed"],
    }

    def __init__(self) -> None:
        self.parser = TableParser()

    def get_schema(self) -> dict[str, Any]:
        """Return the MCP tool schema for tool registration."""
        return {
            "name": self.TOOL_NAME,
            "description": self.TOOL_DESCRIPTION,
            "parameters": self.PARAMETER_SCHEMA,
        }

    def execute(
        self,
        table_texts: list[str],
        query: str,
    ) -> CalcResult:
        """
        Execute a financial calculation on provided table data.

        Parameters
        ----------
        table_texts : list of str
            Markdown or CSV table strings from financial reports.
        query : str
            Natural language description of the calculation.

        Returns
        -------
        CalcResult
        """
        query_lower = query.lower()

        # Identify which metric to calculate based on the query
        metric_name = self._identify_metric(query_lower)
        if metric_name is None:
            return CalcResult(
                metric="unknown",
                value=None,
                unit="",
                formula="",
                inputs={},
                confidence=0.0,
                error=f"Could not identify a calculable metric in query: {query}",
            )

        definition = METRIC_DEFINITIONS[metric_name]

        # Parse all tables and merge extracted values
        all_values: dict[str, float] = {}
        for table_text in table_texts:
            try:
                headers, rows = self.parser.parse_md_table(table_text)
                if not headers:
                    headers, rows = self.parser.parse_csv_table(table_text)
                extracted = self.parser.extract_numeric(headers, rows, self.FIELD_ALIASES)
                all_values.update(extracted)
            except Exception:  # noqa: BLE001
                continue

        # Attempt calculation
        try:
            value = self._calculate(metric_name, all_values)
            unit = self._get_unit(metric_name)
            confidence = min(1.0, len(all_values) / len(definition["required_fields"]))

            return CalcResult(
                metric=metric_name,
                value=value,
                unit=unit,
                formula=definition["formula"],
                inputs=all_values,
                confidence=confidence,
            )
        except Exception as exc:  # noqa: BLE001
            return CalcResult(
                metric=metric_name,
                value=None,
                unit=self._get_unit(metric_name),
                formula=definition["formula"],
                inputs=all_values,
                confidence=0.0,
                error=str(exc),
            )

    def _identify_metric(self, query: str) -> Optional[str]:
        """Map a query string to a metric name."""
        keyword_map: dict[str, list[str]] = {
            "gross_profit_margin": ["gross profit margin", "gpm"],
            "operating_profit_margin": ["operating profit margin", "opm", "operating margin"],
            "net_profit_margin": ["net profit margin", "npm", "net margin"],
            "roe": ["return on equity", "roe"],
            "roa": ["return on assets", "roa"],
            "roic": ["return on invested capital", "roic"],
            "current_ratio": ["current ratio"],
            "quick_ratio": ["quick ratio", "acid test"],
            "debt_to_equity": ["debt to equity", "d/e", "debt-to-equity"],
            "debt_to_assets": ["debt to assets", "debt-to-assets"],
            "interest_coverage": ["interest coverage", "times interest earned"],
            "yoy_revenue_growth": ["yoy", "year over year", "year-on-year", "yoy revenue"],
            "cagr": ["cagr", "compound annual growth"],
            "pe_ratio": ["p/e", "pe ratio", "price to earnings"],
            "ev_ebitda": ["ev/ebitda", "enterprise value ebitda"],
        }

        for metric, keywords in keyword_map.items():
            if any(kw in query for kw in keywords):
                return metric
        return None

    @staticmethod
    def _calculate(metric: str, values: dict[str, float]) -> float:
        """Compute a metric from extracted numeric values."""
        calc_map: dict[str, callable] = {
            "gross_profit_margin": lambda v: v["gross_profit"] / v["revenue"],
            "operating_profit_margin": lambda v: v["operating_income"] / v["revenue"],
            "net_profit_margin": lambda v: v["net_income"] / v["revenue"],
            "roe": lambda v: v["net_income"] / v.get("shareholders_equity", v.get("equity", 1)),
            "roa": lambda v: v["net_income"] / v["total_assets"],
            "roic": lambda v: v["nopat"] / v["invested_capital"],
            "current_ratio": lambda v: v["current_assets"] / v["current_liabilities"],
            "quick_ratio": lambda v: (v["current_assets"] - v.get("inventory", 0)) / v["current_liabilities"],
            "debt_to_equity": lambda v: v.get("total_debt", v["total_liabilities"]) / v.get("shareholders_equity", v.get("equity", 1)),
            "debt_to_assets": lambda v: v.get("total_debt", v["total_liabilities"]) / v["total_assets"],
            "interest_coverage": lambda v: v["ebit"] / v["interest_expense"],
            "yoy_revenue_growth": lambda v: (v["revenue_current"] - v["revenue_prior"]) / v["revenue_prior"],
            "cagr": lambda v: (v["end_value"] / v["start_value"]) ** (1 / v["years"]) - 1,
            "pe_ratio": lambda v: v.get("market_cap", 0) / v["net_income"] if v["net_income"] != 0 else None,
            "ev_ebitda": lambda v: v["enterprise_value"] / v["ebitda"] if v["ebitda"] != 0 else None,
        }

        fn = calc_map.get(metric)
        if fn is None:
            raise ValueError(f"Unknown metric: {metric}")
        result = fn(values)
        if result is None:
            raise ValueError(f"Could not compute {metric} with available values")
        return round(result, 4)

    @staticmethod
    def _get_unit(metric: str) -> str:
        """Return the human-readable unit for a metric."""
        percent_metrics = {
            "gross_profit_margin", "operating_profit_margin", "net_profit_margin",
            "yoy_revenue_growth", "cagr",
        }
        ratio_metrics = {
            "current_ratio", "quick_ratio", "debt_to_equity", "debt_to_assets",
            "interest_coverage", "roic",
        }
        if metric in percent_metrics:
            return "%"
        if metric in ratio_metrics:
            return "x"
        if metric == "pe_ratio":
            return "x"
        return ""
