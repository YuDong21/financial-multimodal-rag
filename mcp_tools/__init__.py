"""
mcp_tools: MCP protocol toolchain for financial multimodal RAG.

Tool Categories
---------------
Category 1 — DeepDoc Parsing:
  DeepDocTableParseTool, DeepDocCrossPageMergeTool,
  DeepDocChartExtractTool, DeepDocOCRCorrectTool

Category 2 — Retrieval Augmentation:
  RetrievalHybridSearchTool, RetrievalRerankTool, RetrievalMultiQueryTool

Category 3 — Analysis:
  AnalysisMetricExtractTool, AnalysisCalcTool, AnalysisUnitNormalizeTool,
  AnalysisCrossTableMapTool, AnalysisCAGRTool, AnalysisYoYGrowthTool

Category 4 — Verification:
  VerificationEvidenceCheckTool, VerificationCitationBacktrackTool,
  VerificationAnswerGroundednessTool, VerificationMissingDataAlertTool

MCP Server
----------
Run as: python -m mcp_tools.mcp_server
Communicates via stdio (JSON-RPC 2.0) with the LangGraph Agent.

MCP Client
----------
Use MCPClient to dynamically discover and invoke tools:
  >>> from mcp_tools.mcp_client import MCPClient
  >>> client = MCPClient()
  >>> client.start()
  >>> bindings = client.get_langgraph_tool_bindings()
"""

from .base import MCPTool
from .deepdoc_tools import (
    DeepDocTableParseTool,
    DeepDocCrossPageMergeTool,
    DeepDocChartExtractTool,
    DeepDocOCRCorrectTool,
)
from .retrieval_tools import (
    RetrievalHybridSearchTool,
    RetrievalRerankTool,
    RetrievalMultiQueryTool,
)
from .analysis_tools import (
    AnalysisMetricExtractTool,
    AnalysisCalcTool,
    AnalysisUnitNormalizeTool,
    AnalysisCrossTableMapTool,
    AnalysisCAGRTool,
    AnalysisYoYGrowthTool,
)
from .verification_tools import (
    VerificationEvidenceCheckTool,
    VerificationCitationBacktrackTool,
    VerificationAnswerGroundednessTool,
    VerificationMissingDataAlertTool,
)
from .mcp_server import main as mcp_server_main
from .mcp_client import MCPClient, ToolSchema, ToolCallResult

__all__ = [
    # Base
    "MCPTool",
    # Category 1
    "DeepDocTableParseTool",
    "DeepDocCrossPageMergeTool",
    "DeepDocChartExtractTool",
    "DeepDocOCRCorrectTool",
    # Category 2
    "RetrievalHybridSearchTool",
    "RetrievalRerankTool",
    "RetrievalMultiQueryTool",
    # Category 3
    "AnalysisMetricExtractTool",
    "AnalysisCalcTool",
    "AnalysisUnitNormalizeTool",
    "AnalysisCrossTableMapTool",
    "AnalysisCAGRTool",
    "AnalysisYoYGrowthTool",
    # Category 4
    "VerificationEvidenceCheckTool",
    "VerificationCitationBacktrackTool",
    "VerificationAnswerGroundednessTool",
    "VerificationMissingDataAlertTool",
    # Server & Client
    "mcp_server_main",
    "MCPClient",
    "ToolSchema",
    "ToolCallResult",
]
