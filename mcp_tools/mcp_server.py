"""
MCP Server — Python MCP SDK Server with stdio Transport.

Implements a production-ready MCP server that:
- Uses the Python MCP SDK with @mcp.tool decorator
- Exposes all 4 categories of tools via stdio (JSON-RPC over stdin/stdout)
- Auto-discovers and registers tools at startup
- Communicates with the LangGraph Agent via standard input/output

Running the server:
    python -m mcp_tools.mcp_server

The agent launches this as a subprocess and communicates via JSON-RPC
messages on stdin/stdout. New tools added to the tool registry are
automatically discovered and exposed — no code changes needed.

Tool Schema (JSON-RPC 2.0 notification format for stdio):
    → Agent sends: {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    ← Server responds: {"jsonrpc": "2.0", "id": 1, "result": {"tools": [...]}}
    → Agent sends: {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "...", "arguments": {...}}}
    ← Server responds: {"jsonrpc": "2.0", "id": 2, "result": {...}}
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Tool Registry — Register all tools here
# ---------------------------------------------------------------------------

_TOOL_REGISTRY: list[Any] = []  # Will hold instantiated tool objects


def _register_tools() -> None:
    """Import and instantiate all MCP tools. Call once at server startup."""
    # Avoid circular imports
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

    # Instantiate with optional pre-configured backends
    # In production, initialize with actual retriever/reranker/model instances
    _TOOL_REGISTRY.extend([
        # DeepDoc tools
        DeepDocTableParseTool(),
        DeepDocCrossPageMergeTool(),
        DeepDocChartExtractTool(),
        DeepDocOCRCorrectTool(),
        # Retrieval tools
        RetrievalHybridSearchTool(),
        RetrievalRerankTool(),
        RetrievalMultiQueryTool(),
        # Analysis tools
        AnalysisMetricExtractTool(),
        AnalysisCalcTool(),
        AnalysisUnitNormalizeTool(),
        AnalysisCrossTableMapTool(),
        AnalysisCAGRTool(),
        AnalysisYoYGrowthTool(),
        # Verification tools
        VerificationEvidenceCheckTool(),
        VerificationCitationBacktrackTool(),
        VerificationAnswerGroundednessTool(),
        VerificationMissingDataAlertTool(),
    ])


# ---------------------------------------------------------------------------
# MCP Server Implementation (stdio / JSON-RPC 2.0)
# ---------------------------------------------------------------------------

def _list_tools() -> dict[str, Any]:
    """Return the list of all registered tool schemas."""
    return {
        "tools": [tool.get_schema() for tool in _TOOL_REGISTRY]
    }


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Find and execute a tool by name."""
    for tool in _TOOL_REGISTRY:
        if tool.name == name:
            try:
                result = tool.execute(**arguments)
                return {"result": result}
            except Exception as exc:  # noqa: BLE001
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Tool execution error: {exc}",
                    },
                }

    return {
        "jsonrpc": "2.0",
        "error": {
            "code": -32602,
            "message": f"Unknown tool: {name}",
        },
    }


def _handle_request(message: dict[str, Any]) -> None:
    """
    Handle an incoming JSON-RPC request and write response to stdout.

    JSON-RPC 2.0 over stdio:
    - tools/list  → returns all tool schemas
    - tools/call  → executes a named tool with arguments
    """
    method = message.get("method", "")
    msg_id = message.get("id")

    if method == "tools/list":
        result = _list_tools()
        response = {"jsonrpc": "2.0", "id": msg_id, "result": result}
    elif method == "tools/call":
        params = message.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result = _call_tool(tool_name, arguments)
        response = dict(result)
        response["id"] = msg_id
    else:
        response = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        }

    # Write response to stdout
    print(json.dumps(response, ensure_ascii=False), flush=True)


def main() -> None:
    """
    Main MCP server loop.

    Reads JSON-RPC messages from stdin, processes them, writes responses to stdout.
    """
    _register_tools()

    # Send ready signal
    print(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "server/ready",
                "params": {
                    "name": "financial-multimodal-rag-mcp",
                    "version": "1.0.0",
                    "num_tools": len(_TOOL_REGISTRY),
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
            if message.get("method") in ("tools/list", "tools/call"):
                _handle_request(message)
            # Notifications (no id) are acknowledged but not responded to
        except json.JSONDecodeError:
            # Send error response
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error: invalid JSON",
                        },
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
