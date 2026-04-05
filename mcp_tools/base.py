"""
MCP Tool Base — Abstract base class for all MCP tools.
All tools implement the standard MCP tool interface with JSON Schema.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class MCPTool(ABC):
    """
    Abstract base class for MCP tools.

    Each tool exposes:
    - name       : str — unique tool identifier
    - description: str — human-readable description for LLM routing
    - parameters : dict — JSON Schema for the tool's input arguments
    - execute()  : callable — the actual tool implementation

    Usage as an MCP tool:
        tool = MyMCPTool()
        schema = tool.get_schema()          # For MCP server registration
        result = tool.execute(**kwargs)     # Actual invocation
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name (e.g. 'deepdoc_table_parse')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for LLM tool selection."""
        raise NotImplementedError

    @property
    def parameters(self) -> dict[str, Any]:
        """
        JSON Schema describing the tool's input parameters.

        Default returns an empty schema (no arguments).
        Subclasses should override to declare their actual parameters.
        """
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def get_schema(self) -> dict[str, Any]:
        """
        Return the full MCP tool schema dict.

        Includes name, description, and parameters per MCP spec.
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments."""
        raise NotImplementedError

    def to_json(self) -> str:
        """Serialize schema to JSON string."""
        return json.dumps(self.get_schema(), ensure_ascii=False, indent=2)
