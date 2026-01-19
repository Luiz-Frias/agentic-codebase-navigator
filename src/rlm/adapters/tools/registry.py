"""
In-memory tool registry implementation.

Provides a simple registry for managing tools during an RLM run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from rlm.adapters.base import BaseToolRegistryAdapter
from rlm.adapters.tools.native import NativeToolAdapter

if TYPE_CHECKING:
    from collections.abc import Callable

    from rlm.domain.agent_ports import ToolDefinition, ToolPort


@dataclass(slots=True)
class InMemoryToolRegistry(BaseToolRegistryAdapter):
    """
    Simple in-memory registry for tools.

    Stores tools by name and provides lookup and listing functionality.
    Automatically wraps plain callables as NativeToolAdapter instances.
    """

    _tools: dict[str, ToolPort] = field(default_factory=dict)

    def register(self, tool: ToolPort | Callable[..., Any], /) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: A ToolPort implementation or a plain callable.
                  Callables are automatically wrapped with NativeToolAdapter.

        Raises:
            ValueError: If a tool with the same name is already registered.

        """
        # Wrap plain callables - use duck typing (check for definition attribute)
        wrapped_tool: ToolPort
        if not hasattr(tool, "definition"):
            if callable(tool):
                wrapped_tool = NativeToolAdapter(tool)
            else:
                raise TypeError(f"Expected a callable or tool adapter, got {type(tool)}")
        else:
            wrapped_tool = cast("ToolPort", tool)

        tool_definition = wrapped_tool.definition
        name = tool_definition["name"]
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")

        self._tools[name] = wrapped_tool

    def get(self, name: str, /) -> ToolPort | None:
        """
        Look up a tool by name.

        Args:
            name: The tool name to look up.

        Returns:
            The tool if found, None otherwise.

        """
        return self._tools.get(name)

    def list_definitions(self) -> list[ToolDefinition]:
        """
        Return schemas for all registered tools.

        Returns:
            List of tool definitions in registration order.

        """
        return [tool.definition for tool in self._tools.values()]

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
