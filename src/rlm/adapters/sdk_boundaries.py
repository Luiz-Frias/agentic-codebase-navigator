"""
SDK Boundary Layer - Centralized handling of external Any types.

This module is the SINGLE SOURCE OF TRUTH for handling `Any` types from external SDKs.
All SDK-to-domain type conversions are centralized here for:

1. **Explicit Documentation**: Each boundary point documents what SDK returns Any and why
2. **Type Safety**: Domain code receives typed values, not Any
3. **Auditability**: One place to review all external type boundaries
4. **pyright Compliance**: This module lives in adapters/ where reportAny=false

SDK Boundaries Tracked:
----------------------
- Tool.execute() -> Any: User-defined tools can return arbitrary types
- LLM SDK responses: Various fields may be Any depending on provider

Design Decision:
---------------
We explicitly accept Any at these boundaries because:
- External SDKs have dynamic return types by design
- User-defined tools are inherently untyped
- The cost of wrapping everything in typed containers outweighs benefits

Usage:
------
    from rlm.adapters.sdk_boundaries import execute_tool_safely

    # In domain code:
    result = execute_tool_safely(tool, **arguments)
    # result is typed as ToolExecutionResult, not Any

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlm.domain.agent_ports import ToolPort


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """
    Typed container for tool execution results.

    Wraps the Any return from tool.execute() in a typed container that
    domain code can work with safely.

    Attributes:
        value: The raw result from tool execution (Any -> object boundary)
        error: Error message if execution failed, None otherwise

    """

    value: object
    error: str | None

    @property
    def is_success(self) -> bool:
        """Check if the tool execution succeeded."""
        return self.error is None

    @property
    def is_error(self) -> bool:
        """Check if the tool execution failed."""
        return self.error is not None


def execute_tool_safely(
    tool: ToolPort,
    **arguments: object,
) -> ToolExecutionResult:
    """
    Execute a tool and wrap the result in a typed container.

    This is the SDK boundary crossing point for tool execution.
    The tool.execute() method returns Any (tools can return anything),
    and we wrap it in ToolExecutionResult for type-safe domain code.

    Args:
        tool: The tool to execute (implements ToolPort)
        **arguments: Arguments to pass to the tool

    Returns:
        ToolExecutionResult with either the value or an error message

    SDK Boundary:
        tool.execute() -> Any (user-defined tools return arbitrary types)

    """
    try:
        # SDK BOUNDARY: Execute user-defined tool
        # Tools can return any type, wrapped in ToolExecutionResult for type safety
        raw_result = tool.execute(**arguments)
        return ToolExecutionResult(value=raw_result, error=None)
    except BaseException as exc:  # noqa: BLE001 - SDK boundary requires catching all
        # Tools are user-defined and can raise any exception type.
        # We catch all exceptions to prevent tool failures from crashing
        # the orchestrator and to report errors back to the LLM.
        return ToolExecutionResult(value=None, error=str(exc))
