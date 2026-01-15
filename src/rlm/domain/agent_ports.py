"""
Agent capability ports (Phase 1 - Core).

Extension points for tool calling, structured output, and agent shapes.
Follows existing hexagonal pattern from ports.py and goal2_ports.py.

These ports enable RLM to support pydantic-ai style agent patterns alongside
its native code execution paradigm. The `agent_mode` parameter in the
orchestrator determines which paradigm is active for a given run.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict


class ToolDefinition(TypedDict):
    """
    Schema for a callable tool.

    This follows the OpenAI function calling schema format, which is the
    de facto standard for LLM tool definitions.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters


class ToolCallRequest(TypedDict):
    """
    Request from LLM to invoke a tool.

    Parsed from the LLM's response when using tool calling mode.
    """

    id: str  # Unique ID for this tool call (for correlation)
    name: str  # Tool name to invoke
    arguments: dict[str, Any]  # Parsed arguments


class ToolCallResult(TypedDict):
    """
    Result of executing a tool.

    Returned to the LLM as context for the next iteration.
    """

    id: str  # Correlation ID from the request
    name: str  # Tool that was called
    result: Any  # Return value (will be JSON serialized)
    error: str | None  # Error message if execution failed


class ToolPort(Protocol):
    """
    Port for a single tool/function.

    Implementations wrap Python callables and provide schema introspection
    for LLM tool calling. The definition property generates the JSON schema
    that the LLM uses to understand how to call the tool.
    """

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool's schema for LLM consumption."""
        ...

    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool synchronously with the given arguments."""
        ...

    async def aexecute(self, **kwargs: Any) -> Any:
        """Execute the tool asynchronously with the given arguments."""
        ...


class ToolRegistryPort(Protocol):
    """
    Port for managing available tools.

    The registry maintains a collection of tools that can be offered to the
    LLM during a run. Tools are looked up by name when the LLM requests
    a tool call.
    """

    def register(self, tool: ToolPort, /) -> None:
        """Register a tool in the registry."""
        ...

    def get(self, name: str, /) -> ToolPort | None:
        """Look up a tool by name. Returns None if not found."""
        ...

    def list_definitions(self) -> list[ToolDefinition]:
        """Return schemas for all registered tools (for LLM context)."""
        ...


class StructuredOutputPort[T](Protocol):
    """
    Port for validating/parsing structured LLM output.

    This enables type-safe extraction of structured data from LLM responses,
    similar to pydantic-ai's output_type functionality. The implementation
    uses Pydantic for validation.
    """

    def validate(self, response: str, output_type: type[T], /) -> T:
        """
        Validate and parse an LLM response into the target type.

        Args:
            response: Raw LLM response (typically JSON string)
            output_type: Target type to parse into (Pydantic model, dataclass, etc.)

        Returns:
            Parsed and validated instance of output_type

        Raises:
            ValidationError: If the response doesn't match the expected schema
        """
        ...

    def get_schema(self, output_type: type[T], /) -> dict[str, Any]:
        """
        Get the JSON schema for an output type.

        This schema can be provided to the LLM to guide its output format.
        """
        ...
