"""
Agent capability ports (Phase 1 - Core, Phase 2.7 - Extension Protocols).

Extension points for tool calling, structured output, and agent shapes.
Follows existing hexagonal pattern from ports.py and goal2_ports.py.

These ports enable RLM to support pydantic-ai style agent patterns alongside
its native code execution paradigm. The `agent_mode` parameter in the
orchestrator determines which paradigm is active for a given run.

Phase 2.7 adds extension protocols for external apps to inject custom policies:
- StoppingPolicy: Control when iteration loops terminate
- ContextCompressor: Compress nested call returns before bubbling up
- NestedCallPolicy: Determine when nested calls should spawn sub-orchestrators
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from rlm.domain.models import ChatCompletion


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
    arguments: dict[str, object]  # Parsed arguments (JSON-compatible values)


class ToolCallResult(TypedDict):
    """
    Result of executing a tool.

    Returned to the LLM as context for the next iteration.
    """

    id: str  # Correlation ID from the request
    name: str  # Tool that was called
    result: object  # Return value (will be JSON serialized)
    error: str | None  # Error message if execution failed


class ToolMessage(TypedDict):
    """
    Message format for tool results in conversation history.

    This follows the OpenAI chat completion message format for tool results,
    which is used to inject tool execution results back into the conversation.

    Example:
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": '{"temperature": 72, "unit": "fahrenheit"}'
        }

    """

    role: str  # Always "tool"
    tool_call_id: str  # Correlation ID from the ToolCallRequest
    content: str  # JSON-serialized result or error message


class ToolPort(Protocol):
    """
    Port for a single tool/function.

    Implementations wrap Python callables and provide schema introspection
    for LLM tool calling. The definition property generates the JSON schema
    that the LLM uses to understand how to call the tool.
    """

    @property
    def definition(self) -> ToolDefinition:  # pyright: ignore[reportReturnType]
        """Return the tool's schema for LLM consumption."""

    def execute(self, **kwargs: object) -> object:  # pyright: ignore[reportReturnType]
        """Execute the tool synchronously with the given arguments."""

    async def aexecute(self, **kwargs: object) -> object:  # pyright: ignore[reportReturnType]
        """Execute the tool asynchronously with the given arguments."""


class ToolRegistryPort(Protocol):
    """
    Port for managing available tools.

    The registry maintains a collection of tools that can be offered to the
    LLM during a run. Tools are looked up by name when the LLM requests
    a tool call.
    """

    def register(self, tool: ToolPort, /) -> None:  # pyright: ignore[reportReturnType]
        """Register a tool in the registry."""

    def get(self, name: str, /) -> ToolPort | None:  # pyright: ignore[reportReturnType]
        """Look up a tool by name. Returns None if not found."""

    def list_definitions(self) -> list[ToolDefinition]:  # pyright: ignore[reportReturnType]
        """Return schemas for all registered tools (for LLM context)."""


class StructuredOutputPort[T](Protocol):
    """
    Port for validating/parsing structured LLM output.

    This enables type-safe extraction of structured data from LLM responses,
    similar to pydantic-ai's output_type functionality. The implementation
    uses Pydantic for validation.
    """

    def validate(self, response: str, output_type: type[T], /) -> T:  # pyright: ignore[reportReturnType]
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

    def get_schema(self, output_type: type[T], /) -> dict[str, Any]:  # pyright: ignore[reportReturnType]
        """
        Get the JSON schema for an output type.

        This schema can be provided to the LLM to guide its output format.
        """


# =============================================================================
# Extension Protocols (Phase 2.7)
# =============================================================================
#
# These protocols enable external apps to inject custom policies without
# modifying RLM core. Key use cases:
# - Debugging app: EIG-gated stopping, belief state tracking
# - MDP exploration: Custom stopping criteria based on entropy
# - Context management: Compress nested returns to fit context budgets


AgentModeName = Literal["code", "tools"]
"""Agent execution mode: 'code' for REPL execution, 'tools' for function calling."""


class NestedConfig(TypedDict, total=False):
    """
    Configuration for nested orchestrator spawning.

    Returned by NestedCallPolicy.get_nested_config() to configure how
    a nested orchestrator should behave.

    All fields are optional - defaults are inherited from parent orchestrator.
    """

    agent_mode: AgentModeName
    """Agent mode for nested calls ('code' or 'tools')."""

    tools: list[ToolDefinition]
    """Tools available to nested orchestrator (if agent_mode='tools')."""

    max_iterations: int
    """Maximum iterations for nested orchestrator loop."""

    max_depth: int
    """Maximum recursion depth for nested orchestrators."""


@runtime_checkable
class StoppingPolicy(Protocol):
    """
    Protocol for controlling when orchestrator iteration loops terminate.

    External apps can implement this to inject custom stopping criteria,
    such as EIG-gated stopping or entropy-based termination.

    The default implementation (DefaultStoppingPolicy) simply checks
    max_iterations.

    Example:
        class EIGStoppingPolicy:
            def __init__(self, min_eig: float = 0.1):
                self.min_eig = min_eig
                self.current_eig = 1.0

            def should_stop(self, context: dict[str, Any]) -> bool:
                return self.current_eig < self.min_eig

            def on_iteration_complete(
                self,
                context: dict[str, Any],
                result: ChatCompletion
            ) -> None:
                # Update EIG based on what was learned
                self.current_eig = compute_eig(context, result)

    """

    def should_stop(self, context: dict[str, Any]) -> bool:  # pyright: ignore[reportReturnType]
        """
        Return True to stop the iteration loop early.

        Called at the start of each iteration, before the LLM call.
        The context dict contains orchestrator state that external apps
        can use to make stopping decisions.

        Context keys (may vary by agent_mode):
            - iteration: Current iteration number (0-indexed)
            - max_iterations: Maximum allowed iterations
            - agent_mode: 'code' or 'tools'
            - depth: Current recursion depth
            - history: Conversation history (list of messages)
            - last_result: Previous ChatCompletion (None on first iteration)

        Returns:
            True to stop immediately, False to continue.

        """

    def on_iteration_complete(  # pyright: ignore[reportReturnType]
        self,
        context: dict[str, Any],
        result: ChatCompletion,
    ) -> None:
        """
        Called after each iteration completes.

        Use this to track state, update beliefs, compute metrics, etc.
        The context dict is mutable - modifications persist across iterations.

        Args:
            context: Mutable orchestrator state dict.
            result: The ChatCompletion from this iteration.

        """


@runtime_checkable
class ContextCompressor(Protocol):
    """
    Protocol for compressing nested call returns before bubbling up.

    When a nested orchestrator completes, its result may be too large
    to fit in the parent's context budget. Implementations can summarize,
    extract key information, or truncate as needed.

    The default implementation (NoOpContextCompressor) passes through
    unchanged.

    Example:
        class SummarizingCompressor:
            def __init__(self, llm: LLMPort, max_tokens: int = 500):
                self.llm = llm
                self.max_tokens = max_tokens

            def compress(self, result: str, max_tokens: int | None = None) -> str:
                limit = max_tokens or self.max_tokens
                if len(result) <= limit * 4:  # rough char estimate
                    return result
                # Use LLM to summarize
                summary = self.llm.complete(f"Summarize: {result}")
                return summary.response[:limit * 4]

    """

    def compress(self, result: str, max_tokens: int | None = None) -> str:  # pyright: ignore[reportReturnType]
        """
        Compress a nested call result before returning to parent.

        Args:
            result: The full result string from a nested orchestrator.
            max_tokens: Optional token budget hint. Implementations should
                try to keep the compressed result within this limit.

        Returns:
            Compressed result string suitable for parent context.

        """


@runtime_checkable
class NestedCallPolicy(Protocol):
    """
    Protocol for determining when nested llm_query() calls should
    spawn sub-orchestrators vs. simple LLM calls.

    In RLM's code execution mode, generated code can call `llm_query()`.
    By default, these are simple LLM calls. With a custom NestedCallPolicy,
    nested calls can spawn full orchestrators with their own tool/code
    execution capabilities.

    The default implementation (SimpleNestedCallPolicy) always returns
    False for should_orchestrate() - nested calls are simple LLM calls.

    Example:
        class DepthLimitedNestedPolicy:
            def __init__(self, max_depth: int = 2):
                self.max_depth = max_depth
                self._config = NestedConfig(
                    agent_mode="tools",
                    max_iterations=5,
                )

            def should_orchestrate(self, prompt: str, depth: int) -> bool:
                return depth < self.max_depth

            def get_nested_config(self) -> NestedConfig:
                return self._config

    """

    def should_orchestrate(self, prompt: str, depth: int) -> bool:  # pyright: ignore[reportReturnType]
        """
        Return True to spawn a nested orchestrator for this call.

        Args:
            prompt: The prompt being passed to the nested call.
            depth: Current recursion depth (0 = root orchestrator).

        Returns:
            True to spawn a nested orchestrator, False for simple LLM call.

        """

    def get_nested_config(self) -> NestedConfig:  # pyright: ignore[reportReturnType]
        """
        Return configuration for nested orchestrator.

        Only called when should_orchestrate() returns True.
        The returned config determines how the nested orchestrator behaves.

        Returns:
            Configuration dict for nested orchestrator.

        """
