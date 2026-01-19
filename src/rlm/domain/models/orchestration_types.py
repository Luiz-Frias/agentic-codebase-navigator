"""
Orchestration state machine types.

This module defines the states, events, and contexts used by the RLMOrchestrator
StateMachine. These types model the control flow of both code-mode and tools-mode
orchestration loops.

Architecture:
    - States: Enum members representing discrete orchestration phases
    - Events: Dataclasses that trigger state transitions (carry relevant data)
    - Contexts: Mutable dataclasses that accumulate state across transitions

Design notes:
    - Code mode and tools mode have separate state/context types since their
      control flow semantics differ significantly
    - Events are shared where semantics overlap (LLMResponseReceived, MaxIterationsReached)
    - Context dataclasses are mutable to allow actions to update state in-place

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rlm.domain.agent_ports import ToolCallRequest, ToolCallResult, ToolDefinition
    from rlm.domain.models.completion import ChatCompletion
    from rlm.domain.models.iteration import CodeBlock
    from rlm.domain.models.llm_request import ToolChoice
    from rlm.domain.models.usage import ModelUsageSummary
    from rlm.domain.types import Prompt


# ============================================================================
# Code Mode States
# ============================================================================


class CodeModeState(Enum):
    """
    States for code-mode orchestration.

    State machine flow:
        INIT → SHALLOW_CALL (if depth exceeded)
        INIT → PROMPTING (normal case)
        PROMPTING → EXECUTING (code blocks found)
        PROMPTING → DONE (final answer found)
        EXECUTING → PROMPTING (continue iteration)
        EXECUTING → DONE (max iterations)

    """

    INIT = auto()
    """Initial state: validate configuration, load context."""

    SHALLOW_CALL = auto()
    """Depth exceeded: make a single LLM call without iteration."""

    PROMPTING = auto()
    """Calling the LLM with the current prompt."""

    EXECUTING = auto()
    """Executing code blocks extracted from LLM response."""

    DONE = auto()
    """Terminal state: final answer found or max iterations reached."""


# ============================================================================
# Tools Mode States
# ============================================================================


class ToolsModeState(Enum):
    """
    States for tools-mode orchestration.

    State machine flow:
        INIT → PROMPTING
        PROMPTING → EXECUTING_TOOLS (tool calls found)
        PROMPTING → DONE (no tool calls = final answer)
        EXECUTING_TOOLS → PROMPTING (continue iteration)
        EXECUTING_TOOLS → DONE (max iterations or policy stop)

    """

    INIT = auto()
    """Initial state: build conversation, validate tools."""

    PROMPTING = auto()
    """Calling the LLM with tools available."""

    EXECUTING_TOOLS = auto()
    """Executing tool calls from LLM response."""

    DONE = auto()
    """Terminal state: final answer or max iterations."""


# ============================================================================
# Shared Events
# ============================================================================


@dataclass(slots=True, frozen=True)
class LLMResponseReceived:
    """
    LLM returned a completion.

    Shared by both code-mode and tools-mode.
    """

    completion: ChatCompletion | None
    """The ChatCompletion object from the LLM (None for tests)."""

    response_text: str
    """The text content of the response."""


@dataclass(slots=True, frozen=True)
class MaxIterationsReached:
    """
    Iteration limit reached without finding final answer.

    Shared by both code-mode and tools-mode.
    """


# ============================================================================
# Code Mode Events
# ============================================================================


@dataclass(slots=True, frozen=True)
class CodeBlocksFound:
    """Code blocks extracted from LLM response."""

    blocks: list[str]
    """Raw code strings to execute."""


@dataclass(slots=True, frozen=True)
class CodeExecuted:
    """Code execution completed."""

    code_blocks: list[CodeBlock]
    """Executed code blocks with results."""


@dataclass(slots=True, frozen=True)
class FinalAnswerFound:
    """Final answer extracted from response."""

    answer: str
    """The extracted final answer text."""


@dataclass(slots=True, frozen=True)
class DepthExceeded:
    """
    Recursion depth limit exceeded.

    Triggers transition to SHALLOW_CALL state.
    """


# ============================================================================
# Tools Mode Events
# ============================================================================


@dataclass(slots=True, frozen=True)
class ToolCallsFound:
    """LLM requested tool calls."""

    tool_calls: list[ToolCallRequest]
    """The tool call requests from the LLM."""


@dataclass(slots=True, frozen=True)
class ToolsExecuted:
    """Tool execution completed."""

    results: list[ToolCallResult]
    """Results of executing the tool calls."""


@dataclass(slots=True, frozen=True)
class NoToolCalls:
    """
    LLM returned text without tool calls (final answer).

    In tools mode, this signals the LLM is done and has a final answer.
    """


@dataclass(slots=True, frozen=True)
class PolicyStop:
    """Custom StoppingPolicy requested early termination."""


# ============================================================================
# Event Unions
# ============================================================================

# Type aliases for event dispatch
CodeModeEvent = (
    LLMResponseReceived
    | CodeBlocksFound
    | CodeExecuted
    | FinalAnswerFound
    | MaxIterationsReached
    | DepthExceeded
)

ToolsModeEvent = (
    LLMResponseReceived
    | ToolCallsFound
    | ToolsExecuted
    | NoToolCalls
    | MaxIterationsReached
    | PolicyStop
)


# ============================================================================
# Code Mode Context
# ============================================================================


@dataclass
class CodeModeContext:
    """
    Mutable context for code-mode state machine.

    This dataclass accumulates state across iterations. Actions mutate
    fields in-place to carry state between transitions.

    """

    # Loop control
    iteration: int = 0
    """Current iteration number (0-indexed)."""

    max_iterations: int = 30
    """Maximum iterations before forcing termination."""

    depth: int = 0
    """Current recursion depth."""

    max_depth: int = 1
    """Maximum recursion depth before switching to shallow call."""

    # Time tracking
    time_start: float = 0.0
    """Start time (perf_counter) for execution timing."""

    # Message history
    message_history: list[dict[str, str]] = field(default_factory=list)
    """Accumulated conversation history for the LLM."""

    # Usage tracking
    root_usage_totals: dict[str, ModelUsageSummary] = field(default_factory=dict)
    """Running totals of token usage by model."""

    cumulative_usage_totals: dict[str, ModelUsageSummary] | None = None
    """Cumulative usage including nested calls (None if no logger)."""

    # Current iteration state
    last_completion: ChatCompletion | None = None
    """Most recent LLM completion."""

    last_response: str | None = None
    """Text content of the most recent LLM response."""

    code_blocks: list[CodeBlock] = field(default_factory=list)
    """Code blocks from current iteration."""

    pending_code_blocks: list[str] = field(default_factory=list)
    """Code blocks awaiting execution."""

    final_answer: str | None = None
    """Final answer when found (None until terminal state)."""

    # Configuration
    correlation_id: str | None = None
    """Tracing correlation ID."""

    root_prompt: str | None = None
    """Root prompt for iteration prompts."""

    prompt: Prompt | None = None
    """Original user prompt."""


# ============================================================================
# Tools Mode Context
# ============================================================================


@dataclass
class ToolsModeContext:
    """
    Mutable context for tools-mode state machine.

    This dataclass accumulates state across tool-calling iterations.
    """

    # Loop control
    iteration: int = 0
    """Current iteration number (0-indexed)."""

    max_iterations: int = 10
    """Maximum tool iterations before forcing termination."""

    depth: int = 0
    """Current recursion depth (for nested orchestration)."""

    # Time tracking
    time_start: float = 0.0
    """Start time (perf_counter) for execution timing."""

    # Conversation history
    conversation: list[dict[str, Any]] = field(default_factory=list)
    """Accumulated tool conversation history."""

    # Usage tracking
    usage_totals: dict[str, ModelUsageSummary] = field(default_factory=dict)
    """Running totals of token usage by model."""

    # Tool configuration
    tool_definitions: list[ToolDefinition] = field(default_factory=list)
    """Available tools for the LLM to call."""

    tool_choice: ToolChoice | None = None
    """Tool choice constraint (auto, none, required, specific)."""

    # Current iteration state
    last_completion: ChatCompletion | None = None
    """Most recent LLM completion."""

    last_response: str | None = None
    """Text content of the most recent LLM response."""

    pending_tool_calls: list[ToolCallRequest] = field(default_factory=list)
    """Tool calls awaiting execution."""

    # Policy context
    policy_stop: bool = False
    """True when a StoppingPolicy requested early termination."""

    policy_context: dict[str, Any] = field(default_factory=dict)
    """Context dict for StoppingPolicy callbacks."""

    # Configuration
    prompt: Prompt | None = None
    """Original user prompt."""
