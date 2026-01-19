"""
TDD tests for orchestration state machine types.

These tests define the contract for OrchestrationState, OrchestrationEvent,
and OrchestrationContext types used by the RLMOrchestrator StateMachine.
"""

from __future__ import annotations

from enum import Enum

import pytest

pytestmark = pytest.mark.unit


# ============================================================================
# Phase 1: State Enum Tests
# ============================================================================


class TestCodeModeState:
    """States for code-mode orchestration."""

    def test_code_mode_state_has_init(self) -> None:
        """Code mode starts in INIT state."""
        from rlm.domain.models.orchestration_types import CodeModeState

        assert hasattr(CodeModeState, "INIT")

    def test_code_mode_state_has_prompting(self) -> None:
        """Code mode has PROMPTING state for LLM calls."""
        from rlm.domain.models.orchestration_types import CodeModeState

        assert hasattr(CodeModeState, "PROMPTING")

    def test_code_mode_state_has_executing(self) -> None:
        """Code mode has EXECUTING state for code execution."""
        from rlm.domain.models.orchestration_types import CodeModeState

        assert hasattr(CodeModeState, "EXECUTING")

    def test_code_mode_state_has_done(self) -> None:
        """Code mode has DONE terminal state."""
        from rlm.domain.models.orchestration_types import CodeModeState

        assert hasattr(CodeModeState, "DONE")

    def test_code_mode_state_has_shallow_call(self) -> None:
        """Code mode has SHALLOW_CALL for depth-exceeded case."""
        from rlm.domain.models.orchestration_types import CodeModeState

        assert hasattr(CodeModeState, "SHALLOW_CALL")

    def test_code_mode_state_is_enum(self) -> None:
        """CodeModeState is a proper enum."""
        from rlm.domain.models.orchestration_types import CodeModeState

        assert issubclass(CodeModeState, Enum)


class TestToolsModeState:
    """States for tools-mode orchestration."""

    def test_tools_mode_state_has_init(self) -> None:
        """Tools mode starts in INIT state."""
        from rlm.domain.models.orchestration_types import ToolsModeState

        assert hasattr(ToolsModeState, "INIT")

    def test_tools_mode_state_has_prompting(self) -> None:
        """Tools mode has PROMPTING state for LLM calls."""
        from rlm.domain.models.orchestration_types import ToolsModeState

        assert hasattr(ToolsModeState, "PROMPTING")

    def test_tools_mode_state_has_executing_tools(self) -> None:
        """Tools mode has EXECUTING_TOOLS state for tool execution."""
        from rlm.domain.models.orchestration_types import ToolsModeState

        assert hasattr(ToolsModeState, "EXECUTING_TOOLS")

    def test_tools_mode_state_has_done(self) -> None:
        """Tools mode has DONE terminal state."""
        from rlm.domain.models.orchestration_types import ToolsModeState

        assert hasattr(ToolsModeState, "DONE")

    def test_tools_mode_state_is_enum(self) -> None:
        """ToolsModeState is a proper enum."""
        from rlm.domain.models.orchestration_types import ToolsModeState

        assert issubclass(ToolsModeState, Enum)


# ============================================================================
# Phase 2: Event Type Tests
# ============================================================================


class TestCodeModeEvents:
    """Events that trigger code-mode state transitions."""

    def test_llm_response_received_has_completion(self) -> None:
        """LLMResponseReceived carries the ChatCompletion."""
        from rlm.domain.models.orchestration_types import LLMResponseReceived

        event = LLMResponseReceived(completion=None, response_text="hello")
        assert hasattr(event, "completion")
        assert hasattr(event, "response_text")

    def test_code_blocks_found_has_blocks(self) -> None:
        """CodeBlocksFound carries the extracted code blocks."""
        from rlm.domain.models.orchestration_types import CodeBlocksFound

        event = CodeBlocksFound(blocks=["print(1)"])
        assert event.blocks == ["print(1)"]

    def test_code_executed_has_results(self) -> None:
        """CodeExecuted carries execution results."""
        from rlm.domain.models.orchestration_types import CodeExecuted

        event = CodeExecuted(code_blocks=[])
        assert hasattr(event, "code_blocks")

    def test_final_answer_found_has_answer(self) -> None:
        """FinalAnswerFound carries the extracted answer."""
        from rlm.domain.models.orchestration_types import FinalAnswerFound

        event = FinalAnswerFound(answer="42")
        assert event.answer == "42"

    def test_max_iterations_reached_exists(self) -> None:
        """MaxIterationsReached signals loop limit hit."""
        from rlm.domain.models.orchestration_types import MaxIterationsReached

        event = MaxIterationsReached()
        assert event is not None

    def test_depth_exceeded_exists(self) -> None:
        """DepthExceeded signals recursion limit hit."""
        from rlm.domain.models.orchestration_types import DepthExceeded

        event = DepthExceeded()
        assert event is not None


class TestToolsModeEvents:
    """Events that trigger tools-mode state transitions."""

    def test_tool_calls_found_has_calls(self) -> None:
        """ToolCallsFound carries the tool call requests."""
        from rlm.domain.models.orchestration_types import ToolCallsFound

        event = ToolCallsFound(tool_calls=[])
        assert hasattr(event, "tool_calls")

    def test_tools_executed_has_results(self) -> None:
        """ToolsExecuted carries tool execution results."""
        from rlm.domain.models.orchestration_types import ToolsExecuted

        event = ToolsExecuted(results=[])
        assert hasattr(event, "results")

    def test_no_tool_calls_exists(self) -> None:
        """NoToolCalls signals final answer (no tools requested)."""
        from rlm.domain.models.orchestration_types import NoToolCalls

        event = NoToolCalls()
        assert event is not None

    def test_policy_stop_exists(self) -> None:
        """PolicyStop signals custom policy stopped the loop."""
        from rlm.domain.models.orchestration_types import PolicyStop

        event = PolicyStop()
        assert event is not None


# ============================================================================
# Phase 3: Context Dataclass Tests
# ============================================================================


class TestCodeModeContext:
    """Context carried through code-mode state machine."""

    def test_context_has_iteration_counter(self) -> None:
        """Context tracks current iteration number."""
        from rlm.domain.models.orchestration_types import CodeModeContext

        ctx = CodeModeContext()
        assert hasattr(ctx, "iteration")
        assert ctx.iteration == 0

    def test_context_has_max_iterations(self) -> None:
        """Context has configurable max iterations."""
        from rlm.domain.models.orchestration_types import CodeModeContext

        ctx = CodeModeContext(max_iterations=50)
        assert ctx.max_iterations == 50

    def test_context_has_depth_tracking(self) -> None:
        """Context tracks recursion depth."""
        from rlm.domain.models.orchestration_types import CodeModeContext

        ctx = CodeModeContext(depth=2, max_depth=5)
        assert ctx.depth == 2
        assert ctx.max_depth == 5

    def test_context_has_message_history(self) -> None:
        """Context maintains conversation history."""
        from rlm.domain.models.orchestration_types import CodeModeContext

        ctx = CodeModeContext()
        assert hasattr(ctx, "message_history")
        assert isinstance(ctx.message_history, list)

    def test_context_has_usage_tracking(self) -> None:
        """Context accumulates token usage."""
        from rlm.domain.models.orchestration_types import CodeModeContext

        ctx = CodeModeContext()
        assert hasattr(ctx, "root_usage_totals")
        assert isinstance(ctx.root_usage_totals, dict)

    def test_context_has_time_tracking(self) -> None:
        """Context tracks execution timing."""
        from rlm.domain.models.orchestration_types import CodeModeContext

        ctx = CodeModeContext()
        assert hasattr(ctx, "time_start")

    def test_context_has_last_completion(self) -> None:
        """Context stores last LLM completion."""
        from rlm.domain.models.orchestration_types import CodeModeContext

        ctx = CodeModeContext()
        assert hasattr(ctx, "last_completion")
        assert ctx.last_completion is None

    def test_context_has_code_blocks(self) -> None:
        """Context stores extracted code blocks."""
        from rlm.domain.models.orchestration_types import CodeModeContext

        ctx = CodeModeContext()
        assert hasattr(ctx, "code_blocks")
        assert isinstance(ctx.code_blocks, list)

    def test_context_has_final_answer(self) -> None:
        """Context stores final answer when found."""
        from rlm.domain.models.orchestration_types import CodeModeContext

        ctx = CodeModeContext()
        assert hasattr(ctx, "final_answer")
        assert ctx.final_answer is None

    def test_context_has_correlation_id(self) -> None:
        """Context carries correlation ID for tracing."""
        from rlm.domain.models.orchestration_types import CodeModeContext

        ctx = CodeModeContext(correlation_id="trace-123")
        assert ctx.correlation_id == "trace-123"

    def test_context_has_prompt(self) -> None:
        """Context stores the original user prompt."""
        from rlm.domain.models.orchestration_types import CodeModeContext

        ctx = CodeModeContext(prompt="What is 2+2?")
        assert ctx.prompt == "What is 2+2?"


class TestToolsModeContext:
    """Context carried through tools-mode state machine."""

    def test_context_has_iteration_counter(self) -> None:
        """Context tracks current iteration number."""
        from rlm.domain.models.orchestration_types import ToolsModeContext

        ctx = ToolsModeContext()
        assert hasattr(ctx, "iteration")
        assert ctx.iteration == 0

    def test_context_has_conversation(self) -> None:
        """Context maintains tool conversation history."""
        from rlm.domain.models.orchestration_types import ToolsModeContext

        ctx = ToolsModeContext()
        assert hasattr(ctx, "conversation")
        assert isinstance(ctx.conversation, list)

    def test_context_has_tool_definitions(self) -> None:
        """Context has available tool definitions."""
        from rlm.domain.models.orchestration_types import ToolsModeContext

        ctx = ToolsModeContext()
        assert hasattr(ctx, "tool_definitions")

    def test_context_has_tool_choice(self) -> None:
        """Context has tool choice constraint."""
        from rlm.domain.models.orchestration_types import ToolsModeContext

        ctx = ToolsModeContext(tool_choice="auto")
        assert ctx.tool_choice == "auto"

    def test_context_has_policy_context(self) -> None:
        """Context carries policy context for StoppingPolicy."""
        from rlm.domain.models.orchestration_types import ToolsModeContext

        ctx = ToolsModeContext()
        assert hasattr(ctx, "policy_context")


# ============================================================================
# Phase 4: Type Union Tests
# ============================================================================


class TestOrchestrationEventUnion:
    """Event union type for all orchestration events."""

    def test_code_mode_event_union_includes_all_events(self) -> None:
        """CodeModeEvent is a union of all code-mode events."""
        from rlm.domain.models.orchestration_types import (
            CodeBlocksFound,
            CodeExecuted,
            CodeModeEvent,
            DepthExceeded,
            FinalAnswerFound,
            LLMResponseReceived,
            MaxIterationsReached,
        )

        # Should be able to assign any event to the union type
        event: CodeModeEvent = LLMResponseReceived(completion=None, response_text="")
        assert event is not None
        event = CodeBlocksFound(blocks=[])
        assert event is not None
        event = CodeExecuted(code_blocks=[])
        assert event is not None
        event = FinalAnswerFound(answer="")
        assert event is not None
        event = MaxIterationsReached()
        assert event is not None
        event = DepthExceeded()
        assert event is not None

    def test_tools_mode_event_union_includes_all_events(self) -> None:
        """ToolsModeEvent is a union of all tools-mode events."""
        from rlm.domain.models.orchestration_types import (
            LLMResponseReceived,
            MaxIterationsReached,
            NoToolCalls,
            PolicyStop,
            ToolCallsFound,
            ToolsExecuted,
            ToolsModeEvent,
        )

        # Should be able to assign any event to the union type
        event: ToolsModeEvent = LLMResponseReceived(completion=None, response_text="")
        assert event is not None
        event = ToolCallsFound(tool_calls=[])
        assert event is not None
        event = ToolsExecuted(results=[])
        assert event is not None
        event = NoToolCalls()
        assert event is not None
        event = MaxIterationsReached()
        assert event is not None
        event = PolicyStop()
        assert event is not None
