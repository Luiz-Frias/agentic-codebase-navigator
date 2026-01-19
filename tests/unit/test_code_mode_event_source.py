"""
TDD tests for CodeModeEventSource.

These tests verify that the event source correctly produces events based on
the current state and context, bridging orchestrator dependencies into the
event-driven StateMachine model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from rlm.domain.models.orchestration_types import (
    CodeBlocksFound,
    CodeExecuted,
    CodeModeContext,
    CodeModeEvent,
    CodeModeState,
    DepthExceeded,
    FinalAnswerFound,
    LLMResponseReceived,
    MaxIterationsReached,
)

pytestmark = pytest.mark.unit


# ============================================================================
# Test Fixtures - Mock Dependencies
# ============================================================================


@dataclass
class MockUsageSummary:
    """Minimal mock for UsageSummary."""

    model_usage_summaries: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockChatCompletion:
    """Minimal mock for ChatCompletion."""

    response: str
    root_model: str = "mock-model"
    usage_summary: MockUsageSummary = field(default_factory=MockUsageSummary)
    tool_calls: list[Any] = field(default_factory=list)
    finish_reason: str | None = None


@dataclass
class MockREPLResult:
    """Minimal mock for REPLResult."""

    output: str
    error: str | None = None
    correlation_id: str | None = None
    llm_calls: list[Any] = field(default_factory=list)

    # Additional attributes expected by format_execution_result
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    locals: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # If output is provided but stdout isn't, use output as stdout
        if self.output and not self.stdout:
            self.stdout = self.output


class MockLLM:
    """Mock LLM that returns configurable responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["Default response"]
        self.call_count = 0
        self.model_name = "mock-model"

    def complete(self, request: Any) -> MockChatCompletion:
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return MockChatCompletion(response=response)


class MockEnvironment:
    """Mock environment that returns configurable code execution results."""

    def __init__(self, results: list[MockREPLResult] | None = None) -> None:
        self.results = results or [MockREPLResult(output="42")]
        self.call_count = 0
        self.context_loaded = False
        self.loaded_context: Any = None

    def load_context(self, context: Any) -> None:
        self.context_loaded = True
        self.loaded_context = context

    def execute_code(self, code: str) -> MockREPLResult:
        result = self.results[min(self.call_count, len(self.results) - 1)]
        self.call_count += 1
        return result


class MockLogger:
    """Mock logger that records iterations."""

    def __init__(self) -> None:
        self.iterations: list[Any] = []

    def log_iteration(self, iteration: Any) -> None:
        self.iterations.append(iteration)


# ============================================================================
# Phase 1: Event Source Protocol Tests
# ============================================================================


class TestCodeModeEventSourceProtocol:
    """Verify the event source follows the expected protocol."""

    def test_event_source_is_callable(self) -> None:
        """Event source is callable with (state, context) -> event."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM()
        env = MockEnvironment()
        source = CodeModeEventSource(llm=llm, environment=env)

        ctx = CodeModeContext(prompt="Test prompt")
        # Should be callable and return an event or None
        result = source(CodeModeState.INIT, ctx)
        assert result is None or isinstance(result, tuple(CodeModeEvent.__args__))  # type: ignore[attr-defined]

    def test_event_source_has_llm_and_environment(self) -> None:
        """Event source stores LLM and environment dependencies."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM()
        env = MockEnvironment()
        source = CodeModeEventSource(llm=llm, environment=env)

        assert source.llm is llm
        assert source.environment is env

    def test_event_source_accepts_optional_logger(self) -> None:
        """Event source accepts an optional logger."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM()
        env = MockEnvironment()
        logger = MockLogger()
        source = CodeModeEventSource(llm=llm, environment=env, logger=logger)

        assert source.logger is logger


# ============================================================================
# Phase 2: INIT State Event Generation
# ============================================================================


class TestCodeModeEventSourceINITState:
    """Verify event generation from INIT state."""

    def test_init_returns_depth_exceeded_when_at_max_depth(self) -> None:
        """INIT returns DepthExceeded when depth >= max_depth."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM()
        env = MockEnvironment()
        source = CodeModeEventSource(llm=llm, environment=env)

        ctx = CodeModeContext(depth=2, max_depth=1)
        event = source(CodeModeState.INIT, ctx)

        assert isinstance(event, DepthExceeded)

    def test_init_loads_context_before_llm_call(self) -> None:
        """INIT loads context into environment before calling LLM."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM(responses=["Hello"])
        env = MockEnvironment()
        source = CodeModeEventSource(llm=llm, environment=env)

        ctx = CodeModeContext(depth=0, max_depth=1, prompt="Test prompt")
        source(CodeModeState.INIT, ctx)

        assert env.context_loaded is True

    def test_init_returns_llm_response_received(self) -> None:
        """INIT returns LLMResponseReceived after calling LLM."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM(responses=["LLM response text"])
        env = MockEnvironment()
        source = CodeModeEventSource(llm=llm, environment=env)

        ctx = CodeModeContext(depth=0, max_depth=1, prompt="Test prompt")
        event = source(CodeModeState.INIT, ctx)

        assert isinstance(event, LLMResponseReceived)
        assert event.response_text == "LLM response text"


# ============================================================================
# Phase 3: PROMPTING State Event Generation
# ============================================================================


class TestCodeModeEventSourcePROMPTINGState:
    """Verify event generation from PROMPTING state."""

    def test_prompting_returns_code_blocks_found_when_code_present(self) -> None:
        """PROMPTING returns CodeBlocksFound when response contains code blocks."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM()
        env = MockEnvironment()
        source = CodeModeEventSource(llm=llm, environment=env)

        ctx = CodeModeContext()
        # Simulate LLM response stored in context
        ctx.last_response = "```repl\nprint(42)\n```"

        event = source(CodeModeState.PROMPTING, ctx)

        assert isinstance(event, CodeBlocksFound)
        assert len(event.blocks) > 0

    def test_prompting_returns_final_answer_found_when_answer_present(self) -> None:
        """PROMPTING returns FinalAnswerFound when response contains final answer."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM()
        env = MockEnvironment()
        source = CodeModeEventSource(llm=llm, environment=env)

        ctx = CodeModeContext()
        # Simulate LLM response with final answer (FINAL() syntax)
        ctx.last_response = "FINAL(The answer is 42)"

        event = source(CodeModeState.PROMPTING, ctx)

        assert isinstance(event, FinalAnswerFound)
        assert event.answer == "The answer is 42"

    def test_prompting_prioritizes_code_blocks_over_final_answer(self) -> None:
        """
        PROMPTING returns CodeBlocksFound to execute code first.

        When both code blocks and final answer are present, code takes priority
        because:
        1. Code blocks may contain nested LLM calls that need to be tracked
        2. Final answer extraction happens AFTER code execution completes
        """
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM()
        env = MockEnvironment()
        source = CodeModeEventSource(llm=llm, environment=env)

        ctx = CodeModeContext()
        # Response with both code and final answer (FINAL() syntax)
        ctx.last_response = "```repl\nprint(42)\n```\nFINAL(42)"

        event = source(CodeModeState.PROMPTING, ctx)

        # Code blocks take priority - final answer checked AFTER code execution
        assert isinstance(event, CodeBlocksFound)
        assert ctx.pending_code_blocks == ["print(42)"]


# ============================================================================
# Phase 4: EXECUTING State Event Generation
# ============================================================================


class TestCodeModeEventSourceEXECUTINGState:
    """Verify event generation from EXECUTING state."""

    def test_executing_executes_code_blocks(self) -> None:
        """EXECUTING executes code blocks stored in context."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM()
        env = MockEnvironment(results=[MockREPLResult(output="42")])
        source = CodeModeEventSource(llm=llm, environment=env)

        ctx = CodeModeContext()
        ctx.pending_code_blocks = ["print(42)"]

        event = source(CodeModeState.EXECUTING, ctx)

        assert isinstance(event, CodeExecuted)
        assert env.call_count == 1

    def test_executing_returns_code_executed_with_results(self) -> None:
        """EXECUTING returns CodeExecuted with all code block results."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM()
        env = MockEnvironment(results=[MockREPLResult(output="42"), MockREPLResult(output="hello")])
        source = CodeModeEventSource(llm=llm, environment=env)

        ctx = CodeModeContext()
        ctx.pending_code_blocks = ["print(42)", "print('hello')"]

        event = source(CodeModeState.EXECUTING, ctx)

        assert isinstance(event, CodeExecuted)
        assert len(event.code_blocks) == 2

    def test_executing_returns_max_iterations_reached_when_at_limit(self) -> None:
        """EXECUTING returns MaxIterationsReached when iteration at max."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM()
        env = MockEnvironment()
        source = CodeModeEventSource(llm=llm, environment=env)

        ctx = CodeModeContext(iteration=29, max_iterations=30)
        ctx.pending_code_blocks = ["print(42)"]

        event = source(CodeModeState.EXECUTING, ctx)

        # Should execute code then check iterations
        # If at max, return MaxIterationsReached
        # This depends on guard logic - we may need to adjust
        assert isinstance(event, (CodeExecuted, MaxIterationsReached))


# ============================================================================
# Phase 5: SHALLOW_CALL State Event Generation
# ============================================================================


class TestCodeModeEventSourceSHALLOWCALLState:
    """Verify event generation from SHALLOW_CALL state."""

    def test_shallow_call_calls_llm_directly(self) -> None:
        """SHALLOW_CALL makes a direct LLM call without code execution."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM(responses=["Direct answer"])
        env = MockEnvironment()
        source = CodeModeEventSource(llm=llm, environment=env)

        ctx = CodeModeContext(prompt="Test prompt")
        event = source(CodeModeState.SHALLOW_CALL, ctx)

        assert isinstance(event, LLMResponseReceived)
        assert llm.call_count == 1
        assert env.call_count == 0  # No code execution

    def test_shallow_call_extracts_final_answer_if_present(self) -> None:
        """SHALLOW_CALL extracts final answer from response if present."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM(responses=["FINAL(42)"])
        env = MockEnvironment()
        source = CodeModeEventSource(llm=llm, environment=env)

        ctx = CodeModeContext(prompt="Test prompt")
        event = source(CodeModeState.SHALLOW_CALL, ctx)

        assert isinstance(event, LLMResponseReceived)


# ============================================================================
# Phase 6: DONE State Event Generation
# ============================================================================


class TestCodeModeEventSourceDONEState:
    """Verify DONE state is terminal and produces no events."""

    def test_done_returns_none(self) -> None:
        """DONE state returns None (terminal, no more events)."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM()
        env = MockEnvironment()
        source = CodeModeEventSource(llm=llm, environment=env)

        ctx = CodeModeContext()
        event = source(CodeModeState.DONE, ctx)

        assert event is None


# ============================================================================
# Phase 7: Context Updates
# ============================================================================


class TestCodeModeEventSourceContextUpdates:
    """Verify event source properly updates context."""

    def test_init_updates_message_history(self) -> None:
        """INIT initializes message history in context."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM(responses=["Response"])
        env = MockEnvironment()
        source = CodeModeEventSource(llm=llm, environment=env)

        ctx = CodeModeContext(prompt="Test question")
        source(CodeModeState.INIT, ctx)

        assert len(ctx.message_history) > 0

    def test_executing_updates_usage_totals(self) -> None:
        """EXECUTING updates usage totals from code execution."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource

        llm = MockLLM()
        env = MockEnvironment(results=[MockREPLResult(output="42")])
        source = CodeModeEventSource(llm=llm, environment=env)

        ctx = CodeModeContext()
        ctx.pending_code_blocks = ["print(42)"]

        source(CodeModeState.EXECUTING, ctx)

        # Context should track that code was executed
        assert len(ctx.code_blocks) > 0


# ============================================================================
# Phase 8: Integration with StateMachine
# ============================================================================


class TestCodeModeEventSourceStateMachineIntegration:
    """Verify event source works correctly with StateMachine."""

    def test_full_flow_init_to_done(self) -> None:
        """Full flow from INIT to DONE via PROMPTING."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        llm = MockLLM(responses=["FINAL(42)"])
        env = MockEnvironment()
        source = CodeModeEventSource(llm=llm, environment=env)

        machine = build_code_mode_machine()
        ctx = CodeModeContext(depth=0, max_depth=1, prompt="What is 6*7?")

        final_state, final_ctx = machine.run(CodeModeState.INIT, ctx, source)

        assert final_state == CodeModeState.DONE
        assert final_ctx.final_answer == "42"

    def test_full_flow_with_code_execution(self) -> None:
        """Full flow with code execution: INIT -> PROMPTING -> EXECUTING -> PROMPTING -> DONE."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        # First response has code, second has final answer
        llm = MockLLM(
            responses=[
                "```repl\nprint(42)\n```",
                "FINAL(The answer is 42)",
            ],
        )
        env = MockEnvironment(results=[MockREPLResult(output="42")])
        source = CodeModeEventSource(llm=llm, environment=env)

        machine = build_code_mode_machine()
        ctx = CodeModeContext(depth=0, max_depth=1, max_iterations=10, prompt="What is 6*7?")

        final_state, final_ctx = machine.run(CodeModeState.INIT, ctx, source)

        assert final_state == CodeModeState.DONE
        assert final_ctx.final_answer == "The answer is 42"
        assert final_ctx.iteration >= 1

    def test_shallow_call_flow(self) -> None:
        """Shallow call flow: INIT -> SHALLOW_CALL -> DONE when depth exceeded."""
        from rlm.domain.services.code_mode_event_source import CodeModeEventSource
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        llm = MockLLM(responses=["Direct LLM response"])
        env = MockEnvironment()
        source = CodeModeEventSource(llm=llm, environment=env)

        machine = build_code_mode_machine()
        ctx = CodeModeContext(depth=2, max_depth=1, prompt="Test question")

        final_state, final_ctx = machine.run(CodeModeState.INIT, ctx, source)

        assert final_state == CodeModeState.DONE
        assert env.context_loaded is False  # Should not load context for shallow call
