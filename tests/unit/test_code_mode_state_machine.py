"""
TDD tests for code-mode state machine integration.

These tests verify the StateMachine wiring for code-mode orchestration.
They define the expected state transitions and action behaviors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rlm.domain.models.orchestration_types import (
    CodeBlocksFound,
    CodeExecuted,
    CodeModeContext,
    CodeModeState,
    DepthExceeded,
    FinalAnswerFound,
    LLMResponseReceived,
    MaxIterationsReached,
)

pytestmark = pytest.mark.unit


# ============================================================================
# Test Fixtures - Mock LLM and Environment
# ============================================================================


@dataclass
class MockCompletion:
    """Minimal mock for ChatCompletion."""

    response: str
    root_model: str = "mock-model"


@dataclass
class MockCodeBlock:
    """Minimal mock for CodeBlock."""

    code: str
    result: Any = None


# ============================================================================
# Phase 1: State Registration Tests
# ============================================================================


class TestCodeModeStateMachineStates:
    """Verify all code-mode states are registered."""

    def test_machine_has_init_state(self) -> None:
        """INIT state is registered."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        assert CodeModeState.INIT in machine._states

    def test_machine_has_shallow_call_state(self) -> None:
        """SHALLOW_CALL state is registered."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        assert CodeModeState.SHALLOW_CALL in machine._states

    def test_machine_has_prompting_state(self) -> None:
        """PROMPTING state is registered."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        assert CodeModeState.PROMPTING in machine._states

    def test_machine_has_executing_state(self) -> None:
        """EXECUTING state is registered."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        assert CodeModeState.EXECUTING in machine._states

    def test_machine_has_done_state(self) -> None:
        """DONE state is registered as terminal."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        assert CodeModeState.DONE in machine._states
        assert CodeModeState.DONE in machine._terminal_states


# ============================================================================
# Phase 2: Transition Registration Tests
# ============================================================================


class TestCodeModeStateMachineTransitions:
    """Verify all code-mode transitions are registered."""

    def test_init_to_shallow_call_on_depth_exceeded(self) -> None:
        """INIT -> SHALLOW_CALL on DepthExceeded."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        # Guard requires depth >= max_depth to pass
        ctx = CodeModeContext(depth=2, max_depth=1)
        event = DepthExceeded()
        trans = machine._find_transition(CodeModeState.INIT, event, ctx)
        assert trans is not None
        assert trans.to_state == CodeModeState.SHALLOW_CALL

    def test_init_to_prompting_on_llm_response(self) -> None:
        """INIT -> PROMPTING on LLMResponseReceived (normal case)."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        ctx = CodeModeContext()
        event = LLMResponseReceived(completion=None, response_text="hello")
        trans = machine._find_transition(CodeModeState.INIT, event, ctx)
        assert trans is not None
        assert trans.to_state == CodeModeState.PROMPTING

    def test_prompting_to_executing_on_code_blocks_found(self) -> None:
        """PROMPTING -> EXECUTING on CodeBlocksFound."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        ctx = CodeModeContext()
        event = CodeBlocksFound(blocks=["print(1)"])
        trans = machine._find_transition(CodeModeState.PROMPTING, event, ctx)
        assert trans is not None
        assert trans.to_state == CodeModeState.EXECUTING

    def test_prompting_to_done_on_final_answer(self) -> None:
        """PROMPTING -> DONE on FinalAnswerFound."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        ctx = CodeModeContext()
        event = FinalAnswerFound(answer="42")
        trans = machine._find_transition(CodeModeState.PROMPTING, event, ctx)
        assert trans is not None
        assert trans.to_state == CodeModeState.DONE

    def test_executing_to_prompting_on_code_executed(self) -> None:
        """EXECUTING -> PROMPTING on CodeExecuted (continue loop)."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        ctx = CodeModeContext(iteration=0, max_iterations=10)
        event = CodeExecuted(code_blocks=[])
        trans = machine._find_transition(CodeModeState.EXECUTING, event, ctx)
        assert trans is not None
        assert trans.to_state == CodeModeState.PROMPTING

    def test_executing_to_done_on_final_answer(self) -> None:
        """EXECUTING -> DONE on FinalAnswerFound."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        ctx = CodeModeContext()
        event = FinalAnswerFound(answer="42")
        trans = machine._find_transition(CodeModeState.EXECUTING, event, ctx)
        assert trans is not None
        assert trans.to_state == CodeModeState.DONE

    def test_executing_to_done_on_max_iterations(self) -> None:
        """EXECUTING -> DONE on MaxIterationsReached."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        ctx = CodeModeContext()
        event = MaxIterationsReached()
        trans = machine._find_transition(CodeModeState.EXECUTING, event, ctx)
        assert trans is not None
        assert trans.to_state == CodeModeState.DONE

    def test_shallow_call_to_done_on_llm_response(self) -> None:
        """SHALLOW_CALL -> DONE on LLMResponseReceived."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        ctx = CodeModeContext()
        event = LLMResponseReceived(completion=None, response_text="answer")
        trans = machine._find_transition(CodeModeState.SHALLOW_CALL, event, ctx)
        assert trans is not None
        assert trans.to_state == CodeModeState.DONE


# ============================================================================
# Phase 3: Action Tests
# ============================================================================


class TestCodeModeActions:
    """Verify actions modify context correctly."""

    def test_action_stores_final_answer(self) -> None:
        """FinalAnswerFound action stores answer in context."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        ctx = CodeModeContext()
        event = FinalAnswerFound(answer="42")

        # Find and execute the transition from PROMPTING
        trans = machine._find_transition(CodeModeState.PROMPTING, event, ctx)
        assert trans is not None
        if trans.action:
            trans.action(event, ctx)

        assert ctx.final_answer == "42"

    def test_code_executed_transition_has_no_iteration_action(self) -> None:
        """
        CodeExecuted transition does NOT increment iteration.

        Iteration tracking was moved to CodeModeEventSource for cleaner separation:
        - Event source: handles LLM calls, code execution, iteration tracking
        - State machine: pure state transitions based on events
        """
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        ctx = CodeModeContext(iteration=5)
        event = CodeExecuted(code_blocks=[])

        trans = machine._find_transition(CodeModeState.EXECUTING, event, ctx)
        assert trans is not None
        # Verify there's no action that would modify iteration
        if trans.action:
            trans.action(event, ctx)
        # Iteration should remain unchanged (event source handles this now)
        assert ctx.iteration == 5

    def test_code_executed_transition_does_not_store_code_blocks(self) -> None:
        """
        CodeExecuted transition does NOT store code blocks in context.

        Code block storage is handled by CodeModeEventSource:
        - Event source: executes code and stores results
        - State machine: pure state transitions based on events
        """
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        ctx = CodeModeContext()
        blocks = [MockCodeBlock(code="print(1)")]
        event = CodeExecuted(code_blocks=blocks)  # type: ignore[arg-type]

        trans = machine._find_transition(CodeModeState.EXECUTING, event, ctx)
        assert trans is not None
        if trans.action:
            trans.action(event, ctx)

        # Code blocks should remain unchanged (event source handles this now)
        assert ctx.code_blocks == []


# ============================================================================
# Phase 4: Guard Tests
# ============================================================================


class TestCodeModeGuards:
    """Verify guards control transition flow."""

    def test_depth_exceeded_guard_passes_when_depth_at_limit(self) -> None:
        """DepthExceeded guard passes when depth >= max_depth."""
        from rlm.domain.services.code_mode_machine import (
            depth_exceeded_guard,
        )

        ctx = CodeModeContext(depth=2, max_depth=2)
        event = DepthExceeded()

        assert depth_exceeded_guard(event, ctx) is True

    def test_depth_exceeded_guard_fails_when_depth_below_limit(self) -> None:
        """DepthExceeded guard fails when depth < max_depth."""
        from rlm.domain.services.code_mode_machine import depth_exceeded_guard

        ctx = CodeModeContext(depth=0, max_depth=2)
        event = DepthExceeded()

        assert depth_exceeded_guard(event, ctx) is False

    def test_continue_loop_guard_passes_when_iterations_remain(self) -> None:
        """CodeExecuted guard passes when iteration < max_iterations."""
        from rlm.domain.services.code_mode_machine import continue_loop_guard

        ctx = CodeModeContext(iteration=5, max_iterations=10)
        event = CodeExecuted(code_blocks=[])

        assert continue_loop_guard(event, ctx) is True

    def test_continue_loop_guard_fails_at_max_iterations(self) -> None:
        """CodeExecuted guard fails when iteration >= max_iterations."""
        from rlm.domain.services.code_mode_machine import continue_loop_guard

        ctx = CodeModeContext(iteration=10, max_iterations=10)
        event = CodeExecuted(code_blocks=[])

        assert continue_loop_guard(event, ctx) is False


# ============================================================================
# Phase 5: Integration Tests
# ============================================================================


class TestCodeModeStateMachineIntegration:
    """End-to-end tests for the code-mode state machine."""

    def test_simple_final_answer_flow(self) -> None:
        """INIT -> PROMPTING -> DONE when final answer found immediately."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        ctx = CodeModeContext(depth=0, max_depth=1)

        # Event source that simulates: start, LLM returns final answer
        events = [
            LLMResponseReceived(completion=None, response_text="thinking..."),
            FinalAnswerFound(answer="The answer is 42"),
        ]
        event_iter = iter(events)

        def event_source(state: CodeModeState, ctx: CodeModeContext) -> Any:
            try:
                return next(event_iter)
            except StopIteration:
                return None

        final_state, final_ctx = machine.run(CodeModeState.INIT, ctx, event_source)

        assert final_state == CodeModeState.DONE
        assert final_ctx.final_answer == "The answer is 42"

    def test_code_execution_loop_flow(self) -> None:
        """
        INIT -> PROMPTING -> EXECUTING -> PROMPTING -> DONE.

        Note: Iteration tracking is handled by CodeModeEventSource, not the state machine.
        The state machine just routes events to states without modifying iteration count.
        """
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        ctx = CodeModeContext(depth=0, max_depth=1, max_iterations=10)

        # Event source: start, code found, executed, final answer found
        events = [
            LLMResponseReceived(completion=None, response_text="Let me compute..."),
            CodeBlocksFound(blocks=["print(2+2)"]),
            CodeExecuted(code_blocks=[MockCodeBlock(code="print(2+2)", result="4")]),  # type: ignore[list-item]
            FinalAnswerFound(answer="4"),
        ]
        event_iter = iter(events)

        def event_source(state: CodeModeState, ctx: CodeModeContext) -> Any:
            try:
                return next(event_iter)
            except StopIteration:
                return None

        final_state, final_ctx = machine.run(CodeModeState.INIT, ctx, event_source)

        assert final_state == CodeModeState.DONE
        assert final_ctx.final_answer == "4"
        # Note: iteration tracking is now done by event source, not state machine
        # This test verifies state machine routing, not iteration counting
        assert final_ctx.iteration == 0  # State machine doesn't modify iteration

    def test_shallow_call_flow(self) -> None:
        """INIT -> SHALLOW_CALL -> DONE when depth exceeded."""
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        ctx = CodeModeContext(depth=2, max_depth=1)  # depth > max_depth

        # Event source: depth exceeded, then LLM response
        events = [
            DepthExceeded(),
            LLMResponseReceived(completion=None, response_text="shallow answer"),
        ]
        event_iter = iter(events)

        def event_source(state: CodeModeState, ctx: CodeModeContext) -> Any:
            try:
                return next(event_iter)
            except StopIteration:
                return None

        final_state, final_ctx = machine.run(CodeModeState.INIT, ctx, event_source)

        assert final_state == CodeModeState.DONE

    def test_max_iterations_terminates(self) -> None:
        """
        State machine terminates on MaxIterationsReached.

        The event source (not the state machine) tracks iterations and emits
        MaxIterationsReached when the limit is hit from EXECUTING state.
        The state machine just routes this event to DONE.
        """
        from rlm.domain.services.code_mode_machine import build_code_mode_machine

        machine = build_code_mode_machine()
        ctx = CodeModeContext(depth=0, max_depth=1, iteration=29, max_iterations=30)

        # Event source detects max iterations from EXECUTING and emits MaxIterationsReached
        # (instead of CodeExecuted which would continue the loop)
        events = [
            LLMResponseReceived(completion=None, response_text="working..."),
            CodeBlocksFound(blocks=["print(1)"]),
            # Event source detects max iterations and emits MaxIterationsReached directly
            MaxIterationsReached(),
        ]
        event_iter = iter(events)

        def event_source(state: CodeModeState, ctx: CodeModeContext) -> Any:
            try:
                return next(event_iter)
            except StopIteration:
                return None

        final_state, _ = machine.run(CodeModeState.INIT, ctx, event_source)

        assert final_state == CodeModeState.DONE
