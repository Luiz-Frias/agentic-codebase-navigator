"""
TDD tests for tools-mode state machine integration.

These tests verify the StateMachine wiring for tools-mode orchestration.
They define the expected state transitions and action behaviors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rlm.domain.models.orchestration_types import (
    LLMResponseReceived,
    MaxIterationsReached,
    NoToolCalls,
    PolicyStop,
    ToolCallsFound,
    ToolsExecuted,
    ToolsModeContext,
    ToolsModeState,
)

pytestmark = pytest.mark.unit


# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockToolCallRequest:
    """Minimal mock for ToolCallRequest."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class MockToolCallResult:
    """Minimal mock for ToolCallResult."""

    id: str
    name: str
    result: Any
    error: str | None = None


# ============================================================================
# Phase 1: State Registration Tests
# ============================================================================


class TestToolsModeStateMachineStates:
    """Verify all tools-mode states are registered."""

    def test_machine_has_init_state(self) -> None:
        """INIT state is registered."""
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        assert ToolsModeState.INIT in machine._states

    def test_machine_has_prompting_state(self) -> None:
        """PROMPTING state is registered."""
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        assert ToolsModeState.PROMPTING in machine._states

    def test_machine_has_executing_tools_state(self) -> None:
        """EXECUTING_TOOLS state is registered."""
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        assert ToolsModeState.EXECUTING_TOOLS in machine._states

    def test_machine_has_done_state(self) -> None:
        """DONE state is registered as terminal."""
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        assert ToolsModeState.DONE in machine._states
        assert ToolsModeState.DONE in machine._terminal_states


# ============================================================================
# Phase 2: Transition Registration Tests
# ============================================================================


class TestToolsModeStateMachineTransitions:
    """Verify all tools-mode transitions are registered."""

    def test_init_to_prompting_on_llm_response(self) -> None:
        """INIT -> PROMPTING on LLMResponseReceived."""
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext()
        event = LLMResponseReceived(completion=None, response_text="thinking...")
        trans = machine._find_transition(ToolsModeState.INIT, event, ctx)
        assert trans is not None
        assert trans.to_state == ToolsModeState.PROMPTING

    def test_prompting_to_executing_tools_on_tool_calls_found(self) -> None:
        """PROMPTING -> EXECUTING_TOOLS on ToolCallsFound."""
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext()
        mock_call = MockToolCallRequest(id="1", name="search", arguments={})
        event = ToolCallsFound(tool_calls=[mock_call])  # type: ignore[list-item]
        trans = machine._find_transition(ToolsModeState.PROMPTING, event, ctx)
        assert trans is not None
        assert trans.to_state == ToolsModeState.EXECUTING_TOOLS

    def test_prompting_to_done_on_no_tool_calls(self) -> None:
        """PROMPTING -> DONE on NoToolCalls (final answer)."""
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext()
        event = NoToolCalls()
        trans = machine._find_transition(ToolsModeState.PROMPTING, event, ctx)
        assert trans is not None
        assert trans.to_state == ToolsModeState.DONE

    def test_executing_tools_to_prompting_on_tools_executed(self) -> None:
        """EXECUTING_TOOLS -> PROMPTING on ToolsExecuted (continue loop)."""
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext(iteration=0, max_iterations=10)
        event = ToolsExecuted(results=[])
        trans = machine._find_transition(ToolsModeState.EXECUTING_TOOLS, event, ctx)
        assert trans is not None
        assert trans.to_state == ToolsModeState.PROMPTING

    def test_executing_tools_to_done_on_max_iterations(self) -> None:
        """EXECUTING_TOOLS -> DONE on MaxIterationsReached."""
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext()
        event = MaxIterationsReached()
        trans = machine._find_transition(ToolsModeState.EXECUTING_TOOLS, event, ctx)
        assert trans is not None
        assert trans.to_state == ToolsModeState.DONE

    def test_prompting_to_done_on_policy_stop(self) -> None:
        """PROMPTING -> DONE on PolicyStop."""
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext()
        event = PolicyStop()
        trans = machine._find_transition(ToolsModeState.PROMPTING, event, ctx)
        assert trans is not None
        assert trans.to_state == ToolsModeState.DONE


# ============================================================================
# Phase 3: Action Tests
# ============================================================================


class TestToolsModeActions:
    """Verify actions modify context correctly."""

    def test_tools_executed_transition_has_no_iteration_action(self) -> None:
        """
        ToolsExecuted transition does NOT increment iteration.

        Iteration tracking was moved to ToolsModeEventSource for cleaner separation:
        - Event source: handles LLM calls, tool execution, iteration tracking
        - State machine: pure state transitions based on events
        """
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext(iteration=5)
        event = ToolsExecuted(results=[])

        trans = machine._find_transition(ToolsModeState.EXECUTING_TOOLS, event, ctx)
        assert trans is not None
        # Verify there's no action that would modify iteration
        if trans.action:
            trans.action(event, ctx)
        # Iteration should remain unchanged (event source handles this now)
        assert ctx.iteration == 5

    def test_action_stores_llm_response(self) -> None:
        """LLMResponseReceived action stores completion in context."""
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext()
        event = LLMResponseReceived(completion=None, response_text="response")

        trans = machine._find_transition(ToolsModeState.INIT, event, ctx)
        assert trans is not None
        if trans.action:
            trans.action(event, ctx)

        # The action stores the completion (None in this test)
        assert ctx.last_completion is None  # We passed None


# ============================================================================
# Phase 4: Guard Tests
# ============================================================================


class TestToolsModeGuards:
    """Verify guard behavior (now handled by event source, not state machine)."""

    def test_tools_executed_transition_has_no_guard(self) -> None:
        """
        ToolsExecuted transition has no guard.

        Guard logic was moved to ToolsModeEventSource:
        - Event source decides when to emit MaxIterationsReached vs ToolsExecuted
        - State machine just routes events to states without guarding
        """
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext(iteration=9, max_iterations=10)
        event = ToolsExecuted(results=[])

        # Transition should always succeed (no guard)
        trans = machine._find_transition(ToolsModeState.EXECUTING_TOOLS, event, ctx)
        assert trans is not None
        assert trans.guard is None  # No guard on this transition
        assert trans.to_state == ToolsModeState.PROMPTING

    def test_max_iterations_reached_always_transitions_to_done(self) -> None:
        """MaxIterationsReached always transitions to DONE (event source decides when to emit)."""
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext()
        event = MaxIterationsReached()

        trans = machine._find_transition(ToolsModeState.EXECUTING_TOOLS, event, ctx)
        assert trans is not None
        assert trans.to_state == ToolsModeState.DONE


# ============================================================================
# Phase 5: Integration Tests
# ============================================================================


class TestToolsModeStateMachineIntegration:
    """End-to-end tests for the tools-mode state machine."""

    def test_simple_tool_call_flow(self) -> None:
        """
        INIT -> PROMPTING -> EXECUTING_TOOLS -> PROMPTING -> DONE.

        Note: Iteration tracking is handled by ToolsModeEventSource, not the state machine.
        The state machine just routes events to states.
        """
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext(max_iterations=10)

        mock_call = MockToolCallRequest(id="1", name="search", arguments={"q": "test"})
        mock_result = MockToolCallResult(id="1", name="search", result="found")

        events = [
            LLMResponseReceived(completion=None, response_text="calling search..."),
            ToolCallsFound(tool_calls=[mock_call]),  # type: ignore[list-item]
            ToolsExecuted(results=[mock_result]),  # type: ignore[list-item]
            NoToolCalls(),  # Final answer
        ]
        event_iter = iter(events)

        def event_source(state: ToolsModeState, ctx: ToolsModeContext) -> Any:
            try:
                return next(event_iter)
            except StopIteration:
                return None

        final_state, final_ctx = machine.run(ToolsModeState.INIT, ctx, event_source)

        assert final_state == ToolsModeState.DONE
        # Note: iteration tracking is now done by event source, not state machine
        # This test verifies state machine routing, not iteration counting
        assert final_ctx.iteration == 0  # State machine doesn't modify iteration

    def test_no_tool_calls_direct_answer(self) -> None:
        """INIT -> PROMPTING -> DONE when LLM doesn't call tools."""
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext()

        events = [
            LLMResponseReceived(completion=None, response_text="I can answer directly"),
            NoToolCalls(),
        ]
        event_iter = iter(events)

        def event_source(state: ToolsModeState, ctx: ToolsModeContext) -> Any:
            try:
                return next(event_iter)
            except StopIteration:
                return None

        final_state, _ = machine.run(ToolsModeState.INIT, ctx, event_source)

        assert final_state == ToolsModeState.DONE

    def test_max_iterations_terminates(self) -> None:
        """
        State machine terminates on MaxIterationsReached.

        The event source (not the state machine) tracks iterations and emits
        MaxIterationsReached when the limit is hit. The state machine just
        routes this event to the DONE state.
        """
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext(max_iterations=10)

        mock_call = MockToolCallRequest(id="1", name="search", arguments={})

        events = [
            LLMResponseReceived(completion=None, response_text="working..."),
            ToolCallsFound(tool_calls=[mock_call]),  # type: ignore[list-item]
            # Event source detects max iterations and emits MaxIterationsReached
            MaxIterationsReached(),
        ]
        event_iter = iter(events)

        def event_source(state: ToolsModeState, ctx: ToolsModeContext) -> Any:
            try:
                return next(event_iter)
            except StopIteration:
                return None

        final_state, _ = machine.run(ToolsModeState.INIT, ctx, event_source)

        assert final_state == ToolsModeState.DONE

    def test_policy_stop_terminates(self) -> None:
        """State machine terminates on PolicyStop."""
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext()

        events = [
            LLMResponseReceived(completion=None, response_text="working..."),
            PolicyStop(),
        ]
        event_iter = iter(events)

        def event_source(state: ToolsModeState, ctx: ToolsModeContext) -> Any:
            try:
                return next(event_iter)
            except StopIteration:
                return None

        final_state, _ = machine.run(ToolsModeState.INIT, ctx, event_source)

        assert final_state == ToolsModeState.DONE
