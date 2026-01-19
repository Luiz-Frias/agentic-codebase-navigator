"""
TDD tests for StateMachine abstraction.

These tests define the contract BEFORE implementation.
The StateMachine pattern replaces complex nested loops (C901) with declarative
state transitions, making control flow explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import pytest

pytestmark = pytest.mark.unit


# ============================================================================
# Test Fixtures - Reusable state/event/context definitions
# ============================================================================


class SimpleState(Enum):
    """Simple state enum for basic tests."""

    INIT = auto()
    RUNNING = auto()
    DONE = auto()


@dataclass
class SimpleContext:
    """Simple mutable context for basic tests."""

    counter: int = 0
    log: list[str] = field(default_factory=list)


@dataclass
class StartEvent:
    """Event that triggers start transition."""


@dataclass
class StopEvent:
    """Event that triggers stop transition."""


@dataclass
class IncrementEvent:
    """Event that carries data."""

    amount: int = 1


# ============================================================================
# Phase 1: Basic State Definition and Registration
# ============================================================================


class TestStateMachineBasicStates:
    """Core state registration behavior."""

    def test_create_empty_state_machine(self) -> None:
        """Can create an empty state machine with type parameters."""
        from rlm.domain.models.state_machine import StateMachine

        machine: StateMachine[SimpleState, object, SimpleContext] = StateMachine()

        assert machine is not None

    def test_register_single_state(self) -> None:
        """Can register a state by name."""
        from rlm.domain.models.state_machine import StateMachine

        machine: StateMachine[SimpleState, object, SimpleContext] = StateMachine()
        result = machine.state(SimpleState.INIT)

        # Fluent API returns self
        assert result is machine

    def test_register_multiple_states(self) -> None:
        """Can register multiple states via chaining."""
        from rlm.domain.models.state_machine import StateMachine

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.INIT)
            .state(SimpleState.RUNNING)
            .state(SimpleState.DONE)
        )

        assert machine is not None

    def test_state_with_on_enter_callback(self) -> None:
        """Can register a state with an on_enter callback."""
        from rlm.domain.models.state_machine import StateMachine

        entered: list[str] = []

        def on_enter(ctx: SimpleContext) -> None:
            entered.append("entered")

        machine = StateMachine[SimpleState, object, SimpleContext]().state(
            SimpleState.INIT,
            on_enter=on_enter,
        )

        assert machine is not None
        # Callback not invoked until state is entered during run()

    def test_state_with_on_exit_callback(self) -> None:
        """Can register a state with an on_exit callback."""
        from rlm.domain.models.state_machine import StateMachine

        exited: list[str] = []

        def on_exit(ctx: SimpleContext) -> None:
            exited.append("exited")

        machine = StateMachine[SimpleState, object, SimpleContext]().state(
            SimpleState.INIT,
            on_exit=on_exit,
        )

        assert machine is not None

    def test_state_with_both_callbacks(self) -> None:
        """Can register a state with both on_enter and on_exit callbacks."""
        from rlm.domain.models.state_machine import StateMachine

        log: list[str] = []

        machine = StateMachine[SimpleState, object, SimpleContext]().state(
            SimpleState.RUNNING,
            on_enter=lambda ctx: log.append("enter"),
            on_exit=lambda ctx: log.append("exit"),
        )

        assert machine is not None


# ============================================================================
# Phase 2: Transition Definitions
# ============================================================================


class TestStateMachineTransitions:
    """Transition registration and event-based dispatch."""

    def test_register_single_transition(self) -> None:
        """Can register a transition from one state to another on an event."""
        from rlm.domain.models.state_machine import StateMachine

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.INIT)
            .state(SimpleState.RUNNING)
            .transition(SimpleState.INIT, StartEvent, SimpleState.RUNNING)
        )

        assert machine is not None

    def test_register_multiple_transitions(self) -> None:
        """Can register multiple transitions."""
        from rlm.domain.models.state_machine import StateMachine

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.INIT)
            .state(SimpleState.RUNNING)
            .state(SimpleState.DONE)
            .transition(SimpleState.INIT, StartEvent, SimpleState.RUNNING)
            .transition(SimpleState.RUNNING, StopEvent, SimpleState.DONE)
        )

        assert machine is not None

    def test_transition_returns_self_for_chaining(self) -> None:
        """transition() returns self for fluent API."""
        from rlm.domain.models.state_machine import StateMachine

        machine: StateMachine[SimpleState, object, SimpleContext] = StateMachine()
        result = machine.transition(SimpleState.INIT, StartEvent, SimpleState.RUNNING)

        assert result is machine

    def test_multiple_transitions_from_same_state(self) -> None:
        """Can register multiple transitions from the same state (different events)."""
        from rlm.domain.models.state_machine import StateMachine

        @dataclass
        class ErrorEvent:
            message: str

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.RUNNING)
            .state(SimpleState.DONE)
            .transition(SimpleState.RUNNING, StopEvent, SimpleState.DONE)
            .transition(SimpleState.RUNNING, ErrorEvent, SimpleState.DONE)
        )

        assert machine is not None


# ============================================================================
# Phase 3: Guards (Conditional Transitions)
# ============================================================================


class TestStateMachineGuards:
    """Guard conditions for conditional transitions."""

    def test_transition_with_guard(self) -> None:
        """Can register a transition with a guard predicate."""
        from rlm.domain.models.state_machine import StateMachine

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.RUNNING)
            .state(SimpleState.DONE)
            .transition(
                SimpleState.RUNNING,
                IncrementEvent,
                SimpleState.DONE,
                guard=lambda event, ctx: ctx.counter >= 10,
            )
        )

        assert machine is not None

    def test_guard_receives_event_and_context(self) -> None:
        """Guard predicate receives both event and context."""
        from rlm.domain.models.state_machine import StateMachine

        received: list[tuple[object, SimpleContext]] = []

        def capture_guard(event: IncrementEvent, ctx: SimpleContext) -> bool:
            received.append((event, ctx))
            return True

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.RUNNING)
            .state(SimpleState.DONE)
            .transition(
                SimpleState.RUNNING,
                IncrementEvent,
                SimpleState.DONE,
                guard=capture_guard,
            )
        )

        assert machine is not None
        # Guard not invoked until run()


# ============================================================================
# Phase 4: Actions (Side Effects on Transition)
# ============================================================================


class TestStateMachineActions:
    """Action callbacks executed during transitions."""

    def test_transition_with_action(self) -> None:
        """Can register a transition with an action callback."""
        from rlm.domain.models.state_machine import StateMachine

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.RUNNING)
            .state(SimpleState.DONE)
            .transition(
                SimpleState.RUNNING,
                IncrementEvent,
                SimpleState.DONE,
                action=lambda event, ctx: setattr(ctx, "counter", ctx.counter + event.amount),
            )
        )

        assert machine is not None

    def test_action_receives_event_and_context(self) -> None:
        """Action callback receives both event and context."""
        from rlm.domain.models.state_machine import StateMachine

        received: list[tuple[object, SimpleContext]] = []

        def capture_action(event: IncrementEvent, ctx: SimpleContext) -> None:
            received.append((event, ctx))

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.RUNNING)
            .state(SimpleState.DONE)
            .transition(
                SimpleState.RUNNING,
                IncrementEvent,
                SimpleState.DONE,
                action=capture_action,
            )
        )

        assert machine is not None

    def test_transition_with_guard_and_action(self) -> None:
        """Can register a transition with both guard and action."""
        from rlm.domain.models.state_machine import StateMachine

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.RUNNING)
            .state(SimpleState.DONE)
            .transition(
                SimpleState.RUNNING,
                IncrementEvent,
                SimpleState.DONE,
                guard=lambda e, c: c.counter >= 10,
                action=lambda e, c: c.log.append("done"),
            )
        )

        assert machine is not None


# ============================================================================
# Phase 5: Terminal States
# ============================================================================


class TestStateMachineTerminalStates:
    """Terminal states that stop execution."""

    def test_mark_single_terminal_state(self) -> None:
        """Can mark a state as terminal."""
        from rlm.domain.models.state_machine import StateMachine

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.DONE)
            .terminal(SimpleState.DONE)
        )

        assert machine is not None

    def test_mark_multiple_terminal_states(self) -> None:
        """Can mark multiple states as terminal."""
        from rlm.domain.models.state_machine import StateMachine

        class ExtendedState(Enum):
            INIT = auto()
            RUNNING = auto()
            DONE = auto()
            ERROR = auto()

        machine = (
            StateMachine[ExtendedState, object, SimpleContext]()
            .state(ExtendedState.DONE)
            .state(ExtendedState.ERROR)
            .terminal(ExtendedState.DONE, ExtendedState.ERROR)
        )

        assert machine is not None

    def test_terminal_returns_self_for_chaining(self) -> None:
        """terminal() returns self for fluent API."""
        from rlm.domain.models.state_machine import StateMachine

        machine: StateMachine[SimpleState, object, SimpleContext] = StateMachine()
        result = machine.terminal(SimpleState.DONE)

        assert result is machine


# ============================================================================
# Phase 6: Synchronous Execution (run)
# ============================================================================


class TestStateMachineSyncExecution:
    """Synchronous execution via run()."""

    def test_run_starts_in_initial_state(self) -> None:
        """run() begins execution in the specified initial state."""
        from rlm.domain.models.state_machine import StateMachine

        entered_states: list[SimpleState] = []

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.INIT, on_enter=lambda c: entered_states.append(SimpleState.INIT))
            .terminal(SimpleState.INIT)
        )

        context = SimpleContext()
        final_state, final_ctx = machine.run(
            SimpleState.INIT,
            context,
            event_source=lambda s, c: None,  # No events - stay in terminal INIT
        )

        assert final_state == SimpleState.INIT
        assert SimpleState.INIT in entered_states

    def test_run_stops_at_terminal_state(self) -> None:
        """run() stops when a terminal state is reached."""
        from rlm.domain.models.state_machine import StateMachine

        events = iter([StartEvent(), StopEvent()])

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.INIT)
            .state(SimpleState.RUNNING)
            .state(SimpleState.DONE)
            .transition(SimpleState.INIT, StartEvent, SimpleState.RUNNING)
            .transition(SimpleState.RUNNING, StopEvent, SimpleState.DONE)
            .terminal(SimpleState.DONE)
        )

        def event_source(state: SimpleState, ctx: SimpleContext) -> object | None:
            try:
                return next(events)
            except StopIteration:
                return None

        context = SimpleContext()
        final_state, _ = machine.run(SimpleState.INIT, context, event_source)

        assert final_state == SimpleState.DONE

    def test_run_executes_on_enter_callback(self) -> None:
        """run() executes on_enter when entering a state."""
        from rlm.domain.models.state_machine import StateMachine

        log: list[str] = []

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.INIT, on_enter=lambda c: log.append("enter_init"))
            .state(SimpleState.DONE, on_enter=lambda c: log.append("enter_done"))
            .transition(SimpleState.INIT, StartEvent, SimpleState.DONE)
            .terminal(SimpleState.DONE)
        )

        events = iter([StartEvent()])
        final_state, _ = machine.run(
            SimpleState.INIT,
            SimpleContext(),
            lambda s, c: next(events, None),
        )

        assert "enter_init" in log
        assert "enter_done" in log

    def test_run_executes_on_exit_callback(self) -> None:
        """run() executes on_exit when leaving a state."""
        from rlm.domain.models.state_machine import StateMachine

        log: list[str] = []

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.INIT, on_exit=lambda c: log.append("exit_init"))
            .state(SimpleState.DONE)
            .transition(SimpleState.INIT, StartEvent, SimpleState.DONE)
            .terminal(SimpleState.DONE)
        )

        events = iter([StartEvent()])
        machine.run(SimpleState.INIT, SimpleContext(), lambda s, c: next(events, None))

        assert "exit_init" in log

    def test_run_executes_transition_action(self) -> None:
        """run() executes the action callback during transition."""
        from rlm.domain.models.state_machine import StateMachine

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.INIT)
            .state(SimpleState.DONE)
            .transition(
                SimpleState.INIT,
                IncrementEvent,
                SimpleState.DONE,
                action=lambda e, c: setattr(c, "counter", c.counter + e.amount),
            )
            .terminal(SimpleState.DONE)
        )

        events = iter([IncrementEvent(amount=5)])
        context = SimpleContext(counter=0)
        _, final_ctx = machine.run(SimpleState.INIT, context, lambda s, c: next(events, None))

        assert final_ctx.counter == 5

    def test_run_respects_guard_condition(self) -> None:
        """run() only takes transition if guard returns True."""
        from rlm.domain.models.state_machine import StateMachine

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.RUNNING)
            .state(SimpleState.DONE)
            .transition(
                SimpleState.RUNNING,
                IncrementEvent,
                SimpleState.DONE,
                guard=lambda e, c: c.counter >= 10,  # Only transition when counter >= 10
                action=lambda e, c: c.log.append("transitioned"),
            )
            .transition(
                SimpleState.RUNNING,
                IncrementEvent,
                SimpleState.RUNNING,  # Stay in RUNNING if guard fails
                action=lambda e, c: setattr(c, "counter", c.counter + e.amount),
            )
            .terminal(SimpleState.DONE)
        )

        context = SimpleContext(counter=0)

        # Generate events dynamically until state machine reaches terminal state
        # The event_source keeps generating IncrementEvent(3) until DONE is reached
        _, final_ctx = machine.run(
            SimpleState.RUNNING,
            context,
            lambda s, c: IncrementEvent(3) if s == SimpleState.RUNNING else None,
        )

        assert final_ctx.counter >= 10
        assert "transitioned" in final_ctx.log

    def test_run_returns_context_mutations(self) -> None:
        """run() returns the mutated context."""
        from rlm.domain.models.state_machine import StateMachine

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.INIT)
            .state(SimpleState.DONE)
            .transition(
                SimpleState.INIT,
                StartEvent,
                SimpleState.DONE,
                action=lambda e, c: c.log.append("mutation"),
            )
            .terminal(SimpleState.DONE)
        )

        context = SimpleContext()
        _, final_ctx = machine.run(
            SimpleState.INIT,
            context,
            lambda s, c: StartEvent() if s == SimpleState.INIT else None,
        )

        assert "mutation" in final_ctx.log

    def test_run_stops_when_no_event(self) -> None:
        """run() stops when event_source returns None (no valid transition)."""
        from rlm.domain.models.state_machine import StateMachine

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.INIT)
            .state(SimpleState.RUNNING)
            .transition(SimpleState.INIT, StartEvent, SimpleState.RUNNING)
            # No terminal states - relies on event_source returning None
        )

        call_count = 0

        def event_source(s: SimpleState, c: SimpleContext) -> object | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return StartEvent()
            return None  # Stop after first event

        final_state, _ = machine.run(SimpleState.INIT, SimpleContext(), event_source)

        assert final_state == SimpleState.RUNNING


# ============================================================================
# Phase 7: Asynchronous Execution (arun)
# ============================================================================


class TestStateMachineAsyncExecution:
    """Asynchronous execution via arun()."""

    async def test_arun_basic_execution(self) -> None:
        """arun() executes state machine asynchronously."""
        from rlm.domain.models.state_machine import StateMachine

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.INIT)
            .state(SimpleState.DONE)
            .transition(SimpleState.INIT, StartEvent, SimpleState.DONE)
            .terminal(SimpleState.DONE)
        )

        async def async_event_source(
            state: SimpleState,
            ctx: SimpleContext,
        ) -> object | None:
            if state == SimpleState.INIT:
                return StartEvent()
            return None

        context = SimpleContext()
        final_state, _ = await machine.arun(SimpleState.INIT, context, async_event_source)

        assert final_state == SimpleState.DONE

    async def test_arun_executes_callbacks(self) -> None:
        """arun() executes on_enter/on_exit callbacks."""
        from rlm.domain.models.state_machine import StateMachine

        log: list[str] = []

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(
                SimpleState.INIT,
                on_enter=lambda c: log.append("enter"),
                on_exit=lambda c: log.append("exit"),
            )
            .state(SimpleState.DONE)
            .transition(SimpleState.INIT, StartEvent, SimpleState.DONE)
            .terminal(SimpleState.DONE)
        )

        async def async_event_source(s: SimpleState, c: SimpleContext) -> object | None:
            return StartEvent() if s == SimpleState.INIT else None

        await machine.arun(SimpleState.INIT, SimpleContext(), async_event_source)

        assert "enter" in log
        assert "exit" in log

    async def test_arun_executes_actions(self) -> None:
        """arun() executes transition actions."""
        from rlm.domain.models.state_machine import StateMachine

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.INIT)
            .state(SimpleState.DONE)
            .transition(
                SimpleState.INIT,
                IncrementEvent,
                SimpleState.DONE,
                action=lambda e, c: setattr(c, "counter", e.amount),
            )
            .terminal(SimpleState.DONE)
        )

        async def async_event_source(s: SimpleState, c: SimpleContext) -> object | None:
            return IncrementEvent(42) if s == SimpleState.INIT else None

        context = SimpleContext()
        _, final_ctx = await machine.arun(SimpleState.INIT, context, async_event_source)

        assert final_ctx.counter == 42


# ============================================================================
# Phase 8: Real-World Pattern - Orchestrator Loop
# ============================================================================


class TestStateMachineOrchestratorPattern:
    """Tests that mirror the rlm_orchestrator.py agent loop pattern."""

    def test_orchestrator_style_state_machine(self) -> None:
        """Pattern from rlm_orchestrator.py - completion() loop."""
        from rlm.domain.models.state_machine import StateMachine

        # Orchestrator-like states
        class OrchestratorState(Enum):
            INIT = auto()
            PROMPTING = auto()
            EXECUTING = auto()
            DONE = auto()

        # Orchestrator-like events
        @dataclass
        class LLMResponseReceived:
            response: str
            has_code: bool

        @dataclass
        class CodeExecuted:
            output: str

        @dataclass
        class FinalAnswerFound:
            answer: str

        # Orchestrator-like context
        @dataclass
        class OrchestratorContext:
            iteration: int = 0
            max_iterations: int = 3
            messages: list[str] = field(default_factory=list)
            final_answer: str | None = None

        # Build the state machine
        machine = (
            StateMachine[OrchestratorState, object, OrchestratorContext]()
            .state(OrchestratorState.INIT)
            .state(OrchestratorState.PROMPTING)
            .state(OrchestratorState.EXECUTING)
            .state(OrchestratorState.DONE)
            # INIT -> PROMPTING (always)
            .transition(
                OrchestratorState.INIT,
                StartEvent,
                OrchestratorState.PROMPTING,
            )
            # PROMPTING -> EXECUTING (if response has code)
            .transition(
                OrchestratorState.PROMPTING,
                LLMResponseReceived,
                OrchestratorState.EXECUTING,
                guard=lambda e, c: e.has_code and c.iteration < c.max_iterations,
                action=lambda e, c: c.messages.append(e.response),
            )
            # PROMPTING -> DONE (if final answer found)
            .transition(
                OrchestratorState.PROMPTING,
                FinalAnswerFound,
                OrchestratorState.DONE,
                action=lambda e, c: setattr(c, "final_answer", e.answer),
            )
            # EXECUTING -> PROMPTING (after code execution)
            .transition(
                OrchestratorState.EXECUTING,
                CodeExecuted,
                OrchestratorState.PROMPTING,
                action=lambda e, c: (
                    c.messages.append(f"Output: {e.output}"),
                    setattr(c, "iteration", c.iteration + 1),
                ),
            )
            .terminal(OrchestratorState.DONE)
        )

        # Simulate orchestrator execution
        events = iter(
            [
                StartEvent(),
                LLMResponseReceived("```python\nprint('hello')```", has_code=True),
                CodeExecuted("hello"),
                FinalAnswerFound("The output is 'hello'"),
            ],
        )

        context = OrchestratorContext()
        final_state, final_ctx = machine.run(
            OrchestratorState.INIT,
            context,
            lambda s, c: next(events, None),
        )

        assert final_state == OrchestratorState.DONE
        assert final_ctx.final_answer == "The output is 'hello'"
        assert final_ctx.iteration == 1
        assert len(final_ctx.messages) >= 1

    def test_max_iterations_guard(self) -> None:
        """Orchestrator stops at max_iterations via guard."""
        from rlm.domain.models.state_machine import StateMachine

        class LoopState(Enum):
            RUNNING = auto()
            DONE = auto()

        @dataclass
        class TickEvent:
            pass

        @dataclass
        class LoopContext:
            iteration: int = 0
            max_iterations: int = 3

        machine = (
            StateMachine[LoopState, object, LoopContext]()
            .state(LoopState.RUNNING)
            .state(LoopState.DONE)
            # Continue looping while under max
            .transition(
                LoopState.RUNNING,
                TickEvent,
                LoopState.RUNNING,
                guard=lambda e, c: c.iteration < c.max_iterations - 1,
                action=lambda e, c: setattr(c, "iteration", c.iteration + 1),
            )
            # Stop when at max
            .transition(
                LoopState.RUNNING,
                TickEvent,
                LoopState.DONE,
                guard=lambda e, c: c.iteration >= c.max_iterations - 1,
            )
            .terminal(LoopState.DONE)
        )

        context = LoopContext(iteration=0, max_iterations=3)
        # Keep generating tick events
        final_state, final_ctx = machine.run(
            LoopState.RUNNING,
            context,
            lambda s, c: TickEvent() if s == LoopState.RUNNING else None,
        )

        assert final_state == LoopState.DONE
        assert final_ctx.iteration == 2  # 0 -> 1 -> 2 -> DONE


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestStateMachineEdgeCases:
    """Edge cases and error conditions."""

    def test_no_matching_transition_stays_in_state(self) -> None:
        """When no transition matches, state machine stays in current state."""
        from rlm.domain.models.state_machine import StateMachine

        @dataclass
        class UnknownEvent:
            pass

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.INIT)
            .state(SimpleState.DONE)
            .transition(SimpleState.INIT, StartEvent, SimpleState.DONE)
            # No transition for UnknownEvent
        )

        call_count = 0

        def event_source(s: SimpleState, c: SimpleContext) -> object | None:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return UnknownEvent()  # No transition defined
            return None  # Stop

        final_state, _ = machine.run(SimpleState.INIT, SimpleContext(), event_source)

        # Should stay in INIT since UnknownEvent has no transition
        assert final_state == SimpleState.INIT

    def test_callback_order_on_transition(self) -> None:
        """Callbacks execute in order: on_exit(old) -> action -> on_enter(new)."""
        from rlm.domain.models.state_machine import StateMachine

        order: list[str] = []

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.INIT, on_exit=lambda c: order.append("1_exit_init"))
            .state(SimpleState.DONE, on_enter=lambda c: order.append("3_enter_done"))
            .transition(
                SimpleState.INIT,
                StartEvent,
                SimpleState.DONE,
                action=lambda e, c: order.append("2_action"),
            )
            .terminal(SimpleState.DONE)
        )

        machine.run(
            SimpleState.INIT,
            SimpleContext(),
            lambda s, c: StartEvent() if s == SimpleState.INIT else None,
        )

        assert order == ["1_exit_init", "2_action", "3_enter_done"]

    def test_context_is_same_object_throughout(self) -> None:
        """Context object identity is preserved (mutations are visible)."""
        from rlm.domain.models.state_machine import StateMachine

        original_ctx = SimpleContext()

        machine = (
            StateMachine[SimpleState, object, SimpleContext]()
            .state(SimpleState.INIT)
            .state(SimpleState.DONE)
            .transition(SimpleState.INIT, StartEvent, SimpleState.DONE)
            .terminal(SimpleState.DONE)
        )

        _, final_ctx = machine.run(
            SimpleState.INIT,
            original_ctx,
            lambda s, c: StartEvent() if s == SimpleState.INIT else None,
        )

        assert final_ctx is original_ctx
