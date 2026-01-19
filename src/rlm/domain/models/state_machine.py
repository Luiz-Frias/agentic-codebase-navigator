"""
Declarative state machine for complex control flow.

This module provides a StateMachine abstraction that replaces complex nested loops
(C901) with declarative state transitions. It integrates with TypeMapper for event
dispatch and supports both synchronous and asynchronous execution.

Example:
    machine = (
        StateMachine[State, Event, Context]()
        .state(State.INIT, on_enter=load_context)
        .state(State.RUNNING)
        .state(State.DONE)
        .transition(State.INIT, StartEvent, State.RUNNING)
        .transition(State.RUNNING, StopEvent, State.DONE)
        .terminal(State.DONE)
    )

    final_state, final_ctx = machine.run(State.INIT, context, event_source)

Design notes:
- States are registered with optional on_enter/on_exit callbacks
- Transitions are triggered by event types, with optional guards and actions
- Guards receive (event, context) and return bool
- Actions receive (event, context) and perform side effects
- Callback order: on_exit(old) -> action -> on_enter(new)
- Context is mutated in place (same object identity throughout)

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

S = TypeVar("S")  # State type
E = TypeVar("E")  # Event type
C = TypeVar("C")  # Context type


@dataclass
class StateConfig[S, C]:
    """Configuration for a single state."""

    name: S
    on_enter: Callable[[C], None] | None = None
    on_exit: Callable[[C], None] | None = None


@dataclass
class Transition[S, E, C]:
    """Configuration for a state transition."""

    from_state: S
    event_type: type[E]
    to_state: S
    guard: Callable[[E, C], bool] | None = None
    action: Callable[[E, C], None] | None = None


@dataclass
class StateMachine[S, E, C]:
    """
    Declarative state machine with typed states, events, and context.

    S = State type (typically an Enum)
    E = Event type (union of event dataclasses)
    C = Context type (mutable state passed through transitions)

    Thread safety:
    - NOT thread-safe during configuration (build at startup)
    - Thread-safe during run() if context mutations are thread-safe
    """

    _states: dict[S, StateConfig[S, C]] = field(default_factory=dict)
    _transitions: list[Transition[S, E, C]] = field(default_factory=list)
    _terminal_states: set[S] = field(default_factory=set)

    def state(
        self,
        name: S,
        on_enter: Callable[[C], None] | None = None,
        on_exit: Callable[[C], None] | None = None,
    ) -> StateMachine[S, E, C]:
        """
        Register a state with optional entry/exit callbacks.

        Args:
            name: The state identifier (typically an enum value)
            on_enter: Callback invoked when entering this state
            on_exit: Callback invoked when leaving this state

        Returns:
            Self for method chaining

        """
        self._states[name] = StateConfig(name=name, on_enter=on_enter, on_exit=on_exit)
        return self

    def transition(
        self,
        from_state: S,
        event_type: type[E],
        to_state: S,
        guard: Callable[[E, C], bool] | None = None,
        action: Callable[[E, C], None] | None = None,
    ) -> StateMachine[S, E, C]:
        """
        Register a transition triggered by an event type.

        Args:
            from_state: Source state for this transition
            event_type: Event class that triggers this transition
            to_state: Target state after transition
            guard: Optional predicate (event, context) -> bool. Transition only
                   occurs if guard returns True.
            action: Optional callback (event, context) -> None executed during
                    transition (after on_exit, before on_enter).

        Returns:
            Self for method chaining

        """
        self._transitions.append(
            Transition(
                from_state=from_state,
                event_type=event_type,
                to_state=to_state,
                guard=guard,
                action=action,
            ),
        )
        return self

    def terminal(self, *states: S) -> StateMachine[S, E, C]:
        """
        Mark states as terminal (machine stops when reached).

        Args:
            *states: One or more states that cause execution to stop

        Returns:
            Self for method chaining

        """
        self._terminal_states.update(states)
        return self

    def _find_transition(
        self,
        current_state: S,
        event: E,
        context: C,
    ) -> Transition[S, E, C] | None:
        """
        Find the first matching transition for an event.

        Transitions are checked in registration order. A transition matches if:
        1. from_state matches current_state
        2. event_type matches the event's type (isinstance check)
        3. guard (if any) returns True

        Args:
            current_state: The current state
            event: The event instance
            context: The current context

        Returns:
            The first matching Transition, or None if no match

        """
        for trans in self._transitions:
            if trans.from_state != current_state:
                continue
            if not isinstance(event, trans.event_type):
                continue
            if trans.guard is not None and not trans.guard(event, context):
                continue
            return trans
        return None

    def _execute_transition(
        self,
        trans: Transition[S, E, C],
        event: E,
        context: C,
    ) -> S:
        """
        Execute a transition: on_exit -> action -> on_enter.

        Args:
            trans: The transition to execute
            event: The triggering event
            context: The mutable context

        Returns:
            The new state after transition

        """
        # Execute on_exit for current state
        current_config = self._states.get(trans.from_state)
        if current_config and current_config.on_exit:
            current_config.on_exit(context)

        # Execute transition action
        if trans.action:
            trans.action(event, context)

        # Execute on_enter for new state
        new_config = self._states.get(trans.to_state)
        if new_config and new_config.on_enter:
            new_config.on_enter(context)

        return trans.to_state

    def run(
        self,
        initial_state: S,
        context: C,
        event_source: Callable[[S, C], E | None],
    ) -> tuple[S, C]:
        """
        Execute state machine synchronously until terminal state or no events.

        Args:
            initial_state: The starting state
            context: Mutable context passed to all callbacks
            event_source: Function (state, context) -> event | None.
                          Returns None to signal no more events.

        Returns:
            Tuple of (final_state, context). Context is the same object
            passed in, potentially mutated.

        """
        current_state = initial_state

        # Execute on_enter for initial state
        initial_config = self._states.get(current_state)
        if initial_config and initial_config.on_enter:
            initial_config.on_enter(context)

        # Main execution loop
        while current_state not in self._terminal_states:
            event = event_source(current_state, context)
            if event is None:
                break

            trans = self._find_transition(current_state, event, context)
            if trans is None:
                # No matching transition - stay in current state but continue
                continue

            current_state = self._execute_transition(trans, event, context)

        return current_state, context

    async def arun(
        self,
        initial_state: S,
        context: C,
        event_source: Callable[[S, C], Awaitable[E | None]],
    ) -> tuple[S, C]:
        """
        Execute state machine asynchronously until terminal state or no events.

        Args:
            initial_state: The starting state
            context: Mutable context passed to all callbacks
            event_source: Async function (state, context) -> event | None.
                          Returns None to signal no more events.

        Returns:
            Tuple of (final_state, context). Context is the same object
            passed in, potentially mutated.

        """
        current_state = initial_state

        # Execute on_enter for initial state
        initial_config = self._states.get(current_state)
        if initial_config and initial_config.on_enter:
            initial_config.on_enter(context)

        # Main execution loop
        while current_state not in self._terminal_states:
            event = await event_source(current_state, context)
            if event is None:
                break

            trans = self._find_transition(current_state, event, context)
            if trans is None:
                # No matching transition - stay in current state but continue
                continue

            current_state = self._execute_transition(trans, event, context)

        return current_state, context
