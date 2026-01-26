from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from rlm.domain.errors import ValidationError
from rlm.domain.relay.baton import Baton

if TYPE_CHECKING:
    from rlm.domain.relay.state import StateSpec


type Guard[T] = Callable[[Baton[T]], bool]


@dataclass(frozen=True, slots=True)
class Edge[InputT, OutputT, NextT]:
    from_state: StateSpec[InputT, OutputT]
    to_state: StateSpec[OutputT, NextT]
    guard: Guard[OutputT] | None = None


class Pipeline:
    def __init__(self) -> None:
        self._states: dict[str, StateSpec[object, object]] = {}
        self._edges: list[Edge[object, object, object]] = []
        self._entry_state: StateSpec[object, object] | None = None

    @property
    def entry_state(self) -> StateSpec[object, object] | None:
        return self._entry_state

    @property
    def states(self) -> tuple[StateSpec[object, object], ...]:
        return tuple(self._states.values())

    @property
    def edges(self) -> tuple[Edge[object, object, object], ...]:
        return tuple(self._edges)

    @property
    def terminal_states(self) -> tuple[StateSpec[object, object], ...]:
        outgoing = {edge.from_state.name for edge in self._edges}
        return tuple(state for state in self._states.values() if state.name not in outgoing)

    def add_state[InputT, OutputT](self, state: StateSpec[InputT, OutputT]) -> Pipeline:
        if state.name in self._states:
            return self
        self._states[state.name] = cast("StateSpec[object, object]", state)
        if self._entry_state is None:
            self._entry_state = cast("StateSpec[object, object]", state)
        return self

    def add_edge[InputT, OutputT, NextT](
        self,
        from_state: StateSpec[InputT, OutputT],
        to_state: StateSpec[OutputT, NextT],
        *,
        guard: Guard[OutputT] | None = None,
    ) -> Pipeline:
        self.add_state(from_state)
        self.add_state(to_state)
        self._edges.append(
            cast(
                "Edge[object, object, object]",
                Edge(
                    from_state=cast("StateSpec[object, object]", from_state),
                    to_state=cast("StateSpec[object, object]", to_state),
                    guard=cast("Guard[object]", guard),
                ),
            )
        )
        return self


class ConditionalPipeline[InputT, OutputT, NextT](Pipeline):
    def __init__(
        self,
        from_state: StateSpec[InputT, OutputT],
        primary_state: StateSpec[OutputT, NextT],
        guard: Guard[OutputT],
    ) -> None:
        super().__init__()
        self._from_state = from_state
        self._primary_state = primary_state
        self._guard = guard
        self.add_edge(from_state, primary_state, guard=guard)

    def otherwise[DefaultT](
        self,
        default_state: StateSpec[OutputT, DefaultT],
    ) -> Pipeline:
        if default_state.name == self._primary_state.name:
            raise ValidationError("Default state must differ from conditional target.")
        self.add_edge(self._from_state, default_state)
        return self
