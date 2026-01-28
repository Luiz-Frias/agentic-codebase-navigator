from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from rlm.domain.relay.join import JoinMode, JoinSpec
from rlm.domain.relay.pipeline import ConditionalPipeline, Guard, Pipeline

if TYPE_CHECKING:
    from rlm.domain.relay.state import StateSpec

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
NextT = TypeVar("NextT")


@dataclass(frozen=True, slots=True)
class ConditionalBuilder[InputT, OutputT]:
    from_state: StateSpec[InputT, OutputT]
    guard: Guard[OutputT]

    def __rshift__[NextT](
        self,
        to_state: StateSpec[OutputT, NextT],
    ) -> ConditionalPipeline[InputT, OutputT, NextT]:
        return ConditionalPipeline(self.from_state, to_state, self.guard)


@dataclass(frozen=True, slots=True)
class ParallelGroup[InputT, OutputT]:
    states: tuple[StateSpec[InputT, OutputT], ...]
    join_spec: JoinSpec

    def __or__(self, other: StateSpec[InputT, OutputT]) -> ParallelGroup[InputT, OutputT]:
        return ParallelGroup(states=(*self.states, other), join_spec=self.join_spec)

    def join(
        self,
        *,
        mode: JoinMode = "all",
        timeout_seconds: float | None = None,
    ) -> ParallelGroup[InputT, OutputT]:
        join_spec = (
            JoinSpec(mode=mode)
            if timeout_seconds is None
            else JoinSpec(mode=mode, timeout_seconds=timeout_seconds)
        )
        return ParallelGroup(
            states=self.states,
            join_spec=join_spec,
        )

    def __rshift__[NextT](self, other: StateSpec[OutputT, NextT]) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_join_group(self.states, other, self.join_spec)
        return pipeline
