from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from rlm.domain.errors import ValidationError
from rlm.domain.relay.join import JoinSpec
from rlm.domain.relay.pipeline import Pipeline
from rlm.domain.relay.transitions import ConditionalBuilder, ParallelGroup

if TYPE_CHECKING:
    from collections.abc import Callable

    from rlm.domain.relay.baton import Baton
    from rlm.domain.relay.ports import StateExecutorPort

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
NextT = TypeVar("NextT")


@dataclass(frozen=True, slots=True)
class StateSpec[InputT, OutputT]:
    name: str
    input_type: type[InputT]
    output_type: type[OutputT]
    executor: StateExecutorPort[InputT, OutputT] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValidationError("State name must be non-empty.")
        if not isinstance(self.input_type, type):
            raise ValidationError("State input_type must be a type.")
        if not isinstance(self.output_type, type):
            raise ValidationError("State output_type must be a type.")

    def __rshift__(self, other: StateSpec[OutputT, NextT]) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_edge(self, other)
        return pipeline

    def __or__(self, other: StateSpec[InputT, OutputT]) -> ParallelGroup[InputT, OutputT]:
        return ParallelGroup(states=(self, other), join_spec=JoinSpec())

    def when(
        self, predicate: Callable[[Baton[OutputT]], bool]
    ) -> ConditionalBuilder[InputT, OutputT]:
        return ConditionalBuilder(from_state=self, guard=predicate)

    def validate(self) -> None:
        pipeline = Pipeline()
        pipeline.add_state(self)
        pipeline.validate()
