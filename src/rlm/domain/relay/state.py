from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from rlm.domain.errors import ValidationError

if TYPE_CHECKING:
    from rlm.domain.relay.ports import StateExecutorPort

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


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
