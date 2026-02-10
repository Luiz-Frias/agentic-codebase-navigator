from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from rlm.domain.relay.baton import Baton
    from rlm.domain.relay.state import StateSpec

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True)
class PipelineStep[InputT, OutputT]:
    state: StateSpec[InputT, OutputT]
    baton: Baton[InputT]
    next_states: tuple[StateSpec[OutputT, object], ...] = ()
