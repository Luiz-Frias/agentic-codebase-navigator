from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

from rlm.domain.models.result import Result
from rlm.domain.relay.baton import Baton
from rlm.domain.relay.errors import StateError

if TYPE_CHECKING:
    from rlm.domain.relay.state import StateSpec

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


type StateResult[T] = Result[Baton[T], StateError]


class StateExecutorPort(Protocol[InputT, OutputT]):
    def execute(
        self,
        state: StateSpec[InputT, OutputT],
        baton: Baton[InputT],
        /,
    ) -> StateResult[OutputT]: ...
