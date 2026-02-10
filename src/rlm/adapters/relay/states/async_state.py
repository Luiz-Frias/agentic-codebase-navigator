from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from rlm.domain.models.result import Err, Ok
from rlm.domain.relay.baton import Baton
from rlm.domain.relay.errors import StateError
from rlm.domain.relay.ports import StateExecutorPort, StateResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from rlm.domain.relay.state import StateSpec

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def _build_output_baton[InputT, OutputT](
    output: object,
    state: StateSpec[InputT, OutputT],
    source: Baton[InputT],
) -> StateResult[OutputT]:
    result = Baton.create(output, state.output_type, metadata=source.metadata, trace=source.trace)
    if isinstance(result, Err):
        return Err(StateError(error_type="fatal", message=str(result.error)))
    return Ok(result.value)


@dataclass(frozen=True, slots=True)
class AsyncStateExecutor[InputT, OutputT](StateExecutorPort[InputT, OutputT]):
    fn: Callable[[InputT], Awaitable[OutputT]]

    def execute(
        self,
        state: StateSpec[InputT, OutputT],
        baton: Baton[InputT],
        /,
    ) -> StateResult[OutputT]:
        async def _await(awaitable: Awaitable[OutputT]) -> OutputT:
            return await awaitable

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                output: OutputT = asyncio.run(_await(self.fn(baton.payload)))
            except Exception as exc:  # noqa: BLE001 - adapter boundary
                return Err(StateError(error_type="fatal", message=str(exc)))
            return _build_output_baton(output, state, baton)

        return Err(
            StateError(
                error_type="fatal",
                message="AsyncStateExecutor cannot run inside an active event loop.",
                retry_hint="Use an async pipeline executor.",
            )
        )
