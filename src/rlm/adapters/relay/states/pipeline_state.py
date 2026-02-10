from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

from rlm.domain.models.result import Err, Ok
from rlm.domain.relay.errors import StateError
from rlm.domain.relay.ports import StateExecutorPort, StateResult

if TYPE_CHECKING:
    from rlm.domain.relay.baton import Baton
    from rlm.domain.relay.pipeline import Pipeline
    from rlm.domain.relay.state import StateSpec

from rlm.adapters.relay.executors.async_ import AsyncPipelineExecutor
from rlm.adapters.relay.executors.sync import SyncPipelineExecutor

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def _run_sync_pipeline[InputT, OutputT](
    pipeline: Pipeline,
    initial: Baton[InputT],
) -> StateResult[OutputT]:
    executor = SyncPipelineExecutor(pipeline, cast("Baton[object]", initial))
    last_baton: Baton[object] | None = None
    for step in executor:
        state_executor = step.state.executor
        if state_executor is None:
            return Err(
                StateError(
                    error_type="fatal",
                    message=f"State {step.state.name} has no executor.",
                )
            )
        result = state_executor.execute(step.state, step.baton)
        executor.advance(result)
        if isinstance(result, Ok):
            last_baton = result.value
    if executor.failed is not None:
        return Err(executor.failed)
    if last_baton is None:
        return Err(StateError(error_type="fatal", message="Pipeline produced no result."))
    return Ok(cast("Baton[OutputT]", last_baton))


async def _run_async_pipeline[InputT, OutputT](
    pipeline: Pipeline,
    initial: Baton[InputT],
) -> StateResult[OutputT]:
    executor = AsyncPipelineExecutor(pipeline, cast("Baton[object]", initial))
    last_baton: Baton[object] | None = None
    async for step in executor:
        state_executor = step.state.executor
        if state_executor is None:
            return Err(
                StateError(
                    error_type="fatal",
                    message=f"State {step.state.name} has no executor.",
                )
            )
        result = state_executor.execute(step.state, step.baton)
        await executor.advance(result)
        if isinstance(result, Ok):
            last_baton = result.value
    if executor.failed is not None:
        return Err(executor.failed)
    if last_baton is None:
        return Err(StateError(error_type="fatal", message="Pipeline produced no result."))
    return Ok(cast("Baton[OutputT]", last_baton))


@dataclass(frozen=True, slots=True)
class SyncPipelineStateExecutor[InputT, OutputT](StateExecutorPort[InputT, OutputT]):
    pipeline: Pipeline

    def execute(
        self,
        _state: StateSpec[InputT, OutputT],
        baton: Baton[InputT],
        /,
    ) -> StateResult[OutputT]:
        return _run_sync_pipeline(self.pipeline, baton)


@dataclass(frozen=True, slots=True)
class AsyncPipelineStateExecutor[InputT, OutputT](StateExecutorPort[InputT, OutputT]):
    pipeline: Pipeline

    def execute(
        self,
        _state: StateSpec[InputT, OutputT],
        baton: Baton[InputT],
        /,
    ) -> StateResult[OutputT]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                result = cast(
                    "StateResult[OutputT]",
                    asyncio.run(_run_async_pipeline(self.pipeline, baton)),
                )
            except Exception as exc:  # noqa: BLE001 - adapter boundary
                return Err(StateError(error_type="fatal", message=str(exc)))
            return result

        return Err(
            StateError(
                error_type="fatal",
                message="AsyncPipelineStateExecutor cannot run inside an active event loop.",
                retry_hint="Use an async pipeline executor.",
            )
        )
