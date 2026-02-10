from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from rlm.domain.errors import ValidationError
from rlm.domain.models.result import Err, Ok
from rlm.domain.relay.baton import Baton
from rlm.domain.relay.errors import ErrorType, StateError
from rlm.domain.relay.ports import StateExecutorPort, StateResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from rlm.adapters.relay.retry import RetryStrategy
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
class FunctionStateExecutor[InputT, OutputT](StateExecutorPort[InputT, OutputT]):
    fn: Callable[[InputT], OutputT]
    retry: RetryStrategy | None = None

    def execute(
        self,
        state: StateSpec[InputT, OutputT],
        baton: Baton[InputT],
        /,
    ) -> StateResult[OutputT]:
        attempts = self.retry.max_attempts if self.retry else 1
        for attempt in range(1, attempts + 1):
            try:
                output = self.fn(baton.payload)
                return _build_output_baton(
                    output=output,
                    state=state,
                    source=baton,
                )
            except ValidationError as exc:
                return Err(StateError(error_type="fatal", message=str(exc)))
            except Exception as exc:  # noqa: BLE001 - adapter boundary
                if self.retry and isinstance(exc, self.retry.retry_on) and attempt < attempts:
                    if self.retry.backoff_seconds:
                        time.sleep(self.retry.backoff_seconds)
                    continue
                error_type: ErrorType = (
                    "transient" if self.retry and isinstance(exc, self.retry.retry_on) else "fatal"
                )
                return Err(StateError(error_type=error_type, message=str(exc)))
        return Err(StateError(error_type="fatal", message="Execution failed without result."))
