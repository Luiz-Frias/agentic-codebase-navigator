from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

from rlm.domain.models import ChatCompletion, LLMRequest
from rlm.domain.models.result import Err, Ok, Result
from rlm.domain.relay.baton import Baton
from rlm.domain.relay.errors import ErrorType, StateError
from rlm.domain.relay.ports import StateExecutorPort, StateResult

if TYPE_CHECKING:
    from rlm.adapters.relay.retry import RetryStrategy
    from rlm.domain.ports import LLMPort
    from rlm.domain.relay.state import StateSpec

InputT = TypeVar("InputT")
InputT_contra = TypeVar("InputT_contra", contravariant=True)
OutputT = TypeVar("OutputT")


class LLMRequestBuilder(Protocol[InputT_contra]):
    def __call__(self, payload: InputT_contra, /) -> LLMRequest: ...


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
class LLMStateExecutor[InputT](StateExecutorPort[InputT, ChatCompletion]):
    llm: LLMPort
    request_builder: LLMRequestBuilder[InputT] | None = None
    retry: RetryStrategy | None = None

    def execute(
        self,
        state: StateSpec[InputT, ChatCompletion],
        baton: Baton[InputT],
        /,
    ) -> StateResult[ChatCompletion]:
        request_result = self._build_request(baton.payload)
        if isinstance(request_result, Err):
            return request_result
        request = request_result.value

        attempts = self.retry.max_attempts if self.retry else 1
        for attempt in range(1, attempts + 1):
            try:
                completion = self.llm.complete(request)
                return _build_output_baton(completion, state, baton)
            except Exception as exc:  # noqa: BLE001 - adapter boundary
                if self.retry and isinstance(exc, self.retry.retry_on) and attempt < attempts:
                    if self.retry.backoff_seconds:
                        time.sleep(self.retry.backoff_seconds)
                    continue
                error_type: ErrorType = (
                    "transient" if self.retry and isinstance(exc, self.retry.retry_on) else "fatal"
                )
                return Err(StateError(error_type=error_type, message=str(exc)))
        return Err(StateError(error_type="fatal", message="LLM execution failed without result."))

    def _build_request(self, payload: InputT) -> Result[LLMRequest, StateError]:
        if self.request_builder is not None:
            try:
                return Ok(self.request_builder(payload))
            except Exception as exc:  # noqa: BLE001 - adapter boundary
                return Err(StateError(error_type="fatal", message=str(exc)))
        if isinstance(payload, LLMRequest):
            return Ok(payload)
        return Err(
            StateError(
                error_type="fatal",
                message="LLMStateExecutor expected payload to be LLMRequest when no builder is provided.",
            )
        )
