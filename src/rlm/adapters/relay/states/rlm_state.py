from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from rlm.domain.models import ChatCompletion
from rlm.domain.models.result import Err, Ok
from rlm.domain.relay.baton import Baton
from rlm.domain.relay.errors import ErrorType, StateError
from rlm.domain.relay.ports import StateExecutorPort, StateResult
from rlm.domain.types import Prompt

if TYPE_CHECKING:
    from rlm.adapters.relay.retry import RetryStrategy
    from rlm.domain.models.llm_request import ToolChoice
    from rlm.domain.relay.state import StateSpec
    from rlm.domain.services.rlm_orchestrator import RLMOrchestrator

InputT = TypeVar("InputT")


def _build_output_baton[InputT](
    output: object,
    state: StateSpec[InputT, ChatCompletion],
    source: Baton[InputT],
) -> StateResult[ChatCompletion]:
    result = Baton.create(output, state.output_type, metadata=source.metadata, trace=source.trace)
    if isinstance(result, Err):
        return Err(StateError(error_type="fatal", message=str(result.error)))
    return Ok(result.value)


@dataclass(frozen=True, slots=True)
class RLMStateExecutor(StateExecutorPort[Prompt, ChatCompletion]):
    orchestrator: RLMOrchestrator
    max_iterations: int = 30
    max_depth: int = 1
    root_prompt: str | None = None
    tool_choice: ToolChoice | None = None
    retry: RetryStrategy | None = None

    def execute(
        self,
        state: StateSpec[Prompt, ChatCompletion],
        baton: Baton[Prompt],
        /,
    ) -> StateResult[ChatCompletion]:
        prompt = baton.payload
        attempts = self.retry.max_attempts if self.retry else 1
        for attempt in range(1, attempts + 1):
            try:
                completion = self.orchestrator.completion(
                    prompt,
                    root_prompt=self.root_prompt,
                    max_depth=self.max_depth,
                    max_iterations=self.max_iterations,
                    tool_choice=self.tool_choice,
                )
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
        return Err(StateError(error_type="fatal", message="RLM execution failed without result."))
