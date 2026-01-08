from __future__ import annotations

from rlm.domain.models import (
    BatchedLLMRequest,
    ChatCompletion,
    Iteration,
    LLMRequest,
    ModelUsageSummary,
    ReplResult,
    RunMetadata,
    UsageSummary,
)
from rlm.domain.ports import (
    BrokerPort,
    ClockPort,
    EnvironmentPort,
    IdGeneratorPort,
    LLMPort,
    LoggerPort,
)
from rlm.domain.types import ContextPayload


class FakeClock(ClockPort):
    def __init__(self, *, start: float = 0.0, step: float = 0.0) -> None:
        self._t = start
        self._step = step

    def now(self) -> float:
        self._t += self._step
        return self._t


class SequenceIdGenerator(IdGeneratorPort):
    def __init__(self, *, prefix: str = "id", start: int = 1) -> None:
        self._prefix = prefix
        self._next = start

    def new_id(self) -> str:
        out = f"{self._prefix}-{self._next}"
        self._next += 1
        return out


class QueueLLM(LLMPort):
    """
    Deterministic LLMPort for tests.

    Provide a sequence of responses (strings) or exceptions. Each call pops one.
    """

    def __init__(self, *, model_name: str = "mock", responses: list[str | Exception] | None = None):
        self._model_name = model_name
        self._queue: list[str | Exception] = list(responses or [])
        self._total_calls = 0
        self._last_usage = UsageSummary(model_usage_summaries={})

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        self._total_calls += 1
        if not self._queue:
            raise AssertionError("QueueLLM: no scripted responses left")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item

        self._last_usage = UsageSummary(
            model_usage_summaries={
                self._model_name: ModelUsageSummary(
                    total_calls=1, total_input_tokens=0, total_output_tokens=0
                )
            }
        )
        return ChatCompletion(
            root_model=request.model or self._model_name,
            prompt=request.prompt,
            response=item,
            usage_summary=self._last_usage,
            execution_time=0.0,
        )

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        return self.complete(request)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                self._model_name: ModelUsageSummary(
                    total_calls=self._total_calls, total_input_tokens=0, total_output_tokens=0
                )
            }
        )

    def get_last_usage(self) -> UsageSummary:
        return self._last_usage


class QueueEnvironment(EnvironmentPort):
    """Deterministic EnvironmentPort for tests."""

    def __init__(self, *, results: list[ReplResult | Exception] | None = None) -> None:
        self._results: list[ReplResult | Exception] = list(results or [])
        self.loaded_contexts: list[ContextPayload] = []
        self.executed_code: list[str] = []

    def load_context(self, context_payload: ContextPayload, /) -> None:
        self.loaded_contexts.append(context_payload)

    def execute_code(self, code: str, /) -> ReplResult:
        self.executed_code.append(code)
        if not self._results:
            raise AssertionError("QueueEnvironment: no scripted results left")
        item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def cleanup(self) -> None:
        return None


class CollectingLogger(LoggerPort):
    def __init__(self) -> None:
        self.metadata: list[RunMetadata] = []
        self.iterations: list[Iteration] = []

    def log_metadata(self, metadata: RunMetadata, /) -> None:
        self.metadata.append(metadata)

    def log_iteration(self, iteration: Iteration, /) -> None:
        self.iterations.append(iteration)


class InMemoryBroker(BrokerPort):
    """Simple broker that routes to registered LLMPort instances."""

    def __init__(self, default_llm: LLMPort) -> None:
        self._llms: dict[str, LLMPort] = {default_llm.model_name: default_llm}
        self._default = default_llm
        self._started = False

    def register_llm(self, model_name: str, llm: LLMPort, /) -> None:
        self._llms[model_name] = llm

    def start(self) -> tuple[str, int]:
        self._started = True
        return ("127.0.0.1", 0)

    def stop(self) -> None:
        self._started = False

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        llm = self._llms.get(request.model or "", self._default)
        return llm.complete(request)

    def complete_batched(self, request: BatchedLLMRequest, /) -> list[ChatCompletion]:
        return [self.complete(LLMRequest(prompt=p, model=request.model)) for p in request.prompts]

    def get_usage_summary(self) -> UsageSummary:
        merged: dict[str, ModelUsageSummary] = {}
        for _name, llm in self._llms.items():
            summary = llm.get_usage_summary()
            for model, mus in summary.model_usage_summaries.items():
                merged[model] = mus
        return UsageSummary(model_usage_summaries=merged)
