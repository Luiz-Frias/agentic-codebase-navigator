from __future__ import annotations

from rlm.domain.errors import ValidationError
from rlm.domain.models import (
    BatchedLLMRequest,
    ChatCompletion,
    Iteration,
    LLMRequest,
    ModelSpec,
    ModelUsageSummary,
    ReplResult,
    RunMetadata,
    UsageSummary,
    build_routing_rules,
)
from rlm.domain.models.usage import merge_usage_summaries
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
        self._routing_rules = build_routing_rules(
            [ModelSpec(name=default_llm.model_name, is_default=True)]
        )
        self._started = False

    def register_llm(self, model_name: str, llm: LLMPort, /) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValidationError("Broker.register_llm requires a non-empty model_name")
        if model_name != llm.model_name:
            raise ValidationError(
                f"Broker.register_llm model_name {model_name!r} must match llm.model_name {llm.model_name!r}"
            )
        self._llms[model_name] = llm
        default = self._default.model_name
        specs = [ModelSpec(name=default, is_default=True)]
        for name in sorted(self._llms):
            if name == default:
                continue
            specs.append(ModelSpec(name=name))
        self._routing_rules = build_routing_rules(specs)

    def start(self) -> tuple[str, int]:
        self._started = True
        return ("127.0.0.1", 0)

    def stop(self) -> None:
        self._started = False

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        resolved = self._routing_rules.resolve(request.model)
        llm = self._llms[resolved]
        # `request.model` is a routing hint; call the selected adapter with its own
        # model name to keep `root_model` consistent.
        return llm.complete(LLMRequest(prompt=request.prompt, model=llm.model_name))

    def complete_batched(self, request: BatchedLLMRequest, /) -> list[ChatCompletion]:
        resolved = self._routing_rules.resolve(request.model)
        llm = self._llms[resolved]
        return [llm.complete(LLMRequest(prompt=p, model=llm.model_name)) for p in request.prompts]

    def get_usage_summary(self) -> UsageSummary:
        return merge_usage_summaries(llm.get_usage_summary() for llm in self._llms.values())
