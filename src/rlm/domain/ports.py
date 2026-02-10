from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rlm.domain.models import (
        BatchedLLMRequest,
        ChatCompletion,
        Iteration,
        LLMRequest,
        NestedCallResponse,
        ReplResult,
        RunMetadata,
        UsageSummary,
    )
    from rlm.domain.types import ContextPayload, Prompt


class LLMPort(Protocol):
    """Port for an LLM provider/client."""

    @property
    def model_name(self) -> str: ...

    def complete(self, request: LLMRequest, /) -> ChatCompletion: ...

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion: ...

    def get_usage_summary(self) -> UsageSummary: ...

    def get_last_usage(self) -> UsageSummary: ...


class BrokerPort(Protocol):
    """Port for routing LLM requests (single + batched) to registered models."""

    def register_llm(self, model_name: str, llm: LLMPort, /) -> None: ...

    def start(self) -> tuple[str, int]: ...

    def stop(self) -> None: ...

    def complete(self, request: LLMRequest, /) -> ChatCompletion: ...

    def complete_batched(self, request: BatchedLLMRequest, /) -> list[ChatCompletion]: ...

    def get_usage_summary(self) -> UsageSummary: ...


class EnvironmentPort(Protocol):
    """Port for an execution environment (local/docker/etc)."""

    def load_context(self, context_payload: ContextPayload, /) -> None: ...

    def execute_code(self, code: str, /) -> ReplResult: ...

    def cleanup(self) -> None: ...


class LoggerPort(Protocol):
    """Port for capturing execution metadata/iterations/artifacts."""

    def log_metadata(self, metadata: RunMetadata, /) -> None: ...

    def log_iteration(self, iteration: Iteration, /) -> None: ...


class ClockPort(Protocol):
    """Port for time measurement (injectable for deterministic tests)."""

    def now(self) -> float: ...


class IdGeneratorPort(Protocol):
    """Port for generating correlation/run IDs (injectable for deterministic tests)."""

    def new_id(self) -> str: ...


@runtime_checkable
class NestedCallHandlerPort(Protocol):
    """Port for handling nested llm_query() calls (optional)."""

    def handle(
        self,
        prompt: Prompt,
        /,
        *,
        depth: int,
        correlation_id: str | None,
        model: str | None,
    ) -> NestedCallResponse: ...
