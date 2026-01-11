from __future__ import annotations

from abc import ABC, abstractmethod

from rlm.domain.models import (
    BatchedLLMRequest,
    ChatCompletion,
    Iteration,
    LLMRequest,
    ReplResult,
    RunMetadata,
    UsageSummary,
)
from rlm.domain.ports import LLMPort
from rlm.domain.types import ContextPayload


class BaseLLMAdapter(ABC):
    """Optional ABC base for adapters implementing `LLMPort`."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def complete(self, request: LLMRequest, /) -> ChatCompletion: ...

    @abstractmethod
    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion: ...

    @abstractmethod
    def get_usage_summary(self) -> UsageSummary: ...

    @abstractmethod
    def get_last_usage(self) -> UsageSummary: ...


class BaseBrokerAdapter(ABC):
    """Optional ABC base for adapters implementing `BrokerPort`."""

    @abstractmethod
    def register_llm(self, model_name: str, llm: LLMPort, /) -> None: ...

    @abstractmethod
    def start(self) -> tuple[str, int]: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def complete(self, request: LLMRequest, /) -> ChatCompletion: ...

    @abstractmethod
    def complete_batched(self, request: BatchedLLMRequest, /) -> list[ChatCompletion]: ...

    @abstractmethod
    def get_usage_summary(self) -> UsageSummary: ...


class BaseEnvironmentAdapter(ABC):
    """Optional ABC base for adapters implementing `EnvironmentPort`."""

    @abstractmethod
    def load_context(self, context_payload: ContextPayload, /) -> None: ...

    @abstractmethod
    def execute_code(self, code: str, /) -> ReplResult: ...

    @abstractmethod
    def cleanup(self) -> None: ...


class BaseLoggerAdapter(ABC):
    """Optional ABC base for adapters implementing `LoggerPort`."""

    __slots__ = ()

    @abstractmethod
    def log_metadata(self, metadata: RunMetadata, /) -> None: ...

    @abstractmethod
    def log_iteration(self, iteration: Iteration, /) -> None: ...
