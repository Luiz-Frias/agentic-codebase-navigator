from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rlm.domain.agent_ports import ToolDefinition, ToolPort
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

    @property
    def tool_prompt_format(self) -> str:
        """Tool calling prompt format expected by the adapter."""
        return "openai"

    @property
    def supports_tools(self) -> bool:
        """Whether the adapter supports native tool calling."""
        return False

    def count_prompt_tokens(self, request: LLMRequest, /) -> int | None:
        """Return prompt token count if the adapter can compute it."""
        return None

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


# -----------------------------------------------------------------------------
# Agent capability adapters (Phase 1 - Core)
# -----------------------------------------------------------------------------


class BaseToolAdapter(ABC):
    """Optional ABC base for adapters implementing `ToolPort`."""

    __slots__ = ()

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition: ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any: ...

    @abstractmethod
    async def aexecute(self, **kwargs: Any) -> Any: ...


class BaseToolRegistryAdapter(ABC):
    """Optional ABC base for adapters implementing `ToolRegistryPort`."""

    __slots__ = ()

    @abstractmethod
    def register(self, tool: ToolPort, /) -> None: ...

    @abstractmethod
    def get(self, name: str, /) -> ToolPort | None: ...

    @abstractmethod
    def list_definitions(self) -> list[ToolDefinition]: ...


class BaseStructuredOutputAdapter[T](ABC):
    """Optional ABC base for adapters implementing `StructuredOutputPort`."""

    __slots__ = ()

    @abstractmethod
    def validate(self, response: str, output_type: type[T], /) -> T: ...

    @abstractmethod
    def get_schema(self, output_type: type[T], /) -> dict[str, Any]: ...
