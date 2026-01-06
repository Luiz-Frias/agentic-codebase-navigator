from __future__ import annotations

from typing import Any, Protocol

# -----------------------------------------------------------------------------
# Shared type aliases (v0)
# -----------------------------------------------------------------------------

# NOTE: These are intentionally broad in Phase 1 to keep the legacy adapter path
# flexible while we migrate to explicit domain models in Phase 2+.
Prompt = str | dict[str, Any] | list[dict[str, Any]]
ContextPayload = dict[str, Any] | list[Any] | str


class LLMPort(Protocol):
    """Port for an LLM provider/client."""

    @property
    def model_name(self) -> str: ...

    def completion(self, prompt: Prompt, /, *, model: str | None = None) -> str: ...

    async def acompletion(self, prompt: Prompt, /, *, model: str | None = None) -> str: ...

    def get_usage_summary(self) -> Any: ...

    def get_last_usage(self) -> Any: ...


class BrokerPort(Protocol):
    """Port for routing LLM requests (single + batched) to registered models."""

    def register_llm(self, model_name: str, llm: LLMPort, /) -> None: ...

    def start(self) -> tuple[str, int]: ...

    def stop(self) -> None: ...

    def completion(self, prompt: Prompt, /, *, model: str | None = None) -> str: ...

    def get_usage_summary(self) -> Any: ...


class EnvironmentPort(Protocol):
    """Port for an execution environment (local/docker/etc)."""

    def load_context(self, context_payload: ContextPayload, /) -> None: ...

    def execute_code(self, code: str, /) -> Any: ...

    def cleanup(self) -> None: ...


class LoggerPort(Protocol):
    """Port for capturing execution metadata/iterations/artifacts."""

    def log_metadata(self, metadata: Any, /) -> None: ...

    def log_iteration(self, iteration: Any, /) -> None: ...
