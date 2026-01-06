from __future__ import annotations

from typing import Any

from rlm._legacy.clients.base_lm import BaseLM
from rlm.domain.ports import LLMPort, Prompt


class LegacyLLMPortAdapter:
    """
    Adapter: legacy `BaseLM` -> domain `LLMPort`.

    Note: In Phase 1, provider clients are intentionally not implemented by
    default; tests can inject/monkeypatch them.
    """

    def __init__(self, client: BaseLM):
        self._client = client

    @property
    def model_name(self) -> str:
        return self._client.model_name

    def completion(self, prompt: Prompt, /, *, model: str | None = None) -> str:
        # `model` selection is broker-level; ignore here.
        return self._client.completion(prompt)  # type: ignore[arg-type]

    async def acompletion(self, prompt: Prompt, /, *, model: str | None = None) -> str:
        return await self._client.acompletion(prompt)  # type: ignore[arg-type]

    def get_usage_summary(self) -> Any:
        return self._client.get_usage_summary()

    def get_last_usage(self) -> Any:
        return self._client.get_last_usage()


class _PortBackedLegacyClient(BaseLM):
    """Internal helper: domain `LLMPort` -> legacy `BaseLM`."""

    def __init__(self, llm: LLMPort, *, model_name: str):
        super().__init__(model_name=model_name)
        self._llm = llm

    def completion(self, prompt: str | dict[str, Any]) -> str:
        return self._llm.completion(prompt)  # type: ignore[arg-type]

    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        return await self._llm.acompletion(prompt)  # type: ignore[arg-type]

    def get_usage_summary(self):
        return self._llm.get_usage_summary()

    def get_last_usage(self):
        return self._llm.get_last_usage()


def _as_legacy_client(llm: LLMPort) -> BaseLM:
    """
    Convert an `LLMPort` to a legacy `BaseLM` for use with `LMHandler`.
    """
    return _PortBackedLegacyClient(llm, model_name=llm.model_name)
