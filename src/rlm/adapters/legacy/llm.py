from __future__ import annotations

import time

from rlm._legacy.clients.base_lm import BaseLM
from rlm.adapters.base import BaseLLMAdapter
from rlm.adapters.legacy.mappers import (
    domain_usage_summary_to_legacy,
    legacy_usage_summary_to_domain,
)
from rlm.domain.models import ChatCompletion, LLMRequest, UsageSummary
from rlm.domain.ports import LLMPort
from rlm.domain.types import Prompt


class LegacyLLMPortAdapter(BaseLLMAdapter):
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

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        start = time.perf_counter()
        content = self._client.completion(request.prompt)  # type: ignore[arg-type]
        end = time.perf_counter()
        usage = legacy_usage_summary_to_domain(self._client.get_last_usage())
        return ChatCompletion(
            root_model=request.model or self._client.model_name,
            prompt=request.prompt,
            response=content,
            usage_summary=usage,
            execution_time=end - start,
        )

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        start = time.perf_counter()
        content = await self._client.acompletion(request.prompt)  # type: ignore[arg-type]
        end = time.perf_counter()
        usage = legacy_usage_summary_to_domain(self._client.get_last_usage())
        return ChatCompletion(
            root_model=request.model or self._client.model_name,
            prompt=request.prompt,
            response=content,
            usage_summary=usage,
            execution_time=end - start,
        )

    def get_usage_summary(self) -> UsageSummary:
        return legacy_usage_summary_to_domain(self._client.get_usage_summary())

    def get_last_usage(self) -> UsageSummary:
        return legacy_usage_summary_to_domain(self._client.get_last_usage())


class _PortBackedLegacyClient(BaseLM):
    """Internal helper: domain `LLMPort` -> legacy `BaseLM`."""

    def __init__(self, llm: LLMPort, *, model_name: str):
        super().__init__(model_name=model_name)
        self._llm = llm

    def completion(self, prompt: Prompt) -> str:  # type: ignore[override]
        cc = self._llm.complete(LLMRequest(prompt=prompt))
        return cc.response

    async def acompletion(self, prompt: Prompt) -> str:  # type: ignore[override]
        cc = await self._llm.acomplete(LLMRequest(prompt=prompt))
        return cc.response

    def get_usage_summary(self):
        return domain_usage_summary_to_legacy(self._llm.get_usage_summary())

    def get_last_usage(self):
        return domain_usage_summary_to_legacy(self._llm.get_last_usage())


def _as_legacy_client(llm: LLMPort, *, model_name: str | None = None) -> BaseLM:
    """
    Convert an `LLMPort` to a legacy `BaseLM` for use with `LMHandler`.
    """
    return _PortBackedLegacyClient(llm, model_name=model_name or llm.model_name)
