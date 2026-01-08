from __future__ import annotations

import time

from rlm._legacy.core.lm_handler import LMHandler
from rlm.adapters.base import BaseBrokerAdapter
from rlm.adapters.legacy.llm import _as_legacy_client
from rlm.adapters.legacy.mappers import legacy_usage_summary_to_domain
from rlm.domain.models import BatchedLLMRequest, ChatCompletion, LLMRequest, UsageSummary
from rlm.domain.ports import LLMPort


class LegacyBrokerAdapter(BaseBrokerAdapter):
    """Adapter: legacy `LMHandler` -> domain `BrokerPort`."""

    def __init__(
        self,
        default_llm: LLMPort,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        self._handler = LMHandler(_as_legacy_client(default_llm), host=host, port=port)

    def register_llm(self, model_name: str, llm: LLMPort, /) -> None:
        self._handler.register_client(model_name, _as_legacy_client(llm, model_name=model_name))

    def start(self) -> tuple[str, int]:
        return self._handler.start()

    def stop(self) -> None:
        self._handler.stop()

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        client = self._handler.get_client(request.model)
        start = time.perf_counter()
        content = client.completion(request.prompt)  # type: ignore[arg-type]
        end = time.perf_counter()
        usage = legacy_usage_summary_to_domain(client.get_last_usage())
        return ChatCompletion(
            root_model=request.model or client.model_name,
            prompt=request.prompt,
            response=content,
            usage_summary=usage,
            execution_time=end - start,
        )

    def complete_batched(self, request: BatchedLLMRequest, /) -> list[ChatCompletion]:
        # Execute sequentially to preserve correct per-prompt usage semantics.
        return [self.complete(LLMRequest(prompt=p, model=request.model)) for p in request.prompts]

    def get_usage_summary(self) -> UsageSummary:
        return legacy_usage_summary_to_domain(self._handler.get_usage_summary())
