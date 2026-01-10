from __future__ import annotations

import pytest

from rlm.api.rlm import RLM
from rlm.domain.models import ChatCompletion, LLMRequest, ModelUsageSummary, UsageSummary
from rlm.domain.ports import LLMPort


class _DummyLLM:
    # TODO(phase4/phase5): Replace/augment this dummy with a real provider adapter call
    # (e.g. OpenAIAdapter hitting an OpenAI-compatible endpoint) in an opt-in test.
    # Keep unit tests hermetic by default.
    def __init__(self) -> None:
        self.model_name = "dummy"
        self._usage = UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(1, 0, 0)})

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        return ChatCompletion(
            root_model=request.model or self.model_name,
            prompt=request.prompt,
            response="FINAL(ok)",
            usage_summary=self._usage,
            execution_time=0.0,
        )

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        return self.complete(request)

    def get_usage_summary(self):
        return self._usage

    def get_last_usage(self):
        return self._usage


@pytest.mark.unit
def test_api_rlm_completion_returns_final_answer() -> None:
    llm: LLMPort = _DummyLLM()
    rlm = RLM(llm, max_iterations=2, verbose=False)
    cc = rlm.completion("hello")
    assert cc.response == "ok"
