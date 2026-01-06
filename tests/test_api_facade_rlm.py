from __future__ import annotations

import pytest

from rlm._legacy.core.types import ModelUsageSummary, UsageSummary
from rlm.api.rlm import RLM
from rlm.domain.ports import LLMPort, Prompt


class _DummyLLM:
    def __init__(self) -> None:
        self.model_name = "dummy"
        self._usage = UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(1, 0, 0)})

    def completion(self, prompt: Prompt, /, *, model: str | None = None) -> str:
        return "FINAL(ok)"

    async def acompletion(self, prompt: Prompt, /, *, model: str | None = None) -> str:
        return self.completion(prompt, model=model)

    def get_usage_summary(self):
        return self._usage

    def get_last_usage(self):
        return self._usage


@pytest.mark.unit
def test_api_rlm_completion_returns_final_answer() -> None:
    llm: LLMPort = _DummyLLM()
    rlm = RLM(llm, max_iterations=2, verbose=False)
    assert rlm.completion("hello") == "ok"
