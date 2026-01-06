from __future__ import annotations

import pytest

from rlm._legacy.core.types import ModelUsageSummary, UsageSummary
from rlm.api import create_rlm, create_rlm_from_config
from rlm.application.config import EnvironmentConfig, LLMConfig, RLMConfig
from rlm.domain.ports import LLMPort, Prompt


class _DummyLLM:
    def __init__(self) -> None:
        self.model_name = "dummy"
        self._usage = UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(1, 0, 0)})

    def completion(self, prompt: Prompt, /, *, model: str | None = None) -> str:
        return "FINAL(factory_ok)"

    async def acompletion(self, prompt: Prompt, /, *, model: str | None = None) -> str:
        return self.completion(prompt, model=model)

    def get_usage_summary(self):
        return self._usage

    def get_last_usage(self):
        return self._usage


@pytest.mark.unit
def test_create_rlm_factory() -> None:
    llm: LLMPort = _DummyLLM()
    rlm = create_rlm(llm, max_iterations=2, verbose=False)
    assert rlm.completion("hi") == "factory_ok"


@pytest.mark.unit
def test_create_rlm_from_config_requires_llm_in_phase1() -> None:
    cfg = RLMConfig(
        llm=LLMConfig(backend="mock", model_name="dummy"),
        env=EnvironmentConfig(environment="local"),
        max_iterations=2,
        verbose=False,
    )
    with pytest.raises(NotImplementedError):
        create_rlm_from_config(cfg)
