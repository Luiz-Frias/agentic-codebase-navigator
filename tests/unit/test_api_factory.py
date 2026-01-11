from __future__ import annotations

import pytest

from rlm.api import create_rlm, create_rlm_from_config
from rlm.application.config import EnvironmentConfig, LLMConfig, RLMConfig
from rlm.domain.models import (
    ChatCompletion,
    LLMRequest,
    ModelUsageSummary,
    UsageSummary,
)
from rlm.domain.ports import LLMPort


class _DummyLLM:
    # TODO(phase4/phase5): Add an opt-in integration test variant that uses a real
    # provider adapter selected by config (registry) instead of this dummy.
    def __init__(self) -> None:
        self.model_name = "dummy"
        self._usage = UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(1, 0, 0)})

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        return ChatCompletion(
            root_model=request.model or self.model_name,
            prompt=request.prompt,
            response="FINAL(factory_ok)",
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
def test_create_rlm_factory() -> None:
    llm: LLMPort = _DummyLLM()
    rlm = create_rlm(llm, max_iterations=2, verbose=False)
    cc = rlm.completion("hi")
    assert cc.response == "factory_ok"


@pytest.mark.unit
def test_create_rlm_from_config_uses_default_llm_registry() -> None:
    cfg = RLMConfig(
        llm=LLMConfig(
            backend="mock",
            model_name="dummy",
            backend_kwargs={"script": ["FINAL(factory_ok)"]},
        ),
        other_llms=[],
        env=EnvironmentConfig(environment="local"),
        max_iterations=2,
        verbose=False,
    )
    rlm = create_rlm_from_config(cfg)
    cc = rlm.completion("hi")
    assert cc.response == "factory_ok"


@pytest.mark.unit
def test_create_rlm_from_config_can_build_llm_from_registry() -> None:
    class _Registry:
        def __init__(self) -> None:
            self.seen: list[LLMConfig] = []

        def build(self, config: LLMConfig, /) -> LLMPort:
            self.seen.append(config)
            return _DummyLLM()  # structural typing

    registry = _Registry()
    cfg = RLMConfig(
        llm=LLMConfig(backend="dummy", model_name="dummy"),
        other_llms=[],
        env=EnvironmentConfig(environment="local"),
        max_iterations=2,
        verbose=False,
    )

    rlm = create_rlm_from_config(cfg, llm_registry=registry)
    cc = rlm.completion("hi")
    assert cc.response == "factory_ok"
    assert registry.seen == [cfg.llm]


@pytest.mark.unit
def test_create_rlm_from_config_builds_default_llm_registry_when_llm_passed_and_other_llms_present() -> (
    None
):
    cfg = RLMConfig(
        llm=LLMConfig(backend="openai", model_name="ignored"),
        other_llms=[
            LLMConfig(
                backend="mock",
                model_name="other",
                backend_kwargs={"script": ["FINAL(x)"]},
            )
        ],
        env=EnvironmentConfig(environment="local"),
        max_iterations=1,
        verbose=False,
    )
    rlm = create_rlm_from_config(cfg, llm=_DummyLLM(), llm_registry=None)
    assert len(rlm._other_llms) == 1  # noqa: SLF001 - intentional: verify factory wiring
    assert rlm._other_llms[0].model_name == "other"  # noqa: SLF001
