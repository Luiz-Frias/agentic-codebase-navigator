from __future__ import annotations

import pytest

import rlm.api.rlm as rlm_mod
from rlm.api.rlm import RLM
from rlm.domain.errors import ValidationError
from rlm.domain.models import ChatCompletion
from rlm.domain.models.usage import UsageSummary
from rlm.domain.services.prompts import RLM_SYSTEM_PROMPT


class _LLM:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


class _SpyBroker:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def register_llm(self, model_name: str, llm, /) -> None:  # noqa: ANN001
        self.registered.append(model_name)


@pytest.mark.unit
def test_rlm_facade_registers_other_models_and_passes_system_prompt_conditionally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = _SpyBroker()

    captured: dict[str, object] = {}

    def _stub_run_completion(req, *, deps):  # noqa: ANN001
        captured["req"] = req
        captured["deps"] = deps
        return ChatCompletion(
            root_model="m",
            prompt=req.prompt,
            response="ok",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    monkeypatch.setattr(rlm_mod, "run_completion", _stub_run_completion)

    rlm = RLM(
        _LLM("root"),
        other_llms=[_LLM("other")],
        broker_factory=lambda _llm: broker,  # type: ignore[arg-type]
    )
    cc = rlm.completion("hi")
    assert cc.response == "ok"
    assert broker.registered == ["other"]
    assert captured["deps"].system_prompt == RLM_SYSTEM_PROMPT

    broker2 = _SpyBroker()
    captured2: dict[str, object] = {}

    def _stub_run_completion2(req, *, deps):  # noqa: ANN001
        captured2["deps"] = deps
        return ChatCompletion(
            root_model="m",
            prompt=req.prompt,
            response="ok",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    monkeypatch.setattr(rlm_mod, "run_completion", _stub_run_completion2)

    rlm2 = RLM(
        _LLM("root"),
        other_llms=[_LLM("other")],
        broker_factory=lambda _llm: broker2,  # type: ignore[arg-type]
        system_prompt="custom",
    )
    _ = rlm2.completion("hi")
    assert captured2["deps"].system_prompt == "custom"


@pytest.mark.unit
def test_rlm_facade_rejects_duplicate_other_model_names() -> None:
    rlm = RLM(
        _LLM("root"),
        other_llms=[_LLM("root")],
        broker_factory=lambda _llm: _SpyBroker(),
    )  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="Duplicate model registered"):
        rlm.completion("hi")


@pytest.mark.unit
async def test_rlm_facade_acompletion_registers_models_and_passes_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = _SpyBroker()
    captured: dict[str, object] = {}

    async def _stub_arun_completion(req, *, deps):  # noqa: ANN001
        captured["deps"] = deps
        return ChatCompletion(
            root_model="m",
            prompt=req.prompt,
            response="ok",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    monkeypatch.setattr(rlm_mod, "arun_completion", _stub_arun_completion)

    rlm = RLM(
        _LLM("root"),
        other_llms=[_LLM("other")],
        broker_factory=lambda _llm: broker,  # type: ignore[arg-type]
        system_prompt="custom",
    )
    cc = await rlm.acompletion("hi")
    assert cc.response == "ok"
    assert broker.registered == ["other"]
    assert captured["deps"].system_prompt == "custom"


@pytest.mark.unit
async def test_rlm_facade_acompletion_rejects_duplicate_other_model_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _stub_arun_completion(req, *, deps):  # noqa: ANN001
        raise AssertionError("should not be called")

    monkeypatch.setattr(rlm_mod, "arun_completion", _stub_arun_completion)

    rlm = RLM(
        _LLM("root"),
        other_llms=[_LLM("root")],
        broker_factory=lambda _llm: _SpyBroker(),  # type: ignore[arg-type]
    )
    with pytest.raises(ValidationError, match="Duplicate model registered"):
        await rlm.acompletion("hi")
