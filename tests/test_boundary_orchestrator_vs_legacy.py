from __future__ import annotations

import pytest

from rlm._legacy.clients.base_lm import BaseLM
from rlm._legacy.core.types import ModelUsageSummary, UsageSummary
from rlm.api import create_rlm
from rlm.domain.ports import LLMPort
from tests.fakes_ports import QueueLLM


class _ScriptedLegacyLM(BaseLM):
    """Legacy BaseLM with a fixed response script (no network)."""

    def __init__(self, *, responses: list[str]) -> None:
        super().__init__(model_name="dummy-legacy")
        self._responses = list(responses)
        self._usage = UsageSummary(
            model_usage_summaries={"dummy-legacy": ModelUsageSummary(1, 0, 0)}
        )

    def completion(self, prompt):  # type: ignore[override]
        if not self._responses:
            raise AssertionError("ScriptedLegacyLM: no scripted responses left")
        return self._responses.pop(0)

    async def acompletion(self, prompt):  # type: ignore[override]
        return self.completion(prompt)

    def get_usage_summary(self) -> UsageSummary:
        return self._usage

    def get_last_usage(self) -> UsageSummary:
        return self._usage


@pytest.mark.integration
def test_new_facade_and_legacy_orchestrator_match_on_simple_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # New (domain) path
    llm: LLMPort = QueueLLM(model_name="mock", responses=["FINAL(ok)"])
    new_rlm = create_rlm(llm, environment="local", max_iterations=1, verbose=False)
    new_cc = new_rlm.completion("hello")
    assert new_cc.response == "ok"

    # Legacy path (patched client router)
    import rlm._legacy.core.rlm as legacy_mod

    legacy_lm = _ScriptedLegacyLM(responses=["FINAL(ok)"])
    monkeypatch.setattr(legacy_mod, "get_client", lambda backend, backend_kwargs: legacy_lm)

    legacy_rlm = legacy_mod.RLM(
        backend="openai",
        backend_kwargs={"model_name": "dummy-legacy"},
        environment="local",
        environment_kwargs={},
        max_iterations=1,
        verbose=False,
    )
    legacy_cc = legacy_rlm.completion("hello")
    assert legacy_cc.response == "ok"


@pytest.mark.integration
def test_new_facade_and_legacy_orchestrator_match_on_code_execution_then_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = [
        "```repl\nprint('HELLO')\n```",
        "FINAL(ok)",
    ]

    # New (domain) path
    llm: LLMPort = QueueLLM(model_name="mock", responses=list(script))
    new_rlm = create_rlm(llm, environment="local", max_iterations=3, verbose=False)
    new_cc = new_rlm.completion("hello")
    assert new_cc.response == "ok"

    # Legacy path
    import rlm._legacy.core.rlm as legacy_mod

    legacy_lm = _ScriptedLegacyLM(responses=list(script))
    monkeypatch.setattr(legacy_mod, "get_client", lambda backend, backend_kwargs: legacy_lm)

    legacy_rlm = legacy_mod.RLM(
        backend="openai",
        backend_kwargs={"model_name": "dummy-legacy"},
        environment="local",
        environment_kwargs={},
        max_iterations=3,
        verbose=False,
    )
    legacy_cc = legacy_rlm.completion("hello")
    assert legacy_cc.response == "ok"
