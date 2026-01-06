from __future__ import annotations

import pytest

from rlm._legacy.clients.base_lm import BaseLM
from rlm._legacy.core.types import ModelUsageSummary, UsageSummary


class _DummyLM(BaseLM):
    """Deterministic LM for boundary testing (no network)."""

    def __init__(self) -> None:
        super().__init__(model_name="dummy")
        self._last_usage = UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(1, 0, 0)})

    def completion(self, prompt):  # type: ignore[override]
        # The legacy orchestrator expects the model to eventually return FINAL(...)
        return "FINAL(done)"

    async def acompletion(self, prompt):  # type: ignore[override]
        return self.completion(prompt)

    def get_usage_summary(self) -> UsageSummary:
        return self._last_usage

    def get_last_usage(self) -> UsageSummary:
        return self._last_usage


@pytest.mark.unit
def test_legacy_orchestrator_boundary_local(monkeypatch: pytest.MonkeyPatch) -> None:
    import rlm._legacy.core.rlm as rlm_mod

    monkeypatch.setattr(rlm_mod, "get_client", lambda backend, backend_kwargs: _DummyLM())

    rlm = rlm_mod.RLM(
        backend="openai",
        backend_kwargs={"model_name": "dummy"},
        environment="local",
        environment_kwargs={},
        max_iterations=2,
        verbose=False,
    )

    cc = rlm.completion("hello")
    assert cc.response == "done"
