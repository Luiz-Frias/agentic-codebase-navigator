from __future__ import annotations

import pytest

from rlm.application.services.legacy_orchestrator import LegacyOrchestratorService
from rlm.domain.models import ChatCompletion, LLMRequest, ModelUsageSummary, UsageSummary
from rlm.domain.ports import LLMPort


class _ScriptLLM:
    # TODO(phase4/phase5): Add an opt-in integration test that runs this service with a
    # real provider adapter (no mocks), gated by env vars/extras so CI stays hermetic.
    def __init__(self) -> None:
        self.model_name = "dummy"
        self._usage = UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(1, 0, 0)})

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        return ChatCompletion(
            root_model=request.model or self.model_name,
            prompt=request.prompt,
            response="FINAL(done)",
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
def test_legacy_orchestrator_service_runs_and_returns_final_answer() -> None:
    llm: LLMPort = _ScriptLLM()  # structural typing
    svc = LegacyOrchestratorService(llm, max_iterations=2, verbose=False)
    cc = svc.completion("hello")
    assert cc.response == "done"
