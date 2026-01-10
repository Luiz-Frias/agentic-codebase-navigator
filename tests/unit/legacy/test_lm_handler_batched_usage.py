from __future__ import annotations

import pytest

from rlm._legacy.clients.base_lm import BaseLM
from rlm._legacy.core.comms_utils import LMRequest
from rlm._legacy.core.lm_handler import LMHandler, LMRequestHandler
from rlm._legacy.core.types import ModelUsageSummary, UsageSummary


class _PerCallUsageLM(BaseLM):
    """Deterministic LM that sets distinct `last_usage` per call based on prompt length."""

    def __init__(self) -> None:
        super().__init__(model_name="dummy")
        self._last_usage = UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(0, 0, 0)})
        self._summary = UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(0, 0, 0)})

    def completion(self, prompt):  # type: ignore[override]
        text = str(prompt)
        self._set_usage(text)
        return f"resp:{text}"

    async def acompletion(self, prompt):  # type: ignore[override]
        return self.completion(prompt)

    def _set_usage(self, text: str) -> None:
        # Use prompt length as a stand-in for input tokens so assertions are easy.
        in_tokens = len(text)
        out_tokens = len(f"resp:{text}")
        self._last_usage = UsageSummary(
            model_usage_summaries={"dummy": ModelUsageSummary(1, in_tokens, out_tokens)}
        )

        # Maintain a simple cumulative summary.
        s = self._summary.model_usage_summaries["dummy"]
        self._summary = UsageSummary(
            model_usage_summaries={
                "dummy": ModelUsageSummary(
                    s.total_calls + 1,
                    s.total_input_tokens + in_tokens,
                    s.total_output_tokens + out_tokens,
                )
            }
        )

    def get_usage_summary(self) -> UsageSummary:
        return self._summary

    def get_last_usage(self) -> UsageSummary:
        return self._last_usage


@pytest.mark.unit
def test_lm_handler_batched_records_per_prompt_last_usage() -> None:
    lm = _PerCallUsageLM()
    handler = LMHandler(lm)
    request = LMRequest(prompts=["a", "bbbb", "cc"])

    # Call the request-handler helper directly (no sockets needed for this unit test).
    req_handler = object.__new__(LMRequestHandler)
    resp = LMRequestHandler._handle_batched(req_handler, request, handler)

    assert resp.success
    assert resp.chat_completions is not None
    assert [c.response for c in resp.chat_completions] == ["resp:a", "resp:bbbb", "resp:cc"]

    # Ensure per-completion usage reflects each prompt (not all identical).
    got = [
        c.usage_summary.model_usage_summaries["dummy"].total_input_tokens
        for c in resp.chat_completions
    ]
    assert got == [1, 4, 2]
