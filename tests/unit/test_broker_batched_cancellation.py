from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

import pytest

from rlm.adapters.broker.tcp import TcpBrokerAdapter
from rlm.domain.errors import LLMError
from rlm.domain.models import BatchedLLMRequest, ChatCompletion, LLMRequest, UsageSummary
from rlm.domain.policies.timeouts import BrokerTimeouts, CancellationPolicy
from rlm.domain.ports import LLMPort


@dataclass
class _CancellableAsyncLLM(LLMPort):
    _model_name: str
    cancelled: threading.Event

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, request: LLMRequest, /) -> ChatCompletion:  # pragma: no cover
        raise AssertionError("_CancellableAsyncLLM.complete should not be used in this test")

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})

    def get_last_usage(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})


@pytest.mark.unit
def test_tcp_broker_batched_timeout_cancels_in_flight_tasks() -> None:
    cancelled = threading.Event()
    llm = _CancellableAsyncLLM("default", cancelled)
    broker = TcpBrokerAdapter(
        llm,
        timeouts=BrokerTimeouts(batched_completion_timeout_s=0.05),
        cancellation=CancellationPolicy(grace_timeout_s=0.2),
    )
    broker.start()
    try:
        with pytest.raises(LLMError, match=r"timed out"):
            broker.complete_batched(BatchedLLMRequest(prompts=["p1", "p2", "p3"]))
        assert cancelled.wait(timeout=1.0) is True
    finally:
        broker.stop()
