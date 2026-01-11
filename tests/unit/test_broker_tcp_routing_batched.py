from __future__ import annotations

from dataclasses import dataclass

import pytest

from rlm.adapters.broker.tcp import TcpBrokerAdapter
from rlm.domain.models import ChatCompletion, LLMRequest, ModelUsageSummary, UsageSummary
from rlm.domain.ports import LLMPort
from rlm.infrastructure.comms.protocol import request_completions_batched


@dataclass
class _BarrierAsyncLLM(LLMPort):
    """
    Async LLM that deadlocks if called sequentially in a batched context.

    Each `acomplete()` waits until all expected calls have started; this forces the
    broker to use real concurrency for batched requests.
    """

    _model_name: str
    expected_calls: int

    def __post_init__(self) -> None:
        import asyncio

        self._started = 0
        self._all_started = asyncio.Event()
        self._usage = UsageSummary(model_usage_summaries={})

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, request: LLMRequest, /) -> ChatCompletion:  # pragma: no cover - not used
        raise AssertionError("_BarrierAsyncLLM.complete should not be used for this test")

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        self._started += 1
        if self._started >= self.expected_calls:
            self._all_started.set()
        await self._all_started.wait()

        self._usage = UsageSummary(
            model_usage_summaries={self._model_name: ModelUsageSummary(1, 0, 0)}
        )
        return ChatCompletion(
            root_model=request.model or self._model_name,
            prompt=request.prompt,
            response=f"resp({request.prompt})",
            usage_summary=self._usage,
            execution_time=0.0,
        )

    def get_usage_summary(self) -> UsageSummary:
        return self._usage

    def get_last_usage(self) -> UsageSummary:
        return self._usage


@pytest.mark.unit
def test_tcp_broker_batched_requests_are_concurrent_and_preserve_order() -> None:
    llm = _BarrierAsyncLLM("default", expected_calls=3)
    broker = TcpBrokerAdapter(llm)
    addr = broker.start()
    try:
        prompts = ["p1", "p2", "p3"]
        results = request_completions_batched(addr, prompts, timeout_s=1.0)

        assert [r.error for r in results] == [None, None, None]
        assert [
            r.chat_completion.prompt for r in results if r.chat_completion is not None
        ] == prompts
        assert (
            results[0].chat_completion is not None
            and results[0].chat_completion.response == "resp(p1)"
        )
        assert (
            results[2].chat_completion is not None
            and results[2].chat_completion.response == "resp(p3)"
        )
    finally:
        broker.stop()
