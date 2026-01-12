from __future__ import annotations

import asyncio

import pytest

from rlm.adapters.broker.tcp import TcpBrokerAdapter
from rlm.domain.models import (
    BatchedLLMRequest,
    ChatCompletion,
    LLMRequest,
    UsageSummary,
)
from rlm.domain.policies.timeouts import BrokerTimeouts, CancellationPolicy
from rlm.domain.ports import LLMPort
from tests.benchmark.bench_utils import run_pedantic_once


class _BarrierLLM(LLMPort):
    """
    LLM that blocks until *all* batched acomplete calls have started.
    """

    def __init__(self, *, expected: int) -> None:
        self._expected = expected
        self._started = 0
        self._all_started: asyncio.Event | None = None

    @property
    def model_name(self) -> str:
        return "barrier-llm"

    def complete(self, request: LLMRequest, /) -> ChatCompletion:  # pragma: no cover
        raise AssertionError("This test expects broker.complete_batched to use acomplete()")

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        if self._all_started is None:
            self._all_started = asyncio.Event()

        self._started += 1
        if self._started >= self._expected:
            self._all_started.set()
        await self._all_started.wait()

        return ChatCompletion(
            root_model=request.model or self.model_name,
            prompt=request.prompt,
            response=f"R({request.prompt})",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})

    def get_last_usage(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})


@pytest.mark.benchmark
def test_tcp_broker_complete_batched_runs_subcalls_concurrently_under_load(benchmark) -> None:
    def run() -> list[str]:
        expected = 25
        llm: LLMPort = _BarrierLLM(expected=expected)
        broker = TcpBrokerAdapter(
            llm,
            timeouts=BrokerTimeouts(batched_completion_timeout_s=0.75),
            cancellation=CancellationPolicy(grace_timeout_s=0.1),
        )
        broker.start()
        try:
            req = BatchedLLMRequest(prompts=[f"p{i}" for i in range(expected)], model=None)
            out = broker.complete_batched(req)
            return [c.response for c in out]
        finally:
            broker.stop()

    responses = run_pedantic_once(benchmark, run)
    assert responses == [f"R(p{i})" for i in range(25)]
