from __future__ import annotations

import asyncio

import pytest

from rlm.api.factory import create_rlm
from rlm.domain.models import ChatCompletion, LLMRequest, UsageSummary
from rlm.domain.policies.timeouts import BrokerTimeouts, CancellationPolicy
from rlm.domain.ports import BrokerPort, LLMPort
from tests.fakes_ports import CollectingLogger


class _OrchestratorThenBarrierLLM(LLMPort):
    """
    LLM that:
    - returns a single code-block response that calls llm_query_batched([...])
    - implements `acomplete()` with a barrier so the broker must run batched calls concurrently
    """

    def __init__(self, *, expected_subcalls: int) -> None:
        self._model_name = "dummy"
        self._expected = expected_subcalls
        self._started = 0
        self._all_started: asyncio.Event | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        # One-shot orchestrator response: execute a batched subcall and return it as FINAL_VAR.
        response = (
            "```repl\njoined = '|'.join(llm_query_batched(['a','b','c']))\n```\nFINAL_VAR('joined')"
        )
        return ChatCompletion(
            root_model=self._model_name,
            prompt=request.prompt,
            response=response,
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        # Lazily create the barrier event in the running loop.
        if self._all_started is None:
            self._all_started = asyncio.Event()

        self._started += 1
        if self._started >= self._expected:
            self._all_started.set()
        await self._all_started.wait()

        # Echo back per-prompt so ordering is easy to assert.
        return ChatCompletion(
            root_model=request.model or self._model_name,
            prompt=request.prompt,
            response=f"R({request.prompt})",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})

    def get_last_usage(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})


@pytest.mark.integration
def test_local_env_llm_query_batched_executes_subcalls_concurrently_via_broker() -> None:
    llm = _OrchestratorThenBarrierLLM(expected_subcalls=3)
    logger = CollectingLogger()

    def _broker_factory(default_llm: LLMPort, /) -> BrokerPort:
        from rlm.adapters.broker.tcp import TcpBrokerAdapter

        # Protect the test from hangs if concurrency regresses.
        return TcpBrokerAdapter(
            default_llm,
            timeouts=BrokerTimeouts(batched_completion_timeout_s=0.5),
            cancellation=CancellationPolicy(grace_timeout_s=0.1),
        )

    rlm = create_rlm(
        llm,
        environment="local",
        max_iterations=3,
        verbose=False,
        logger=logger,
        broker_factory=_broker_factory,
    )

    cc = rlm.completion("hello")
    assert cc.response == "R(a)|R(b)|R(c)"

    assert len(logger.iterations) == 1
    iter0 = logger.iterations[0]
    assert len(iter0.code_blocks) == 1
    repl = iter0.code_blocks[0].result
    assert [c.response for c in repl.llm_calls] == ["R(a)", "R(b)", "R(c)"]
