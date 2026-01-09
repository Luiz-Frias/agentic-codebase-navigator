from __future__ import annotations

import asyncio

import pytest

from rlm.api.factory import create_rlm
from rlm.domain.models import ChatCompletion, LLMRequest, UsageSummary
from rlm.domain.policies.timeouts import BrokerTimeouts, CancellationPolicy
from rlm.domain.ports import BrokerPort, LLMPort
from tests.fakes_ports import CollectingLogger


class _OrchestratorThenHangingAsyncLLM(LLMPort):
    """
    Boundary helper:
    - sync `complete()` returns code that calls llm_query_batched(...)
    - async `acomplete()` never returns (broker must time out/cancel)
    """

    @property
    def model_name(self) -> str:
        return "dummy"

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        response = (
            "```repl\njoined = '|'.join(llm_query_batched(['a','b','c']))\n```\nFINAL_VAR('joined')"
        )
        return ChatCompletion(
            root_model=self.model_name,
            prompt=request.prompt,
            response=response,
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        await asyncio.sleep(3600)
        return ChatCompletion(
            root_model=self.model_name,
            prompt=request.prompt,
            response="unreachable",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})

    def get_last_usage(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})


@pytest.mark.integration
def test_timeouts_and_correlation_id_are_visible_end_to_end_in_logger_and_replresult() -> None:
    llm = _OrchestratorThenHangingAsyncLLM()
    logger = CollectingLogger()

    def _broker_factory(default_llm: LLMPort, /) -> BrokerPort:
        from rlm.adapters.broker.tcp import TcpBrokerAdapter

        # Keep the boundary test fast and non-hanging.
        return TcpBrokerAdapter(
            default_llm,
            timeouts=BrokerTimeouts(batched_completion_timeout_s=0.2),
            cancellation=CancellationPolicy(grace_timeout_s=0.05),
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
    assert "timed out" in cc.response.lower()
    assert "traceback" not in cc.response.lower()

    # Correlation ID must be consistent across metadata + iteration + repl result.
    assert len(logger.metadata) == 1
    md = logger.metadata[0]
    assert md.correlation_id is not None
    assert md.environment_type == "local"

    assert len(logger.iterations) == 1
    it0 = logger.iterations[0]
    assert it0.correlation_id == md.correlation_id
    assert len(it0.code_blocks) == 1
    repl = it0.code_blocks[0].result
    assert repl.correlation_id == md.correlation_id
