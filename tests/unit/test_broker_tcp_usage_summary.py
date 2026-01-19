from __future__ import annotations

import pytest

from rlm.adapters.broker.tcp import TcpBrokerAdapter
from rlm.domain.errors import ValidationError
from rlm.domain.models import LLMRequest
from tests.fakes_ports import QueueLLM


@pytest.mark.unit
def test_tcp_broker_usage_summary_merges_registered_llm_summaries() -> None:
    llm1 = QueueLLM(model_name="m1", responses=["FINAL(a)", "FINAL(b)"])
    llm2 = QueueLLM(model_name="m2", responses=["FINAL(c)"])

    broker = TcpBrokerAdapter(llm1)
    broker.register_llm("m2", llm2)

    broker.complete(LLMRequest(prompt="p1", model=None))
    broker.complete(LLMRequest(prompt="p1b", model="m1"))
    broker.complete(LLMRequest(prompt="p2", model="m2"))
    with pytest.raises(ValidationError, match="Unknown model"):
        broker.complete(LLMRequest(prompt="p3", model="unknown"))

    summary = broker.get_usage_summary()
    assert summary.model_usage_summaries["m1"].total_calls == 2
    assert summary.model_usage_summaries["m2"].total_calls == 1


@pytest.mark.unit
def test_tcp_broker_usage_summary_counts_only_calls_routed_through_broker() -> None:
    """
    Regression test: broker usage must not include direct LLM calls made outside the broker.

    This matters because the orchestrator calls its root LLM directly; the broker should
    only account for environment subcalls routed through `BrokerPort`.
    """
    llm = QueueLLM(model_name="m1", responses=["FINAL(direct)", "FINAL(broker)"])
    broker = TcpBrokerAdapter(llm)

    # Call the LLM directly (simulates orchestrator's root call).
    llm.complete(LLMRequest(prompt="direct", model=None))

    # Call through broker (simulates env subcall).
    broker.complete(LLMRequest(prompt="via broker", model=None))

    summary = broker.get_usage_summary()
    assert summary.model_usage_summaries["m1"].total_calls == 1
