from __future__ import annotations

import pytest

from rlm.adapters.broker.tcp import TcpBrokerAdapter
from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.domain.errors import BrokerError
from rlm.infrastructure.comms.protocol import request_completion


@pytest.mark.integration
def test_tcp_broker_routes_between_two_mock_llms_by_model_name() -> None:
    llm_a = MockLLMAdapter(model="a", script=["A"])
    llm_b = MockLLMAdapter(model="b", script=["B"])

    broker = TcpBrokerAdapter(llm_a)
    broker.register_llm("b", llm_b)

    addr = broker.start()
    try:
        cc_a = request_completion(addr, "hi")
        assert cc_a.root_model == "a"
        assert cc_a.response == "A"

        cc_b = request_completion(addr, "hi", model="b")
        assert cc_b.root_model == "b"
        assert cc_b.response == "B"

        with pytest.raises(BrokerError, match="Unknown model"):
            request_completion(addr, "hi", model="unknown")
    finally:
        broker.stop()
