from __future__ import annotations

import pytest

from rlm.adapters.broker.tcp import TcpBrokerAdapter
from rlm.infrastructure.comms.protocol import request_completion
from tests.fakes_ports import QueueLLM


@pytest.mark.unit
def test_tcp_broker_routes_single_requests_by_model_name_over_wire_protocol() -> None:
    default_llm = QueueLLM(model_name="default", responses=["FINAL(default)"])
    other_llm = QueueLLM(model_name="other", responses=["FINAL(other)"])

    broker = TcpBrokerAdapter(default_llm)
    broker.register_llm("other", other_llm)

    addr = broker.start()
    try:
        cc_default = request_completion(addr, "hi")
        assert cc_default.root_model == "default"
        assert cc_default.response == "FINAL(default)"

        cc_other = request_completion(addr, "hi", model="other")
        assert cc_other.root_model == "other"
        assert cc_other.response == "FINAL(other)"
    finally:
        broker.stop()
