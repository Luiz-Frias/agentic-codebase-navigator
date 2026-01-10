from __future__ import annotations

import pytest

from rlm._legacy.environments.local_repl import LocalREPL
from rlm.adapters.broker.tcp import TcpBrokerAdapter
from tests.fakes_ports import QueueLLM


@pytest.mark.integration
def test_localrepl_llm_query_routes_through_tcp_broker_and_records_llm_calls() -> None:
    llm = QueueLLM(model_name="dummy", responses=["FINAL(pong)"])
    broker = TcpBrokerAdapter(llm)
    addr = broker.start()
    try:
        with LocalREPL(lm_handler_address=addr) as env:
            res = env.execute_code("resp = llm_query('ping')\nprint(resp)\n")
            assert res.stdout.strip() == "FINAL(pong)"
            assert res.stderr.strip() == ""
            assert len(res.llm_calls) == 1
            assert res.llm_calls[0].root_model == "dummy"
            assert res.llm_calls[0].response == "FINAL(pong)"
    finally:
        broker.stop()
