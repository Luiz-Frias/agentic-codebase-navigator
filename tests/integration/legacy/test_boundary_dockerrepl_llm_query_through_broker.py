from __future__ import annotations

import pytest

from rlm._legacy.environments.docker_repl import DockerREPL
from rlm.adapters.broker.tcp import TcpBrokerAdapter
from rlm.api.registries import ensure_docker_available
from tests.fakes_ports import QueueLLM


@pytest.mark.integration
@pytest.mark.docker
def test_dockerrepl_llm_query_routes_through_broker_and_records_llm_calls() -> None:
    """
    Boundary integration: docker env -> host proxy -> BrokerPort.

    Best-effort and should skip cleanly when Docker isn't available or container
    startup/pulls are blocked in the environment running tests.
    """

    try:
        ensure_docker_available(timeout_s=0.5)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    llm = QueueLLM(model_name="dummy", responses=["FINAL(pong)"])
    broker = TcpBrokerAdapter(llm)
    addr = broker.start()
    try:
        try:
            with DockerREPL(
                image="python:3.12-slim", broker=broker, lm_handler_address=addr
            ) as env:
                res = env.execute_code("resp = llm_query('ping')\nprint(resp)\n")
        except RuntimeError as exc:
            if "Failed to start container" in str(exc):
                pytest.skip(str(exc))
            raise

        assert res.stdout.strip() == "FINAL(pong)"
        assert res.stderr.strip() == ""

        assert len(res.llm_calls) == 1
        call0 = res.llm_calls[0]
        assert call0.root_model == "dummy"
        assert call0.response == "FINAL(pong)"
    finally:
        broker.stop()
