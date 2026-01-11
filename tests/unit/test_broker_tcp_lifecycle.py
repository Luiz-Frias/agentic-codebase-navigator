from __future__ import annotations

import pytest

from rlm.adapters.broker.tcp import TcpBrokerAdapter
from tests.fakes_ports import QueueLLM


@pytest.mark.unit
def test_tcp_broker_start_is_idempotent() -> None:
    broker = TcpBrokerAdapter(QueueLLM(responses=["FINAL(ok)"]))
    addr1 = broker.start()
    addr2 = broker.start()
    assert addr1 == addr2
    broker.stop()


@pytest.mark.unit
def test_tcp_broker_stop_is_idempotent_and_allows_restart() -> None:
    broker = TcpBrokerAdapter(QueueLLM(responses=["FINAL(ok)"]))

    broker.stop()  # stop-before-start should be safe

    addr1 = broker.start()
    broker.stop()
    broker.stop()

    addr2 = broker.start()
    broker.stop()

    assert addr1[0] == addr2[0] == "127.0.0.1"
