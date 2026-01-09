from __future__ import annotations

import json
import socket
import struct

import pytest

from rlm.adapters.broker.tcp import TcpBrokerAdapter
from rlm.infrastructure.comms.codec import recv_frame, send_frame
from tests.fakes_ports import QueueLLM


def _send_raw_json_frame(sock: socket.socket, payload_obj: object) -> None:
    payload = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload)) + payload)


@pytest.mark.unit
def test_tcp_broker_rejects_non_object_json_payload_safely() -> None:
    broker = TcpBrokerAdapter(QueueLLM(responses=["FINAL(ok)"]))
    addr = broker.start()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            sock.connect(addr)
            _send_raw_json_frame(sock, ["not", "an", "object"])
            resp = recv_frame(sock, max_message_bytes=1_000_000)
            assert resp is not None
            assert resp.get("results") is None
            assert isinstance(resp.get("error"), str) and resp["error"]
            assert "Traceback" not in resp["error"]
    finally:
        broker.stop()


@pytest.mark.unit
def test_tcp_broker_rejects_missing_prompt_or_prompts() -> None:
    broker = TcpBrokerAdapter(QueueLLM(responses=["FINAL(ok)"]))
    addr = broker.start()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            sock.connect(addr)
            send_frame(sock, {"correlation_id": "cid"})
            resp = recv_frame(sock, max_message_bytes=1_000_000)
            assert resp is not None
            assert resp.get("correlation_id") == "cid"
            assert resp.get("results") is None
            assert isinstance(resp.get("error"), str) and resp["error"]
    finally:
        broker.stop()


@pytest.mark.unit
def test_tcp_broker_rejects_unknown_keys_in_request() -> None:
    broker = TcpBrokerAdapter(QueueLLM(responses=["FINAL(ok)"]))
    addr = broker.start()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            sock.connect(addr)
            send_frame(sock, {"prompt": "hi", "unknown": 1})
            resp = recv_frame(sock, max_message_bytes=1_000_000)
            assert resp is not None
            assert resp.get("results") is None
            assert isinstance(resp.get("error"), str)
            assert "Unknown keys" in resp["error"]
    finally:
        broker.stop()
