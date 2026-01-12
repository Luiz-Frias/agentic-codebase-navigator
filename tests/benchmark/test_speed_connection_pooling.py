"""
Benchmarks for connection pooling vs per-request sockets using pytest-benchmark.
"""

from __future__ import annotations

import socket
import threading

import pytest

from rlm.infrastructure.comms.codec import recv_frame, send_frame
from tests.benchmark.bench_utils import run_pedantic_once


def _start_echo_server() -> tuple[int, threading.Event, threading.Thread]:
    ready = threading.Event()
    stop = threading.Event()
    port_holder = {"port": 0}

    def server() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            port_holder["port"] = sock.getsockname()[1]
            sock.listen(5)
            sock.settimeout(0.1)
            ready.set()

            while not stop.is_set():
                try:
                    conn, _ = sock.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break

                with conn:
                    conn.settimeout(1.0)
                    while not stop.is_set():
                        try:
                            frame = recv_frame(conn)
                        except (ConnectionError, OSError):
                            break
                        if frame is None:
                            break
                        try:
                            send_frame(conn, frame)
                        except OSError:
                            break

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    ready.wait(timeout=1.0)
    return port_holder["port"], stop, thread


def _run_pooled(port: int, message: dict[str, object], iterations: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect(("127.0.0.1", port))
        client.settimeout(5.0)
        for _ in range(iterations):
            send_frame(client, message)
            response = recv_frame(client)
            assert response is not None


def _run_unpooled(port: int, message: dict[str, object], iterations: int) -> None:
    for _ in range(iterations):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect(("127.0.0.1", port))
            client.settimeout(5.0)
            send_frame(client, message)
            response = recv_frame(client)
            assert response is not None


@pytest.mark.benchmark
@pytest.mark.parametrize("mode", ["pooled", "unpooled"])
def test_connection_pooling_roundtrip(benchmark, mode: str) -> None:
    message = {"type": "benchmark", "payload": "x" * 1000, "items": list(range(100))}
    port, stop, thread = _start_echo_server()
    assert port != 0

    iterations = 50
    try:

        def run() -> None:
            if mode == "pooled":
                _run_pooled(port, message, iterations)
            else:
                _run_unpooled(port, message, iterations)

        run_pedantic_once(benchmark, run)
    finally:
        stop.set()
        thread.join(timeout=1.0)
