"""
Speed benchmarks for connection pooling vs per-request sockets.

These tests measure the relative overhead of opening new TCP connections for
each request compared to reusing a single connection (simple pooling).
"""

from __future__ import annotations

import socket
import threading

import pytest

from rlm.infrastructure.comms.codec import recv_frame, send_frame

from .perf_utils import perf_timer


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
                    while True:
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


@pytest.mark.performance
def test_connection_pooling_roundtrip() -> None:
    """
    Compare pooled (single connection) vs unpooled (new connection per request).
    """
    message = {"type": "benchmark", "payload": "x" * 1000, "items": list(range(100))}
    port, stop, thread = _start_echo_server()
    assert port != 0

    iterations = 50
    try:
        with perf_timer() as pooled_timing:
            _run_pooled(port, message, iterations)
        pooled_timing.iterations = iterations

        with perf_timer() as unpooled_timing:
            _run_unpooled(port, message, iterations)
        unpooled_timing.iterations = iterations

        assert pooled_timing.elapsed_seconds < 1.0, (
            f"Pooled roundtrip too slow: {pooled_timing.elapsed_seconds:.3f}s"
        )
        assert unpooled_timing.elapsed_seconds < 2.5, (
            f"Unpooled roundtrip too slow: {unpooled_timing.elapsed_seconds:.3f}s"
        )
        assert pooled_timing.per_iteration_ms < 20, (
            f"Pooled per-roundtrip latency too high: {pooled_timing.per_iteration_ms:.2f}ms"
        )
        assert unpooled_timing.per_iteration_ms < 50, (
            f"Unpooled per-roundtrip latency too high: {unpooled_timing.per_iteration_ms:.2f}ms"
        )
    finally:
        stop.set()
        thread.join(timeout=1.0)
