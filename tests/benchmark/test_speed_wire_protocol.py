"""
Benchmarks for wire protocol and network operations using pytest-benchmark.
"""

from __future__ import annotations

import json
import socket
import threading

import pytest

from rlm.domain.models import ChatCompletion, UsageSummary
from rlm.infrastructure.comms.codec import encode_frame, recv_frame, send_frame
from rlm.infrastructure.comms.messages import WireRequest, WireResponse, WireResult
from rlm.infrastructure.comms.protocol import parse_response, try_parse_request
from tests.benchmark.bench_utils import run_pedantic_once


@pytest.mark.benchmark
@pytest.mark.parametrize("size", [100, 1000, 10000])
def test_encode_frame_throughput(benchmark, size: int) -> None:
    message = {"type": "size", "data": "x" * size}
    frame = benchmark(encode_frame, message)
    assert frame


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "raw",
    [
        {"prompt": "Hello", "model": None, "correlation_id": None},
        {"prompts": ["a", "b", "c"], "model": "test", "correlation_id": "corr-123"},
        {"prompt": [{"role": "user", "content": "test"}], "model": None},
    ],
)
def test_try_parse_request_valid(benchmark, raw: dict[str, object]) -> None:
    request, error_response = benchmark(try_parse_request, raw)
    assert request is not None
    assert error_response is None


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "raw",
    [
        {"invalid": "request"},
        {"prompt": None, "prompts": None},
    ],
)
def test_try_parse_request_invalid(benchmark, raw: dict[str, object]) -> None:
    request, error_response = benchmark(try_parse_request, raw)
    assert request is None
    assert error_response is not None
    assert error_response.error is not None


@pytest.mark.benchmark
def test_parse_response_throughput(benchmark) -> None:
    response_data = {
        "correlation_id": "test-corr",
        "error": None,
        "results": [
            {
                "error": None,
                "chat_completion": {
                    "root_model": "test",
                    "prompt": "test prompt",
                    "response": "x" * 1000,
                    "usage_summary": {"model_usage_summaries": {}},
                    "execution_time": 0.1,
                },
            }
            for _ in range(10)
        ],
    }

    response = benchmark(parse_response, response_data)
    assert response.error is None
    assert len(response.results) == 10


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "data",
    [
        {"prompt": "test", "model": None, "correlation_id": None},
        {"prompts": [f"prompt_{i}" for i in range(50)], "model": "test"},
    ],
)
def test_wire_request_validation_overhead(benchmark, data: dict[str, object]) -> None:
    request = benchmark(WireRequest.from_dict, data)
    assert request is not None


@pytest.mark.benchmark
def test_wire_response_construction(benchmark) -> None:
    results = [
        WireResult(
            error=None,
            chat_completion=ChatCompletion(
                root_model="test",
                prompt="prompt",
                response=f"response_{i}",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=0.01,
            ),
        )
        for i in range(20)
    ]

    def run() -> dict:
        response = WireResponse(
            correlation_id="test",
            error=None,
            results=results,
        )
        return response.to_dict()

    payload = benchmark(run)
    assert payload["results"]


@pytest.mark.benchmark
def test_socket_frame_roundtrip(benchmark) -> None:
    def run() -> int:
        message = {"type": "test", "data": "x" * 1000, "items": list(range(100))}
        server_ready = threading.Event()
        stop = threading.Event()
        server_port = [0]
        received_count = [0]

        def server() -> None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", 0))
                server_port[0] = sock.getsockname()[1]
                sock.listen(1)
                sock.settimeout(0.1)
                server_ready.set()

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
                            received_count[0] += 1
                            try:
                                send_frame(conn, frame)
                            except OSError:
                                break

        server_thread = threading.Thread(target=server, daemon=True)
        server_thread.start()
        server_ready.wait(timeout=1.0)

        iterations = 50
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.connect(("127.0.0.1", server_port[0]))
                client.settimeout(5.0)

                for _ in range(iterations):
                    send_frame(client, message)
                    response = recv_frame(client)
                    assert response is not None
        finally:
            stop.set()
            server_thread.join(timeout=1.0)

        return received_count[0]

    count = run_pedantic_once(benchmark, run)
    assert count > 0


@pytest.mark.benchmark
@pytest.mark.parametrize("encoder", ["json", "frame"])
def test_json_serialization_vs_frame_encoding(benchmark, encoder: str) -> None:
    message = {
        "prompt": "A test prompt " * 100,
        "metadata": {f"key_{i}": f"value_{i}" for i in range(50)},
        "results": [{"item": i, "data": "x" * 100} for i in range(20)],
    }

    if encoder == "json":
        result = benchmark(json.dumps, message)
    else:
        result = benchmark(encode_frame, message)

    assert result


@pytest.mark.benchmark
def test_large_response_parsing(benchmark) -> None:
    large_response = {
        "correlation_id": "test",
        "error": None,
        "results": [
            {
                "error": None,
                "chat_completion": {
                    "root_model": "large-model",
                    "prompt": f"prompt_{i}",
                    "response": "Response content " * 500,
                    "usage_summary": {
                        "model_usage_summaries": {
                            "model-a": {
                                "total_calls": 1,
                                "total_input_tokens": 1000,
                                "total_output_tokens": 500,
                            },
                            "model-b": {
                                "total_calls": 2,
                                "total_input_tokens": 2000,
                                "total_output_tokens": 1000,
                            },
                        }
                    },
                    "execution_time": 1.5,
                },
            }
            for i in range(50)
        ],
    }

    response = benchmark(parse_response, large_response)
    assert len(response.results) == 50
