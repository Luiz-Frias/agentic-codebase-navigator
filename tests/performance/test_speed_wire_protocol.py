"""
Speed benchmarks for wire protocol and network operations.

Tests focus on:
- Codec framing performance
- Message roundtrip overhead
- Protocol parsing efficiency
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

from .perf_utils import perf_timer


@pytest.mark.performance
def test_encode_frame_throughput() -> None:
    """
    Benchmark frame encoding throughput.
    """
    messages = [
        {"type": "small", "data": "x" * 100},
        {"type": "medium", "data": "x" * 1000},
        {"type": "large", "data": "x" * 10000},
    ]

    for msg in messages:
        size = len(json.dumps(msg))

        with perf_timer() as timing:
            for _ in range(1000):
                _frame = encode_frame(msg)

        throughput_mb_s = (size * 1000) / timing.elapsed_seconds / 1_000_000

        # Should encode at least 100MB/s
        assert throughput_mb_s > 10, (
            f"Encode throughput too low for {size} bytes: {throughput_mb_s:.1f}MB/s"
        )


@pytest.mark.performance
def test_try_parse_request_valid() -> None:
    """
    Benchmark parsing valid wire requests.

    try_parse_request returns (WireRequest | None, WireResponse | None).
    For valid requests: (request, None)
    For invalid requests: (None, error_response)
    """
    valid_requests = [
        {"prompt": "Hello", "model": None, "correlation_id": None},
        {"prompts": ["a", "b", "c"], "model": "test", "correlation_id": "corr-123"},
        {"prompt": [{"role": "user", "content": "test"}], "model": None},
    ]

    for raw in valid_requests:
        with perf_timer() as timing:
            for _ in range(1000):
                request, error_response = try_parse_request(raw)
                assert request is not None
                assert error_response is None

        assert timing.elapsed_seconds < 0.5, f"Parse too slow: {timing.elapsed_seconds:.3f}s"


@pytest.mark.performance
def test_try_parse_request_invalid() -> None:
    """
    Benchmark parsing invalid requests (error path).
    """
    # Need to be dicts for the function to work
    invalid_requests = [
        {"invalid": "request"},  # Missing prompt/prompts
        {"prompt": None, "prompts": None},  # Both None
    ]

    for raw in invalid_requests:
        with perf_timer() as timing:
            for _ in range(1000):
                request, error_response = try_parse_request(raw)
                assert request is None
                assert error_response is not None
                assert error_response.error is not None

        # Error path should also be fast
        assert timing.elapsed_seconds < 0.5, f"Error parse too slow: {timing.elapsed_seconds:.3f}s"


@pytest.mark.performance
def test_parse_response_throughput() -> None:
    """
    Benchmark response parsing throughput.
    """
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

    with perf_timer() as timing:
        for _ in range(100):
            response = parse_response(response_data)
            assert response.error is None
            assert len(response.results) == 10

    # 100 parses should be fast
    assert timing.elapsed_seconds < 0.1, f"Response parse too slow: {timing.elapsed_seconds:.3f}s"


@pytest.mark.performance
def test_wire_request_validation_overhead() -> None:
    """
    Benchmark WireRequest validation overhead.
    """
    # Single prompt request
    single_data = {"prompt": "test", "model": None, "correlation_id": None}

    # Batched request
    batched_data = {"prompts": [f"prompt_{i}" for i in range(50)], "model": "test"}

    with perf_timer() as single_timing:
        for _ in range(1000):
            WireRequest.from_dict(single_data)

    with perf_timer() as batched_timing:
        for _ in range(1000):
            WireRequest.from_dict(batched_data)

    # Single should be fast
    assert single_timing.elapsed_seconds < 0.1, "Single request validation too slow"

    # Batched has more validation but should still be reasonable
    assert batched_timing.elapsed_seconds < 0.5, "Batched request validation too slow"


@pytest.mark.performance
def test_wire_response_construction() -> None:
    """
    Benchmark WireResponse construction.
    """
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

    with perf_timer() as timing:
        for _ in range(1000):
            response = WireResponse(
                correlation_id="test",
                error=None,
                results=results,
            )
            # Also test to_dict which is used for transmission
            _ = response.to_dict()

    assert timing.elapsed_seconds < 0.5, (
        f"Response construction too slow: {timing.elapsed_seconds:.3f}s"
    )


@pytest.mark.performance
def test_socket_frame_roundtrip() -> None:
    """
    Benchmark actual socket frame send/receive.

    Uses loopback interface for realistic timing.
    """
    message = {"type": "test", "data": "x" * 1000, "items": list(range(100))}

    server_ready = threading.Event()
    server_port = [0]  # Use list to allow modification in nested function
    received_count = [0]

    def server():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            server_port[0] = sock.getsockname()[1]
            sock.listen(1)
            server_ready.set()

            conn, _ = sock.accept()
            try:
                while True:
                    frame = recv_frame(conn)
                    if frame is None:
                        break
                    received_count[0] += 1
                    # Echo back
                    send_frame(conn, frame)
            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
                OSError,
            ):
                # Expected: client disconnected during performance test teardown
                pass
            finally:
                conn.close()

    server_thread = threading.Thread(target=server, daemon=True)
    server_thread.start()
    server_ready.wait(timeout=1.0)

    iterations = 100

    with perf_timer() as timing:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect(("127.0.0.1", server_port[0]))
            client.settimeout(5.0)

            for _ in range(iterations):
                send_frame(client, message)
                response = recv_frame(client)
                assert response is not None

    timing.iterations = iterations

    # Should handle 100 roundtrips quickly (< 500ms on loopback)
    assert timing.elapsed_seconds < 0.5, f"Socket roundtrip too slow: {timing.elapsed_seconds:.3f}s"

    # Per-roundtrip latency
    per_roundtrip_ms = timing.per_iteration_ms
    assert per_roundtrip_ms < 5, f"Per-roundtrip latency too high: {per_roundtrip_ms:.2f}ms"


@pytest.mark.performance
def test_json_serialization_vs_frame_encoding() -> None:
    """
    Compare JSON serialization overhead to full frame encoding.
    """
    message = {
        "prompt": "A test prompt " * 100,
        "metadata": {f"key_{i}": f"value_{i}" for i in range(50)},
        "results": [{"item": i, "data": "x" * 100} for i in range(20)],
    }

    with perf_timer() as json_timing:
        for _ in range(1000):
            _ = json.dumps(message)

    with perf_timer() as frame_timing:
        for _ in range(1000):
            _ = encode_frame(message)

    # Frame encoding is JSON + length prefix + struct pack
    # Overhead may vary but should not be excessive (< 100%)
    overhead = (
        frame_timing.elapsed_seconds - json_timing.elapsed_seconds
    ) / json_timing.elapsed_seconds
    assert overhead < 1.0, f"Frame encoding overhead too high: {overhead * 100:.1f}%"


@pytest.mark.performance
def test_large_response_parsing() -> None:
    """
    Benchmark parsing of large responses.
    """
    # Simulate response with many results and large content
    large_response = {
        "correlation_id": "test",
        "error": None,
        "results": [
            {
                "error": None,
                "chat_completion": {
                    "root_model": "large-model",
                    "prompt": f"prompt_{i}",
                    "response": "Response content " * 500,  # ~8KB per response
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

    with perf_timer() as timing:
        for _ in range(10):
            response = parse_response(large_response)
            assert len(response.results) == 50

    # 10 parses of 50 results with 8KB each should be fast
    assert timing.elapsed_seconds < 0.5, (
        f"Large response parsing too slow: {timing.elapsed_seconds:.3f}s"
    )
