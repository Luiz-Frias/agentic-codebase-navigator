"""
Speed benchmarks for serialization operations.

Tests focus on:
- Value serialization for JSON conversion
- Query metadata computation
- Codec encoding/decoding
"""

from __future__ import annotations

import json

import pytest

from rlm.domain.models.query_metadata import QueryMetadata
from rlm.domain.models.serialization import serialize_value
from rlm.infrastructure.comms.codec import encode_frame

from .perf_utils import (
    generate_deep_nested_data,
    generate_large_context,
    perf_timer,
)


@pytest.mark.performance
def test_serialize_value_flat_dict() -> None:
    """
    Benchmark serialization of flat dictionary structures.
    """
    data = {f"key_{i}": f"value_{i}" for i in range(100)}

    with perf_timer() as timing:
        for _ in range(1000):
            result = serialize_value(data)

    assert len(result) == 100
    assert timing.elapsed_seconds < 0.1, (
        f"Flat dict serialization too slow: {timing.elapsed_seconds:.3f}s"
    )


@pytest.mark.performance
def test_serialize_value_nested_structures() -> None:
    """
    Benchmark serialization of nested structures.

    Tests recursive serialization performance.
    """
    depths = [5, 10, 20, 50]
    results = []

    for depth in depths:
        data = generate_deep_nested_data(depth=depth, width=2)

        with perf_timer() as timing:
            for _ in range(100):
                _result = serialize_value(data)

        results.append((depth, timing.elapsed_seconds))

    # All depths should complete
    for depth, elapsed in results:
        # Even depth=50 should be fast
        assert elapsed < 0.5, f"Depth {depth} too slow: {elapsed:.3f}s"


@pytest.mark.performance
def test_serialize_value_large_lists() -> None:
    """
    Test serialization of large list structures.
    """
    sizes = [100, 1000, 10000]
    results = []

    for size in sizes:
        data = [{"item": i, "value": f"val_{i}"} for i in range(size)]

        with perf_timer() as timing:
            for _ in range(10):
                result = serialize_value(data)

        results.append((size, timing.elapsed_seconds, len(result)))

    # Verify all items serialized
    for size, _elapsed, result_len in results:
        assert result_len == size

    # 10000 items should still be reasonable
    assert results[-1][1] < 1.0, f"10000 items too slow: {results[-1][1]:.3f}s"


@pytest.mark.performance
def test_serialize_value_mixed_types() -> None:
    """
    Test serialization of mixed type structures.
    """
    import types

    data = {
        "string": "hello",
        "int": 42,
        "float": 3.14,
        "bool": True,
        "none": None,
        "list": [1, 2, 3],
        "nested": {"a": {"b": {"c": "deep"}}},
        "callable": lambda x: x,
        "module": types,
    }

    with perf_timer() as timing:
        for _ in range(1000):
            result = serialize_value(data)

    assert isinstance(result, dict)
    assert timing.elapsed_seconds < 0.1, f"Mixed types too slow: {timing.elapsed_seconds:.3f}s"


@pytest.mark.performance
def test_query_metadata_from_context_dict() -> None:
    """
    Benchmark QueryMetadata.from_context with dict context.
    """
    context = generate_large_context(num_keys=50, value_size=1000, nested_depth=2)

    with perf_timer() as timing:
        for _ in range(100):
            metadata = QueryMetadata.from_context(context)

    assert metadata.context_type == "dict"
    assert timing.elapsed_seconds < 0.5, (
        f"Dict context metadata too slow: {timing.elapsed_seconds:.3f}s"
    )


@pytest.mark.performance
def test_query_metadata_from_context_list() -> None:
    """
    Benchmark QueryMetadata.from_context with list context.
    """
    context = [{"role": "user", "content": "x" * 1000} for _ in range(100)]

    with perf_timer() as timing:
        for _ in range(100):
            metadata = QueryMetadata.from_context(context)

    assert metadata.context_type == "list"
    assert timing.elapsed_seconds < 0.5, (
        f"List context metadata too slow: {timing.elapsed_seconds:.3f}s"
    )


@pytest.mark.performance
def test_query_metadata_from_context_string() -> None:
    """
    Benchmark QueryMetadata.from_context with large string context.
    """
    context = "x" * 100_000

    with perf_timer() as timing:
        for _ in range(100):
            metadata = QueryMetadata.from_context(context)

    assert metadata.context_type == "str"
    assert metadata.context_total_length == 100_000
    assert timing.elapsed_seconds < 0.1, (
        f"String context metadata too slow: {timing.elapsed_seconds:.3f}s"
    )


@pytest.mark.performance
def test_encode_frame_various_sizes() -> None:
    """
    Benchmark frame encoding for various message sizes.
    """
    sizes = [100, 1000, 10000, 100000]
    results = []

    for size in sizes:
        message = {"data": "x" * size, "type": "test"}

        with perf_timer() as timing:
            for _ in range(100):
                frame = encode_frame(message)

        results.append((size, timing.elapsed_seconds, len(frame)))

    # All should complete quickly
    for size, elapsed, frame_len in results:
        # Frame should be slightly larger than data (JSON overhead + 4 byte header)
        assert frame_len > size
        assert elapsed < 0.5, f"Encode size {size} too slow: {elapsed:.3f}s"


@pytest.mark.performance
def test_json_dumps_vs_encode_frame() -> None:
    """
    Compare raw json.dumps to full frame encoding.

    Frame encoding adds length prefix overhead.
    """
    message = {"data": "x" * 10000, "type": "test", "items": list(range(100))}

    with perf_timer() as json_timing:
        for _ in range(1000):
            json.dumps(message)

    with perf_timer() as frame_timing:
        for _ in range(1000):
            encode_frame(message)

    # Frame encoding should not add more than 2x overhead
    ratio = frame_timing.elapsed_seconds / json_timing.elapsed_seconds
    assert ratio < 2.0, f"Frame encoding too slow vs JSON: {ratio:.2f}x"


@pytest.mark.performance
def test_serialize_value_recursion_depth() -> None:
    """
    Test serialization handles deep recursion safely.

    Note: Python default recursion limit is 1000.
    """
    import sys

    # Save original limit
    original_limit = sys.getrecursionlimit()

    try:
        # Increase for this test
        sys.setrecursionlimit(2000)

        # Create structure near recursion limit
        depth = 500

        def build_deep(d: int) -> dict:
            if d <= 0:
                return {"leaf": True}
            return {"child": build_deep(d - 1)}

        deep_data = build_deep(depth)

        with perf_timer() as timing:
            result = serialize_value(deep_data)

        # Should complete without stack overflow
        assert result is not None
        assert timing.elapsed_seconds < 0.5, (
            f"Deep recursion too slow: {timing.elapsed_seconds:.3f}s"
        )

    finally:
        sys.setrecursionlimit(original_limit)


@pytest.mark.performance
def test_serialize_value_repeated_references() -> None:
    """
    Test serialization with shared object references.

    Same object appearing multiple times should still serialize.
    """
    shared = {"shared": "data", "values": list(range(100))}
    data = {
        "ref1": shared,
        "ref2": shared,
        "ref3": shared,
        "items": [shared for _ in range(10)],
    }

    with perf_timer() as timing:
        for _ in range(100):
            result = serialize_value(data)

    # Each reference should be serialized independently
    assert result["ref1"] == result["ref2"] == result["ref3"]
    assert timing.elapsed_seconds < 0.1, f"Shared refs too slow: {timing.elapsed_seconds:.3f}s"
