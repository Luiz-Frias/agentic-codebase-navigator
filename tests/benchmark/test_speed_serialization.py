"""
Benchmarks for serialization operations using pytest-benchmark.
"""

from __future__ import annotations

import json

import pytest

from rlm.domain.models.query_metadata import QueryMetadata
from rlm.domain.models.serialization import serialize_value
from rlm.infrastructure.comms.codec import encode_frame
from tests.performance.perf_utils import generate_deep_nested_data, generate_large_context


@pytest.mark.benchmark
def test_serialize_value_flat_dict(benchmark) -> None:
    data = {f"key_{i}": f"value_{i}" for i in range(100)}
    result = benchmark(serialize_value, data)
    assert len(result) == 100


@pytest.mark.benchmark
@pytest.mark.parametrize("depth", [5, 10, 20, 50])
def test_serialize_value_nested_structures(benchmark, depth: int) -> None:
    data = generate_deep_nested_data(depth=depth, width=2)
    result = benchmark(serialize_value, data)
    assert result


@pytest.mark.benchmark
@pytest.mark.parametrize("size", [100, 1000, 10000])
def test_serialize_value_large_lists(benchmark, size: int) -> None:
    data = [{"item": i, "value": f"val_{i}"} for i in range(size)]
    result = benchmark(serialize_value, data)
    assert len(result) == size


@pytest.mark.benchmark
def test_serialize_value_mixed_types(benchmark) -> None:
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

    result = benchmark(serialize_value, data)
    assert isinstance(result, dict)


@pytest.mark.benchmark
def test_query_metadata_from_context_dict(benchmark) -> None:
    context = generate_large_context(num_keys=50, value_size=1000, nested_depth=2)
    metadata = benchmark(QueryMetadata.from_context, context)
    assert metadata.context_type == "dict"


@pytest.mark.benchmark
def test_query_metadata_from_context_list(benchmark) -> None:
    context = [{"role": "user", "content": "x" * 1000} for _ in range(100)]
    metadata = benchmark(QueryMetadata.from_context, context)
    assert metadata.context_type == "list"


@pytest.mark.benchmark
def test_query_metadata_from_context_string(benchmark) -> None:
    context = "x" * 100_000
    metadata = benchmark(QueryMetadata.from_context, context)
    assert metadata.context_type == "str"
    assert metadata.context_total_length == 100_000


@pytest.mark.benchmark
@pytest.mark.parametrize("size", [100, 1000, 10000, 100000])
def test_encode_frame_various_sizes(benchmark, size: int) -> None:
    message = {"data": "x" * size, "type": "test"}
    frame = benchmark(encode_frame, message)
    assert len(frame) > size


@pytest.mark.benchmark
@pytest.mark.parametrize("encoder", ["json", "frame"])
def test_json_dumps_vs_encode_frame(benchmark, encoder: str) -> None:
    message = {"data": "x" * 10000, "type": "test", "items": list(range(100))}

    if encoder == "json":
        result = benchmark(json.dumps, message)
    else:
        result = benchmark(encode_frame, message)

    assert result


@pytest.mark.benchmark
def test_serialize_value_recursion_depth(benchmark) -> None:
    import sys

    original_limit = sys.getrecursionlimit()

    try:
        sys.setrecursionlimit(2000)
        depth = 500

        def build_deep(d: int) -> dict:
            if d <= 0:
                return {"leaf": True}
            return {"child": build_deep(d - 1)}

        deep_data = build_deep(depth)
        result = benchmark(serialize_value, deep_data)
        assert result is not None
    finally:
        sys.setrecursionlimit(original_limit)


@pytest.mark.benchmark
def test_serialize_value_repeated_references(benchmark) -> None:
    shared = {"shared": "data", "values": list(range(100))}
    data = {
        "ref1": shared,
        "ref2": shared,
        "ref3": shared,
        "items": [shared for _ in range(10)],
    }

    result = benchmark(serialize_value, data)
    assert result["ref1"] == result["ref2"] == result["ref3"]
