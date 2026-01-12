"""
pytest-benchmark smoke tests to generate benchmark artifacts.
"""

from __future__ import annotations

import pytest

from rlm.domain.services.parsing import find_code_blocks
from rlm.infrastructure.comms.codec import encode_frame

from .perf_utils import generate_response_with_code_blocks


@pytest.mark.performance
def test_benchmark_encode_frame(benchmark) -> None:
    message = {"type": "bench", "data": "x" * 1000, "items": list(range(100))}
    result = benchmark(encode_frame, message)
    assert result


@pytest.mark.performance
def test_benchmark_find_code_blocks(benchmark) -> None:
    text = generate_response_with_code_blocks(num_blocks=5, code_size=100)
    blocks = benchmark(find_code_blocks, text)
    assert len(blocks) == 5
