"""
Benchmarks for parsing operations using pytest-benchmark.
"""

from __future__ import annotations

import pytest

from rlm.domain.models import CodeBlock, Iteration, ReplResult
from rlm.domain.services.parsing import (
    find_code_blocks,
    find_final_answer,
    format_execution_result,
    format_iteration,
)
from tests.performance.perf_utils import (
    generate_long_text_with_final,
    generate_response_with_code_blocks,
    make_iteration,
)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", [1_000, 10_000, 100_000])
def test_find_code_blocks_scales_with_text_size(benchmark, size: int) -> None:
    text = generate_response_with_code_blocks(num_blocks=10, code_size=size // 10)
    blocks = benchmark(find_code_blocks, text)
    assert len(blocks) == 10


@pytest.mark.benchmark
def test_find_code_blocks_many_blocks(benchmark) -> None:
    num_blocks = 50
    text = generate_response_with_code_blocks(num_blocks=num_blocks, code_size=200)
    blocks = benchmark(find_code_blocks, text)
    assert len(blocks) == num_blocks


@pytest.mark.benchmark
@pytest.mark.parametrize("size", [10_000, 100_000, 1_000_000])
def test_find_final_answer_long_text(benchmark, size: int) -> None:
    text = generate_long_text_with_final(length=size, final_answer="the answer is 42")
    answer = benchmark(find_final_answer, text)
    assert answer == "the answer is 42"


@pytest.mark.benchmark
def test_find_final_answer_nested_parentheses(benchmark) -> None:
    depth = 50
    nested = "(" * depth + "value" + ")" * depth
    text = f"Some preamble text.\n\nFINAL({nested})"
    answer = benchmark(find_final_answer, text)
    assert answer == nested


@pytest.mark.benchmark
def test_find_final_answer_no_match_fast(benchmark) -> None:
    text = "Lorem ipsum " * 10000
    answer = benchmark(find_final_answer, text)
    assert answer is None


@pytest.mark.benchmark
@pytest.mark.parametrize("num_blocks", [1, 5, 10, 20])
def test_format_iteration_scales_with_blocks(benchmark, num_blocks: int) -> None:
    iteration = make_iteration(num_code_blocks=num_blocks, stdout_size=500)
    messages = benchmark(format_iteration, iteration)
    assert len(messages) == 1 + num_blocks


@pytest.mark.benchmark
def test_format_iteration_truncation_fast(benchmark) -> None:
    huge_stdout = "x" * 100_000
    iteration = Iteration(
        correlation_id=None,
        prompt=[{"role": "user", "content": "test"}],
        response="Response with code block",
        code_blocks=[
            CodeBlock(
                code="print('x' * 100000)",
                result=ReplResult(
                    stdout=huge_stdout,
                    stderr="",
                    locals={},
                    llm_calls=[],
                    execution_time=0.0,
                ),
            )
        ],
        final_answer=None,
        iteration_time=0.1,
        iteration_usage_summary=None,
        cumulative_usage_summary=None,
    )

    def run() -> list[dict[str, str]]:
        return format_iteration(iteration, max_character_length=20000)

    messages = benchmark(run)
    user_content = messages[1]["content"]
    assert len(user_content) < 25000


@pytest.mark.benchmark
@pytest.mark.parametrize("size", [100, 1000, 10000])
def test_format_execution_result_various_sizes(benchmark, size: int) -> None:
    repl_result = ReplResult(
        stdout="x" * size,
        stderr="warning: " + "y" * (size // 10),
        locals={f"var_{i}": i for i in range(20)},
        llm_calls=[],
        execution_time=0.01,
    )
    formatted = benchmark(format_execution_result, repl_result)
    assert formatted


@pytest.mark.benchmark
def test_code_block_regex_pathological_input(benchmark) -> None:
    pathological = "```not_repl\ncode\n```\n" * 100 + "```repl\nreal_code\n```"
    blocks = benchmark(find_code_blocks, pathological)
    assert blocks == ["real_code"]
