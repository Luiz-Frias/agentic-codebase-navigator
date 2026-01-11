"""
Speed benchmarks for parsing operations.

Tests focus on:
- Code block extraction regex performance
- Final answer parsing with nested parentheses
- Iteration formatting overhead
"""

from __future__ import annotations

import pytest

from rlm.domain.services.parsing import (
    find_code_blocks,
    find_final_answer,
    format_execution_result,
    format_iteration,
)

from .perf_utils import (
    generate_long_text_with_final,
    generate_response_with_code_blocks,
    make_iteration,
    perf_timer,
)


@pytest.mark.performance
def test_find_code_blocks_scales_with_text_size() -> None:
    """
    Benchmark code block extraction on varying text sizes.

    Regex with DOTALL should still be efficient on large text.
    """
    sizes = [1_000, 10_000, 100_000]
    results = []

    for size in sizes:
        text = generate_response_with_code_blocks(num_blocks=10, code_size=size // 10)

        with perf_timer() as timing:
            for _ in range(100):
                blocks = find_code_blocks(text)

        results.append((size, timing.elapsed_seconds, len(blocks)))

    # All sizes should find 10 blocks
    for size, elapsed, block_count in results:
        assert block_count == 10, f"Expected 10 blocks for size {size}"

    # Performance should scale sub-linearly (not O(n^2))
    # 100x larger text should not take 100x longer
    ratio_1k_to_10k = results[1][1] / results[0][1]
    ratio_10k_to_100k = results[2][1] / results[1][1]

    # Should scale less than 20x for 10x size increase
    assert ratio_1k_to_10k < 20, f"Bad scaling 1k->10k: {ratio_1k_to_10k:.1f}x"
    assert ratio_10k_to_100k < 20, f"Bad scaling 10k->100k: {ratio_10k_to_100k:.1f}x"


@pytest.mark.performance
def test_find_code_blocks_many_blocks() -> None:
    """
    Test extraction of many code blocks from single response.
    """
    num_blocks = 50
    text = generate_response_with_code_blocks(num_blocks=num_blocks, code_size=200)

    with perf_timer() as timing:
        for _ in range(100):
            blocks = find_code_blocks(text)

    assert len(blocks) == num_blocks

    # 100 extractions of 50 blocks should be fast
    assert timing.elapsed_seconds < 0.5, f"Many blocks extraction too slow: {timing.elapsed_seconds:.3f}s"


@pytest.mark.performance
def test_find_final_answer_long_text() -> None:
    """
    Benchmark FINAL() extraction from long text.

    FINAL() typically appears at the end, so regex should be efficient.
    """
    sizes = [10_000, 100_000, 1_000_000]
    results = []

    for size in sizes:
        text = generate_long_text_with_final(length=size, final_answer="the answer is 42")

        with perf_timer() as timing:
            for _ in range(100):
                answer = find_final_answer(text)

        results.append((size, timing.elapsed_seconds, answer))

    # All should find the answer
    for size, elapsed, answer in results:
        assert answer == "the answer is 42", f"Failed to find answer in {size} chars"

    # Even 1MB text should be fast (< 1s for 100 iterations)
    assert results[2][1] < 1.0, f"1MB text too slow: {results[2][1]:.3f}s"


@pytest.mark.performance
def test_find_final_answer_nested_parentheses() -> None:
    """
    Test FINAL() parsing with deeply nested parentheses.

    The parenthesis-matching algorithm should handle deep nesting.
    """
    # Create deeply nested expression
    depth = 50
    nested = "(" * depth + "value" + ")" * depth
    text = f"Some preamble text.\n\nFINAL({nested})"

    with perf_timer() as timing:
        for _ in range(1000):
            answer = find_final_answer(text)

    assert answer == nested
    assert timing.elapsed_seconds < 0.1, f"Nested parens too slow: {timing.elapsed_seconds:.3f}s"


@pytest.mark.performance
def test_find_final_answer_no_match_fast() -> None:
    """
    Verify that non-matching text doesn't cause excessive scanning.
    """
    # Large text without FINAL()
    text = "Lorem ipsum " * 10000

    with perf_timer() as timing:
        for _ in range(100):
            answer = find_final_answer(text)

    assert answer is None
    # Scanning large text without match should still be fast (< 1s for 100 iterations)
    assert timing.elapsed_seconds < 1.0, f"No-match case too slow: {timing.elapsed_seconds:.3f}s"


@pytest.mark.performance
def test_format_iteration_scales_with_blocks() -> None:
    """
    Benchmark format_iteration with varying code block counts.
    """
    block_counts = [1, 5, 10, 20]
    results = []

    for num_blocks in block_counts:
        iteration = make_iteration(num_code_blocks=num_blocks, stdout_size=500)

        with perf_timer() as timing:
            for _ in range(100):
                messages = format_iteration(iteration)

        results.append((num_blocks, timing.elapsed_seconds, len(messages)))

    # Message count should be 1 (assistant) + num_blocks (user results)
    for num_blocks, elapsed, msg_count in results:
        assert msg_count == 1 + num_blocks

    # 100 formats of 20 blocks should still be fast
    assert results[-1][1] < 0.1, f"20-block format too slow: {results[-1][1]:.3f}s"


@pytest.mark.performance
def test_format_iteration_truncation_fast() -> None:
    """
    Test that large REPL outputs are truncated efficiently.
    """
    # Create iteration with very large stdout
    from rlm.domain.models import CodeBlock, Iteration, ReplResult

    huge_stdout = "x" * 100_000  # 100KB output

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

    with perf_timer() as timing:
        for _ in range(100):
            messages = format_iteration(iteration, max_character_length=20000)

    # Should truncate to ~20000 chars
    user_content = messages[1]["content"]
    assert len(user_content) < 25000  # Some overhead for formatting

    assert timing.elapsed_seconds < 0.1, f"Truncation too slow: {timing.elapsed_seconds:.3f}s"


@pytest.mark.performance
def test_format_execution_result_various_sizes() -> None:
    """
    Benchmark format_execution_result for different output sizes.
    """
    from rlm.domain.models import ReplResult

    sizes = [100, 1000, 10000]
    results = []

    for size in sizes:
        repl_result = ReplResult(
            stdout="x" * size,
            stderr="warning: " + "y" * (size // 10),
            locals={f"var_{i}": i for i in range(20)},
            llm_calls=[],
            execution_time=0.01,
        )

        with perf_timer() as timing:
            for _ in range(1000):
                formatted = format_execution_result(repl_result)

        results.append((size, timing.elapsed_seconds, len(formatted)))

    # All should complete quickly
    for size, elapsed, _ in results:
        assert elapsed < 0.1, f"Format for size {size} too slow: {elapsed:.3f}s"


@pytest.mark.performance
def test_code_block_regex_pathological_input() -> None:
    """
    Test regex performance on pathological input patterns.

    Some regex patterns can exhibit exponential backtracking.
    """
    # Pattern with many backtick sequences that don't match
    pathological = "```not_repl\ncode\n```\n" * 100 + "```repl\nreal_code\n```"

    with perf_timer() as timing:
        for _ in range(100):
            blocks = find_code_blocks(pathological)

    assert len(blocks) == 1
    assert blocks[0] == "real_code"
    assert timing.elapsed_seconds < 0.1, f"Pathological input too slow: {timing.elapsed_seconds:.3f}s"
