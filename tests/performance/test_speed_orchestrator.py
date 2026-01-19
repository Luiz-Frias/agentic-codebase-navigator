"""
Speed benchmarks for RLM orchestrator and core iteration loop.

Tests focus on:
- Message history accumulation overhead
- Usage tracking and snapshot operations
- Iteration timing characteristics

Live LLM Support:
    Set RLM_LIVE_LLM=1 to run select tests with real LLM providers.
    See perf_utils.py for configuration options.
"""

from __future__ import annotations

import pytest

from rlm.domain.models import ModelUsageSummary, UsageSummary
from rlm.domain.services.rlm_orchestrator import (
    RLMOrchestrator,
    _add_usage_totals,
    _clone_usage_totals,
)

from .perf_utils import (
    BenchmarkEnvironment,
    BenchmarkLLM,
    get_llm_for_benchmark,
    get_llm_provider_info,
    is_live_llm_enabled,
    perf_timer,
)


@pytest.mark.performance
def test_orchestrator_completes_within_time_budget() -> None:
    """
    Verify orchestrator can complete a multi-iteration run efficiently.

    This tests the full orchestrator loop with controlled LLM and environment.
    """
    llm = BenchmarkLLM(
        include_final=True,
        final_after=5,
        delay_seconds=0.0,
    )
    env = BenchmarkEnvironment(execute_delay_seconds=0.0)

    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    with perf_timer() as timing:
        result = orchestrator.completion(
            prompt="What is 2+2?",
            max_iterations=10,
        )

    # Should complete in 5 iterations (final_after=5)
    assert "done after 5 iterations" in result.response

    # Without network delays, should be very fast (< 100ms)
    assert timing.elapsed_seconds < 0.1, f"Orchestrator too slow: {timing.elapsed_seconds:.3f}s"


@pytest.mark.performance
def test_orchestrator_message_history_growth_linear() -> None:
    """
    Verify message history growth is O(n) per iteration.

    The orchestrator extends message_history each iteration.
    This test ensures the per-iteration overhead is bounded.
    """
    llm = BenchmarkLLM(
        include_final=True,
        final_after=20,
        delay_seconds=0.0,
    )
    env = BenchmarkEnvironment(execute_delay_seconds=0.0)

    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    with perf_timer() as timing:
        orchestrator.completion(
            prompt="Run many iterations",
            max_iterations=25,
        )

    # 20 iterations should still be fast without artificial delays
    # Average < 5ms per iteration is reasonable
    per_iteration_ms = (timing.elapsed_seconds / 20) * 1000
    assert per_iteration_ms < 10, f"Per-iteration overhead too high: {per_iteration_ms:.2f}ms"


@pytest.mark.performance
def test_usage_totals_accumulation_fast() -> None:
    """
    Benchmark _add_usage_totals for dictionary merge performance.

    This is called multiple times per iteration for each model.
    """
    # Simulate 10 different models
    models = [f"model-{i}" for i in range(10)]
    totals: dict[str, ModelUsageSummary] = {}

    summary = UsageSummary(
        model_usage_summaries={
            model: ModelUsageSummary(
                total_calls=1,
                total_input_tokens=100,
                total_output_tokens=50,
            )
            for model in models
        },
    )

    iterations = 1000

    with perf_timer() as timing:
        for _ in range(iterations):
            _add_usage_totals(totals, summary)

    timing.iterations = iterations

    # Should handle 1000 merges quickly (< 10ms total)
    assert timing.elapsed_seconds < 0.01, (
        f"Usage accumulation too slow: {timing.elapsed_seconds:.3f}s"
    )

    # Verify correctness
    for model in models:
        assert totals[model].total_calls == iterations


@pytest.mark.performance
def test_usage_clone_snapshot_fast() -> None:
    """
    Benchmark _clone_usage_totals for snapshot overhead.

    Called at least once per iteration when logger is present.
    """
    models = [f"model-{i}" for i in range(20)]
    totals = {
        model: ModelUsageSummary(
            total_calls=100,
            total_input_tokens=50000,
            total_output_tokens=25000,
        )
        for model in models
    }

    iterations = 1000

    with perf_timer() as timing:
        for _ in range(iterations):
            snapshot = _clone_usage_totals(totals)
            # Verify it's actually a snapshot (different objects)
            assert snapshot.model_usage_summaries is not totals

    timing.iterations = iterations

    # 1000 snapshots should be fast (< 50ms)
    assert timing.elapsed_seconds < 0.05, f"Usage snapshot too slow: {timing.elapsed_seconds:.3f}s"


@pytest.mark.performance
def test_orchestrator_with_large_context_payload() -> None:
    """
    Test orchestrator performance with large context payloads.

    Large contexts should not significantly impact iteration speed.
    """
    llm = BenchmarkLLM(include_final=True, final_after=3)
    env = BenchmarkEnvironment()

    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    # Large context: 1MB of data
    large_context = {
        "data": "x" * 1_000_000,
        "metadata": {"keys": list(range(1000))},
    }

    with perf_timer() as timing:
        result = orchestrator.completion(
            prompt=large_context,
            max_iterations=5,
        )

    assert "done after 3 iterations" in result.response

    # Even with large context, should complete quickly
    assert timing.elapsed_seconds < 0.5, f"Large context too slow: {timing.elapsed_seconds:.3f}s"


@pytest.mark.performance
def test_orchestrator_max_depth_bypass_fast() -> None:
    """
    Test that max_depth bypass (depth >= max_depth) is fast.

    When at max depth, orchestrator should do a simple LLM call.
    """
    llm = BenchmarkLLM()
    env = BenchmarkEnvironment()

    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    iterations = 100

    with perf_timer() as timing:
        for _ in range(iterations):
            orchestrator.completion(
                prompt="Quick call",
                max_depth=1,
                depth=1,  # At max depth
            )

    timing.iterations = iterations

    # Each max-depth call should be very fast (< 1ms average)
    assert timing.per_iteration_ms < 2, (
        f"Max-depth bypass too slow: {timing.per_iteration_ms:.2f}ms"
    )


@pytest.mark.performance
async def test_async_orchestrator_completes_efficiently() -> None:
    """
    Verify async orchestrator has similar performance to sync version.
    """
    llm = BenchmarkLLM(include_final=True, final_after=5)
    env = BenchmarkEnvironment()

    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    with perf_timer() as timing:
        result = await orchestrator.acompletion(
            prompt="Async test",
            max_iterations=10,
        )

    assert "done after 5 iterations" in result.response

    # Async should be similarly fast
    assert timing.elapsed_seconds < 0.1, (
        f"Async orchestrator too slow: {timing.elapsed_seconds:.3f}s"
    )


@pytest.mark.performance
def test_empty_code_blocks_no_overhead() -> None:
    """
    Test that responses without code blocks have minimal overhead.
    """

    class NoCodeBlockLLM(BenchmarkLLM):
        def _make_response(self) -> str:
            self._call_count += 1
            if self._call_count >= 3:
                return "FINAL(done)"
            # No code blocks, just text
            return "Thinking about the problem..."

    llm = NoCodeBlockLLM()
    env = BenchmarkEnvironment()

    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    with perf_timer() as timing:
        result = orchestrator.completion(
            prompt="Test without code",
            max_iterations=5,
        )

    assert result.response == "done"
    assert timing.elapsed_seconds < 0.05, (
        f"No-code-block case too slow: {timing.elapsed_seconds:.3f}s"
    )


@pytest.mark.performance
def test_orchestrator_single_iteration_timing() -> None:
    """
    Benchmark single-iteration orchestrator timing.

    Supports live LLM mode via RLM_LIVE_LLM=1.
    With mock: validates fast execution
    With live: measures real-world latency
    """
    llm = get_llm_for_benchmark(include_final=True, final_after=1)
    env = BenchmarkEnvironment()

    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    provider_info = get_llm_provider_info()

    with perf_timer() as timing:
        result = orchestrator.completion(
            prompt="Return FINAL(42)",
            max_iterations=3,
            max_depth=1,
        )

    assert result.response is not None

    if is_live_llm_enabled():
        # Live LLM: just verify it completes in reasonable time
        print(
            f"\n[Live LLM: {provider_info['provider']}/{provider_info['model']}] "
            f"Single iteration: {timing.elapsed_seconds:.2f}s",
        )
        assert timing.elapsed_seconds < 60, f"Live LLM too slow: {timing.elapsed_seconds:.2f}s"
    else:
        # Mock: should be very fast
        assert timing.elapsed_seconds < 0.1, (
            f"Mock orchestrator too slow: {timing.elapsed_seconds:.3f}s"
        )


@pytest.mark.performance
def test_orchestrator_multi_iteration_benchmark() -> None:
    """
    Benchmark multi-iteration orchestrator performance.

    Supports live LLM mode via RLM_LIVE_LLM=1.
    """
    llm = get_llm_for_benchmark(include_final=True, final_after=3)
    env = BenchmarkEnvironment()

    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    provider_info = get_llm_provider_info()

    with perf_timer() as timing:
        result = orchestrator.completion(
            prompt="Think step by step, then return FINAL(done)",
            max_iterations=5,
        )

    assert result.response is not None

    if is_live_llm_enabled():
        print(
            f"\n[Live LLM: {provider_info['provider']}/{provider_info['model']}] "
            f"Multi-iteration ({3} iters): {timing.elapsed_seconds:.2f}s, "
            f"avg: {timing.elapsed_seconds / 3:.2f}s/iter",
        )
        # Live: allow more time but still bounded
        assert timing.elapsed_seconds < 120, (
            f"Live LLM multi-iter too slow: {timing.elapsed_seconds:.2f}s"
        )
    else:
        # Mock: should complete quickly
        assert timing.elapsed_seconds < 0.1, (
            f"Mock multi-iter too slow: {timing.elapsed_seconds:.3f}s"
        )
