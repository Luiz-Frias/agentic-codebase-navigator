"""
Benchmarks for RLM orchestrator and iteration loop using pytest-benchmark.
"""

from __future__ import annotations

import asyncio

import pytest

from rlm.domain.models import ModelUsageSummary, UsageSummary
from rlm.domain.services.rlm_orchestrator import (
    RLMOrchestrator,
    _add_usage_totals,
    _clone_usage_totals,
)
from tests.benchmark.bench_utils import run_pedantic_once
from tests.performance.perf_utils import (
    BenchmarkEnvironment,
    BenchmarkLLM,
    get_llm_for_benchmark,
    is_live_llm_enabled,
)


@pytest.mark.benchmark
def test_orchestrator_completes_within_time_budget(benchmark) -> None:
    def run():
        llm = BenchmarkLLM(include_final=True, final_after=5, delay_seconds=0.0)
        env = BenchmarkEnvironment(execute_delay_seconds=0.0)
        orchestrator = RLMOrchestrator(llm=llm, environment=env)
        return orchestrator.completion(prompt="What is 2+2?", max_iterations=10)

    result = run_pedantic_once(benchmark, run)
    assert "done after 5 iterations" in result.response


@pytest.mark.benchmark
def test_orchestrator_message_history_growth_linear(benchmark) -> None:
    def run():
        llm = BenchmarkLLM(include_final=True, final_after=20, delay_seconds=0.0)
        env = BenchmarkEnvironment(execute_delay_seconds=0.0)
        orchestrator = RLMOrchestrator(llm=llm, environment=env)
        return orchestrator.completion(prompt="Run many iterations", max_iterations=25)

    result = run_pedantic_once(benchmark, run)
    assert "done after 20 iterations" in result.response


@pytest.mark.benchmark
def test_usage_totals_accumulation_fast(benchmark) -> None:
    models = [f"model-{i}" for i in range(10)]
    summary = UsageSummary(
        model_usage_summaries={
            model: ModelUsageSummary(
                total_calls=1,
                total_input_tokens=100,
                total_output_tokens=50,
            )
            for model in models
        }
    )

    iterations = 1000

    def run() -> dict[str, ModelUsageSummary]:
        totals: dict[str, ModelUsageSummary] = {}
        for _ in range(iterations):
            _add_usage_totals(totals, summary)
        return totals

    totals = benchmark(run)
    for model in models:
        assert totals[model].total_calls == iterations


@pytest.mark.benchmark
def test_usage_clone_snapshot_fast(benchmark) -> None:
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

    def run():
        snapshot = None
        for _ in range(iterations):
            snapshot = _clone_usage_totals(totals)
        return snapshot

    snapshot = benchmark(run)
    assert snapshot is not None
    assert snapshot.model_usage_summaries is not totals


@pytest.mark.benchmark
def test_orchestrator_with_large_context_payload(benchmark) -> None:
    def run():
        llm = BenchmarkLLM(include_final=True, final_after=3)
        env = BenchmarkEnvironment()
        orchestrator = RLMOrchestrator(llm=llm, environment=env)
        large_context = {
            "data": "x" * 1_000_000,
            "metadata": {"keys": list(range(1000))},
        }
        return orchestrator.completion(prompt=large_context, max_iterations=5)

    result = run_pedantic_once(benchmark, run)
    assert "done after 3 iterations" in result.response


@pytest.mark.benchmark
def test_orchestrator_max_depth_bypass_fast(benchmark) -> None:
    def run():
        llm = BenchmarkLLM()
        env = BenchmarkEnvironment()
        orchestrator = RLMOrchestrator(llm=llm, environment=env)
        return orchestrator.completion(prompt="Quick call", max_depth=1, depth=1)

    result = run_pedantic_once(benchmark, run)
    assert result.response


@pytest.mark.benchmark
def test_async_orchestrator_completes_efficiently(benchmark) -> None:
    def run():
        llm = BenchmarkLLM(include_final=True, final_after=5)
        env = BenchmarkEnvironment()
        orchestrator = RLMOrchestrator(llm=llm, environment=env)
        return asyncio.run(orchestrator.acompletion(prompt="Async test", max_iterations=10))

    result = run_pedantic_once(benchmark, run)
    assert "done after 5 iterations" in result.response


@pytest.mark.benchmark
def test_empty_code_blocks_no_overhead(benchmark) -> None:
    class NoCodeBlockLLM(BenchmarkLLM):
        def _make_response(self) -> str:
            self._call_count += 1
            if self._call_count >= 3:
                return "FINAL(done)"
            return "Thinking about the problem..."

    def run():
        llm = NoCodeBlockLLM()
        env = BenchmarkEnvironment()
        orchestrator = RLMOrchestrator(llm=llm, environment=env)
        return orchestrator.completion(prompt="Test without code", max_iterations=5)

    result = run_pedantic_once(benchmark, run)
    assert result.response == "done"


@pytest.mark.benchmark
def test_orchestrator_single_iteration_timing(benchmark) -> None:
    def run():
        llm = get_llm_for_benchmark(include_final=True, final_after=1)
        env = BenchmarkEnvironment()
        orchestrator = RLMOrchestrator(llm=llm, environment=env)
        return orchestrator.completion(prompt="Return FINAL(42)", max_iterations=3, max_depth=1)

    result = run_pedantic_once(benchmark, run)
    assert result.response is not None
    if is_live_llm_enabled():
        benchmark.extra_info["live_llm"] = True
    else:
        benchmark.extra_info["live_llm"] = False


@pytest.mark.benchmark
def test_orchestrator_multi_iteration_benchmark(benchmark) -> None:
    def run():
        llm = get_llm_for_benchmark(include_final=True, final_after=3)
        env = BenchmarkEnvironment()
        orchestrator = RLMOrchestrator(llm=llm, environment=env)
        return orchestrator.completion(
            prompt="Think step by step, then return FINAL(done)",
            max_iterations=5,
        )

    result = run_pedantic_once(benchmark, run)
    assert result.response is not None
