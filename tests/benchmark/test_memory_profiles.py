"""
Memory-focused benchmarks using pytest-benchmark.
"""

from __future__ import annotations

import gc

import pytest

from rlm.domain.models import CodeBlock, ReplResult
from rlm.domain.models.serialization import serialize_value
from rlm.domain.services.parsing import format_iteration
from rlm.domain.services.rlm_orchestrator import RLMOrchestrator
from tests.benchmark.bench_utils import run_pedantic_once
from tests.performance.perf_utils import (
    BenchmarkEnvironment,
    BenchmarkLLM,
    generate_deep_nested_data,
    generate_large_context,
    make_iteration,
    measure_object_size,
    memory_tracker,
)


@pytest.mark.benchmark
def test_message_history_memory_growth(benchmark) -> None:
    llm = BenchmarkLLM(include_final=True, final_after=20)
    env = BenchmarkEnvironment(result_size=1000)
    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    gc.collect()

    def run():
        with memory_tracker() as mem:
            result = orchestrator.completion(
                prompt="Run many iterations for memory test",
                max_iterations=25,
            )
        return result, mem

    result, mem = run_pedantic_once(benchmark, run)
    assert "done after 20 iterations" in result.response
    benchmark.extra_info["memory_delta_bytes"] = mem.delta_bytes
    benchmark.extra_info["memory_peak_bytes"] = mem.peak_bytes


@pytest.mark.benchmark
def test_serialization_memory_overhead(benchmark) -> None:
    data = generate_deep_nested_data(depth=20, width=5)
    original_size = measure_object_size(data)

    gc.collect()

    def run():
        with memory_tracker() as mem:
            for _ in range(10):
                _ = serialize_value(data)
        return mem

    mem = run_pedantic_once(benchmark, run)
    per_call_overhead = mem.delta_bytes / 10
    benchmark.extra_info["original_size_bytes"] = original_size
    benchmark.extra_info["per_call_overhead_bytes"] = per_call_overhead


@pytest.mark.benchmark
def test_large_context_memory_handling(benchmark) -> None:
    large_context = generate_large_context(num_keys=100, value_size=100_000, nested_depth=0)
    context_size = measure_object_size(large_context)

    gc.collect()

    def run():
        llm = BenchmarkLLM(include_final=True, final_after=2)
        env = BenchmarkEnvironment()
        orchestrator = RLMOrchestrator(llm=llm, environment=env)
        with memory_tracker() as mem:
            _ = orchestrator.completion(prompt=large_context, max_iterations=5)
        return mem

    mem = run_pedantic_once(benchmark, run)
    benchmark.extra_info["context_size_bytes"] = context_size
    benchmark.extra_info["memory_peak_bytes"] = mem.peak_bytes


@pytest.mark.benchmark
def test_format_iteration_memory_efficiency(benchmark) -> None:
    iterations = [make_iteration(num_code_blocks=5, stdout_size=1000) for _ in range(10)]

    gc.collect()

    def run():
        with memory_tracker() as mem:
            for iteration in iterations:
                _ = format_iteration(iteration)
        return mem

    mem = run_pedantic_once(benchmark, run)
    benchmark.extra_info["memory_delta_bytes"] = mem.delta_bytes


@pytest.mark.benchmark
def test_iteration_object_memory_size(benchmark) -> None:
    def run():
        small_iter = make_iteration(num_code_blocks=1, response_size=100, stdout_size=100)
        medium_iter = make_iteration(num_code_blocks=5, response_size=1000, stdout_size=500)
        large_iter = make_iteration(num_code_blocks=10, response_size=5000, stdout_size=2000)

        small_size = measure_object_size(small_iter)
        medium_size = measure_object_size(medium_iter)
        large_size = measure_object_size(large_iter)

        return small_size, medium_size, large_size

    small_size, medium_size, large_size = run_pedantic_once(benchmark, run)
    assert small_size < medium_size < large_size
    benchmark.extra_info["small_size_bytes"] = small_size
    benchmark.extra_info["medium_size_bytes"] = medium_size
    benchmark.extra_info["large_size_bytes"] = large_size


@pytest.mark.benchmark
def test_repl_result_locals_memory(benchmark) -> None:
    many_locals = {f"var_{i}": {"data": f"value_{i}" * 100} for i in range(100)}

    gc.collect()

    def run():
        with memory_tracker() as mem:
            for _ in range(100):
                _ = ReplResult(
                    stdout="output",
                    stderr="",
                    locals=many_locals.copy(),
                    llm_calls=[],
                    execution_time=0.01,
                )
        return mem

    mem = run_pedantic_once(benchmark, run)
    benchmark.extra_info["memory_peak_bytes"] = mem.peak_bytes


@pytest.mark.benchmark
def test_code_block_accumulation_memory(benchmark) -> None:
    def run():
        code_blocks: list[CodeBlock] = []
        for i in range(100):
            code_blocks.append(
                CodeBlock(
                    code=f"x = {i}\n" * 10,
                    result=ReplResult(
                        stdout=f"Output {i}\n" * 10,
                        stderr="",
                        locals={"x": i},
                        llm_calls=[],
                        execution_time=0.001,
                    ),
                )
            )
        total_size = measure_object_size(code_blocks)
        return total_size

    total_size = run_pedantic_once(benchmark, run)
    benchmark.extra_info["code_blocks_size_bytes"] = total_size


@pytest.mark.benchmark
def test_usage_summary_memory_scaling(benchmark) -> None:
    from rlm.domain.models import ModelUsageSummary, UsageSummary

    def run():
        model_counts = [10, 50, 100]
        sizes = []
        for num_models in model_counts:
            summary = UsageSummary(
                model_usage_summaries={
                    f"model-{i}": ModelUsageSummary(
                        total_calls=1000,
                        total_input_tokens=50000,
                        total_output_tokens=25000,
                    )
                    for i in range(num_models)
                }
            )
            sizes.append((num_models, measure_object_size(summary)))
        return sizes

    sizes = run_pedantic_once(benchmark, run)
    for num_models, size in sizes:
        per_model = size / num_models
        benchmark.extra_info[f"per_model_bytes_{num_models}"] = per_model


@pytest.mark.benchmark
def test_prompt_list_memory_growth(benchmark) -> None:
    gc.collect()

    def run():
        with memory_tracker() as mem:
            message_history: list[dict[str, str]] = []
            for i in range(30):
                message_history.append({"role": "assistant", "content": f"Response {i}" * 100})
                message_history.append({"role": "user", "content": f"Result {i}" * 100})
        return mem

    mem = run_pedantic_once(benchmark, run)
    benchmark.extra_info["memory_delta_bytes"] = mem.delta_bytes


@pytest.mark.benchmark
def test_cleanup_releases_memory(benchmark) -> None:
    from rlm.adapters.environments.local import LocalEnvironmentAdapter

    gc.collect()

    def run():
        baseline_objects = len(gc.get_objects())
        env = LocalEnvironmentAdapter()
        env.load_context(generate_large_context(num_keys=100, value_size=10000))

        for i in range(10):
            env.execute_code(f"data_{i} = 'x' * 10000")

        after_use_objects = len(gc.get_objects())
        env.cleanup()
        gc.collect()
        after_cleanup_objects = len(gc.get_objects())

        return baseline_objects, after_use_objects, after_cleanup_objects

    baseline_objects, after_use_objects, after_cleanup_objects = run_pedantic_once(benchmark, run)
    objects_released = after_use_objects - after_cleanup_objects
    objects_added = after_use_objects - baseline_objects
    benchmark.extra_info["objects_added"] = objects_added
    benchmark.extra_info["objects_released"] = objects_released


@pytest.mark.benchmark
def test_serialization_no_memory_leak(benchmark) -> None:
    data = generate_deep_nested_data(depth=10, width=3)

    gc.collect()
    gc.collect()

    def run():
        for _ in range(1000):
            _ = serialize_value(data)

        gc.collect()
        gc.collect()

        with memory_tracker() as mem:
            for _ in range(1000):
                _ = serialize_value(data)
        return mem

    mem = run_pedantic_once(benchmark, run)
    benchmark.extra_info["memory_delta_bytes"] = mem.delta_bytes
