"""
Memory profiling tests for RLM hotspots.

Tests focus on:
- Message history memory accumulation
- Serialization memory overhead
- Namespace snapshot memory usage
- Large payload handling
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
import tracemalloc

import pytest

from rlm.domain.models import CodeBlock, ReplResult
from rlm.domain.models.serialization import serialize_value
from rlm.domain.services.parsing import format_iteration
from rlm.domain.services.rlm_orchestrator import RLMOrchestrator

from .perf_utils import (
    BenchmarkEnvironment,
    BenchmarkLLM,
    generate_deep_nested_data,
    generate_large_context,
    make_iteration,
    measure_object_size,
    memory_tracker,
)


@pytest.mark.performance
def test_message_history_memory_growth() -> None:
    """
    Profile memory growth of message history over iterations.

    Message history grows O(n) per iteration, accumulating all
    previous responses and execution results.
    """
    llm = BenchmarkLLM(include_final=True, final_after=20)
    env = BenchmarkEnvironment(result_size=1000)  # 1KB per execution

    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    gc.collect()

    with memory_tracker() as mem:
        result = orchestrator.completion(
            prompt="Run many iterations for memory test",
            max_iterations=25,
        )

    assert "done after 20 iterations" in result.response

    # Check memory delta is reasonable
    # 20 iterations with 1KB results each = ~20KB minimum
    # With message overhead, should be less than 1MB
    assert mem.delta_bytes < 1_000_000, f"Memory growth too high: {mem.delta_bytes / 1024:.1f}KB"


@pytest.mark.performance
def test_serialization_memory_overhead() -> None:
    """
    Profile memory overhead of serialize_value on nested structures.
    """
    data = generate_deep_nested_data(depth=20, width=5)
    original_size = measure_object_size(data)

    gc.collect()

    with memory_tracker() as mem:
        for _ in range(10):
            _result = serialize_value(data)

    # Serialization shouldn't use more than 10x the original size per call
    per_call_overhead = mem.delta_bytes / 10
    assert per_call_overhead < original_size * 10, (
        f"Serialization overhead too high: {per_call_overhead / 1024:.1f}KB vs original {original_size / 1024:.1f}KB"
    )


@pytest.mark.performance
def test_large_context_memory_handling() -> None:
    """
    Profile memory usage when handling large context payloads.
    """
    # Create 10MB context
    large_context = generate_large_context(
        num_keys=100,
        value_size=100_000,  # 100KB per value
        nested_depth=0,
    )

    context_size = measure_object_size(large_context)

    gc.collect()

    with memory_tracker() as mem:
        llm = BenchmarkLLM(include_final=True, final_after=2)
        env = BenchmarkEnvironment()

        orchestrator = RLMOrchestrator(llm=llm, environment=env)
        _result = orchestrator.completion(
            prompt=large_context,
            max_iterations=5,
        )

    # Memory should not double for processing
    assert mem.peak_bytes < context_size * 3, (
        f"Peak memory too high: {mem.peak_bytes / 1024 / 1024:.1f}MB for {context_size / 1024 / 1024:.1f}MB context"
    )


@pytest.mark.performance
def test_format_iteration_memory_efficiency() -> None:
    """
    Profile memory allocation during iteration formatting.
    """
    iterations = [make_iteration(num_code_blocks=5, stdout_size=1000) for _ in range(10)]

    gc.collect()

    with memory_tracker() as mem:
        for iteration in iterations:
            _messages = format_iteration(iteration)

    # Should not create excessive copies
    # 10 iterations with 5KB output each = ~50KB
    # With formatting overhead, should be less than 500KB
    assert mem.delta_bytes < 500_000, (
        f"Format iteration memory too high: {mem.delta_bytes / 1024:.1f}KB"
    )


@pytest.mark.performance
def test_iteration_object_memory_size() -> None:
    """
    Measure memory size of Iteration objects with varying content.
    """
    small_iter = make_iteration(num_code_blocks=1, response_size=100, stdout_size=100)
    medium_iter = make_iteration(num_code_blocks=5, response_size=1000, stdout_size=500)
    large_iter = make_iteration(num_code_blocks=10, response_size=5000, stdout_size=2000)

    small_size = measure_object_size(small_iter)
    medium_size = measure_object_size(medium_iter)
    large_size = measure_object_size(large_iter)

    # Sizes should scale approximately linearly with content
    assert small_size < medium_size < large_size

    # Large iteration should still be manageable (< 100KB)
    assert large_size < 100_000, f"Large iteration too big: {large_size / 1024:.1f}KB"


@pytest.mark.performance
def test_repl_result_locals_memory() -> None:
    """
    Profile memory usage of ReplResult.locals with many variables.
    """
    # Simulate environment with many variables
    many_locals = {f"var_{i}": {"data": f"value_{i}" * 100} for i in range(100)}

    gc.collect()

    with memory_tracker() as mem:
        for _ in range(100):
            _result = ReplResult(
                stdout="output",
                stderr="",
                locals=many_locals.copy(),  # Copy simulates snapshot
                llm_calls=[],
                execution_time=0.01,
            )

    # 100 copies of 100 variables shouldn't explode memory
    # Each copy ~50KB, 100 copies = ~5MB
    assert mem.peak_bytes < 10_000_000, (
        f"Locals memory too high: {mem.peak_bytes / 1024 / 1024:.1f}MB"
    )


@pytest.mark.performance
def test_code_block_accumulation_memory() -> None:
    """
    Profile memory when accumulating many code blocks.
    """
    gc.collect()

    with memory_tracker() as _mem:
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
                ),
            )

    # 100 code blocks should be manageable (< 1MB)
    total_size = measure_object_size(code_blocks)
    assert total_size < 1_000_000, f"Code blocks too large: {total_size / 1024:.1f}KB"


@pytest.mark.performance
def test_usage_summary_memory_scaling() -> None:
    """
    Profile memory usage of UsageSummary with many models.
    """
    from rlm.domain.models import ModelUsageSummary, UsageSummary

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
            },
        )
        sizes.append((num_models, measure_object_size(summary)))

    # Should scale linearly with model count
    # Each model adds fixed overhead
    for num_models, size in sizes:
        per_model = size / num_models
        # Each model entry should be < 1KB
        assert per_model < 1000, f"Per-model memory too high: {per_model:.0f} bytes"


@pytest.mark.performance
def test_prompt_list_memory_growth() -> None:
    """
    Profile memory growth of prompt/message history list.

    This simulates the message_history.extend() pattern in orchestrator.
    """
    gc.collect()

    with memory_tracker() as mem:
        message_history: list[dict[str, str]] = []

        for i in range(30):  # Simulate 30 iterations
            # Add assistant response
            message_history.append({"role": "assistant", "content": f"Response {i}" * 100})

            # Add user feedback (execution results)
            message_history.append({"role": "user", "content": f"Result {i}" * 100})

    # 60 messages with ~500 chars each = ~30KB
    # With dict overhead, should be < 200KB
    assert mem.delta_bytes < 200_000, (
        f"Message history memory too high: {mem.delta_bytes / 1024:.1f}KB"
    )


@pytest.mark.performance
def test_cleanup_releases_memory() -> None:
    """
    Verify that cleanup() properly releases environment memory.
    """
    from rlm.adapters.environments.local import LocalEnvironmentAdapter

    report_enabled = os.getenv("RLM_MEMORY_REPORT") == "1"
    tolerance_ratio = float(os.getenv("RLM_MEMORY_TOLERANCE_RATIO", "0.05"))
    tolerance_floor = int(os.getenv("RLM_MEMORY_TOLERANCE_FLOOR", "1000000"))
    report_path = Path(
        os.getenv("RLM_MEMORY_REPORT_PATH", ".benchmarks/memory_report_local_env.txt"),
    )

    try:
        import psutil  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        psutil = None

    snapshots: list[tuple[str, tracemalloc.Snapshot]] = []
    bytes_by_label: dict[str, int] = {}
    rss_by_label: dict[str, int | None] = {}
    child_rss_by_label: dict[str, int | None] = {}

    def _safe_rss_bytes(pid: int | None) -> int | None:
        if psutil is None or pid is None:
            return None
        try:
            proc = psutil.Process(pid)
            return int(proc.memory_info().rss)
        except Exception:  # noqa: BLE001
            return None

    env: LocalEnvironmentAdapter | None = None

    def _child_pid() -> int | None:
        if env is None:
            return None
        proc = getattr(env, "_worker_process", None)
        return getattr(proc, "pid", None) if proc is not None else None

    def _record(label: str) -> None:
        current_bytes, _ = tracemalloc.get_traced_memory()
        bytes_by_label[label] = current_bytes
        rss_by_label[label] = _safe_rss_bytes(os.getpid())
        child_rss_by_label[label] = _safe_rss_bytes(_child_pid())
        if report_enabled:
            snapshots.append((label, tracemalloc.take_snapshot()))

    gc.collect()
    tracemalloc.start()
    _record("baseline")

    context = generate_large_context(num_keys=100, value_size=10000)
    gc.collect()
    _record("after_context")

    env = LocalEnvironmentAdapter()
    gc.collect()
    _record("after_env")

    env.load_context(context)
    gc.collect()
    _record("after_load")

    # Execute some code to populate namespace
    for i in range(10):
        env.execute_code(f"data_{i} = 'x' * 10000")

    gc.collect()
    _record("after_exec")

    env.cleanup()
    gc.collect()

    _record("after_cleanup")

    env = None
    context = None
    gc.collect()
    _record("after_del")

    tracemalloc.stop()

    baseline_bytes = bytes_by_label["baseline"]
    after_context_bytes = bytes_by_label["after_context"]
    after_load_bytes = bytes_by_label["after_load"]
    after_exec_bytes = bytes_by_label["after_exec"]
    after_cleanup_bytes = bytes_by_label["after_cleanup"]
    after_del_bytes = bytes_by_label["after_del"]

    load_delta = max(0, after_load_bytes - after_context_bytes)
    exec_delta = max(0, after_exec_bytes - after_load_bytes)
    total_added = max(0, after_exec_bytes - baseline_bytes)
    bytes_released = max(0, after_exec_bytes - after_cleanup_bytes)

    # Phase 1 (load_context): ensure overhead is bounded.
    assert load_delta < 1_000_000, (
        "Context load memory too high: "
        f"{load_delta} bytes allocated"
    )

    # Phase 2 (execute_code): ensure overhead is bounded.
    assert exec_delta < 500_000, (
        "Execute memory too high: "
        f"{exec_delta} bytes allocated"
    )

    # Cleanup should release a meaningful share of execution allocations.
    if exec_delta > 0:
        assert bytes_released > exec_delta * 0.5, (
            "Cleanup didn't release enough memory after execution: "
            f"{bytes_released} bytes released of {exec_delta} added "
            f"(load={load_delta} bytes, total={total_added} bytes)"
        )

    # End-of-life: memory should return near baseline within tolerance.
    allowed_delta = max(tolerance_floor, int(baseline_bytes * tolerance_ratio))
    assert abs(after_del_bytes - baseline_bytes) <= allowed_delta, (
        "End-of-life memory not near baseline: "
        f"baseline={baseline_bytes} bytes, after_del={after_del_bytes} bytes, "
        f"allowed_delta={allowed_delta} bytes"
    )

    if report_enabled:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [
            "Memory report: test_cleanup_releases_memory",
            "",
            "Stage bytes:",
        ]
        for label, _snapshot in snapshots:
            rss = rss_by_label.get(label)
            child_rss = child_rss_by_label.get(label)
            rss_text = f"{rss} bytes" if isinstance(rss, int) else "n/a"
            child_text = f"{child_rss} bytes" if isinstance(child_rss, int) else "n/a"
            lines.append(
                f"- {label}: {bytes_by_label[label]} bytes "
                f"(rss={rss_text}, worker_rss={child_text})",
            )
        lines.append("")
        lines.append("Top allocation diffs:")
        for (before_label, before_snap), (after_label, after_snap) in zip(
            snapshots,
            snapshots[1:],
        ):
            lines.append(f"{before_label} -> {after_label}")
            stats = after_snap.compare_to(before_snap, "lineno")[:10]
            if not stats:
                lines.append("  (no diffs)")
                continue
            for stat in stats:
                frame = stat.traceback[0]
                size_kb = stat.size_diff / 1024
                lines.append(
                    f"  {size_kb:8.1f} KiB {stat.count_diff:+} "
                    f"{frame.filename}:{frame.lineno}",
                )
            lines.append("")
        report_path.write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.performance
def test_serialization_no_memory_leak() -> None:
    """
    Verify serialize_value doesn't leak memory over many calls.
    """
    data = generate_deep_nested_data(depth=10, width=3)

    gc.collect()
    gc.collect()

    # Run many serializations
    for _ in range(1000):
        _ = serialize_value(data)

    gc.collect()
    gc.collect()

    # Run more and check for growth
    with memory_tracker() as mem:
        for _ in range(1000):
            _ = serialize_value(data)

    # Memory should be stable (not growing unbounded)
    # Allow some fluctuation but not significant growth
    assert mem.delta_bytes < 100_000, f"Possible memory leak: {mem.delta_bytes / 1024:.1f}KB growth"
