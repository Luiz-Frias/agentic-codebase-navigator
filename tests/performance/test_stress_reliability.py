"""
Reliability stress tests for RLM system behavior under load.

Tests focus on:
- Concurrent request handling
- Error recovery and graceful degradation
- Timeout behavior and cancellation
- Resource cleanup under failure conditions
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from rlm.adapters.broker.tcp import TcpBrokerAdapter
from rlm.domain.errors import LLMError
from rlm.domain.models import (
    BatchedLLMRequest,
    ChatCompletion,
    LLMRequest,
    ModelUsageSummary,
    ReplResult,
    UsageSummary,
)
from rlm.domain.policies.timeouts import BrokerTimeouts, CancellationPolicy
from rlm.domain.ports import EnvironmentPort, LLMPort
from rlm.domain.services.rlm_orchestrator import RLMOrchestrator

from .perf_utils import BenchmarkEnvironment, BenchmarkLLM


class SlowLLM(LLMPort):
    """LLM that simulates slow responses."""

    def __init__(self, *, delay_seconds: float = 0.1, model_name: str = "slow-llm") -> None:
        self._delay = delay_seconds
        self._model_name = model_name
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        time.sleep(self._delay)
        self._call_count += 1
        return ChatCompletion(
            root_model=self._model_name,
            prompt=request.prompt,
            response=f"Response after {self._delay}s delay",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=self._delay,
        )

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        await asyncio.sleep(self._delay)
        self._call_count += 1
        return ChatCompletion(
            root_model=self._model_name,
            prompt=request.prompt,
            response=f"Async response after {self._delay}s delay",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=self._delay,
        )

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                self._model_name: ModelUsageSummary(
                    total_calls=self._call_count,
                    total_input_tokens=0,
                    total_output_tokens=0,
                )
            }
        )

    def get_last_usage(self) -> UsageSummary:
        return self.get_usage_summary()


class FailingLLM(LLMPort):
    """LLM that fails after N successful calls."""

    def __init__(
        self,
        *,
        fail_after: int = 5,
        error_type: type[Exception] = RuntimeError,
        model_name: str = "failing-llm",
    ) -> None:
        self._fail_after = fail_after
        self._error_type = error_type
        self._model_name = model_name
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        self._call_count += 1
        if self._call_count > self._fail_after:
            raise self._error_type(f"Simulated failure after {self._fail_after} calls")

        return ChatCompletion(
            root_model=self._model_name,
            prompt=request.prompt,
            response=f"Success {self._call_count}",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        return self.complete(request)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})

    def get_last_usage(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})


class FailingEnvironment(EnvironmentPort):
    """Environment that fails after N executions."""

    def __init__(self, *, fail_after: int = 3) -> None:
        self._fail_after = fail_after
        self._exec_count = 0

    def load_context(self, context_payload, /) -> None:
        pass

    def execute_code(self, code: str, /) -> ReplResult:
        self._exec_count += 1
        if self._exec_count > self._fail_after:
            raise RuntimeError(f"Simulated execution failure after {self._fail_after} executions")

        return ReplResult(
            stdout=f"Execution {self._exec_count}",
            stderr="",
            locals={},
            llm_calls=[],
            execution_time=0.0,
        )

    def cleanup(self) -> None:
        self._exec_count = 0


@pytest.mark.performance
def test_concurrent_broker_requests() -> None:
    """
    Test broker handles many concurrent requests.
    """
    llm = SlowLLM(delay_seconds=0.01)
    broker = TcpBrokerAdapter(llm)
    broker.start()

    results = []
    errors = []

    def make_request(i: int) -> tuple[int, str]:
        try:
            cc = broker.complete(LLMRequest(prompt=f"Request {i}"))
            return (i, cc.response)
        except Exception as e:
            return (i, f"Error: {e}")

    try:
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request, i) for i in range(50)]

            for future in as_completed(futures):
                idx, result = future.result()
                if result.startswith("Error:"):
                    errors.append((idx, result))
                else:
                    results.append((idx, result))
    finally:
        broker.stop()

    # All requests should succeed
    assert len(errors) == 0, f"Errors occurred: {errors[:5]}"
    assert len(results) == 50


@pytest.mark.performance
def test_broker_batched_stress() -> None:
    """
    Stress test batched request handling.
    """
    llm = SlowLLM(delay_seconds=0.005)
    broker = TcpBrokerAdapter(
        llm,
        timeouts=BrokerTimeouts(batched_completion_timeout_s=5.0),
    )
    broker.start()

    try:
        # Large batch
        prompts = [f"Prompt {i}" for i in range(100)]
        request = BatchedLLMRequest(prompts=prompts, model=None)

        start = time.perf_counter()
        results = broker.complete_batched(request)
        elapsed = time.perf_counter() - start

        assert len(results) == 100

        # With concurrent execution, should be faster than sequential
        # 100 * 0.005s = 0.5s sequential, should be < 0.5s with concurrency
        assert elapsed < 0.5, f"Batched requests too slow: {elapsed:.2f}s"

    finally:
        broker.stop()


@pytest.mark.performance
def test_orchestrator_high_iteration_count() -> None:
    """
    Test orchestrator stability with high iteration counts.
    """
    llm = BenchmarkLLM(include_final=True, final_after=50)
    env = BenchmarkEnvironment()

    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    start = time.perf_counter()
    result = orchestrator.completion(
        prompt="Run many iterations",
        max_iterations=60,
    )
    elapsed = time.perf_counter() - start

    assert "done after 50 iterations" in result.response

    # Even 50 iterations should complete in reasonable time
    assert elapsed < 2.0, f"High iteration count too slow: {elapsed:.2f}s"


@pytest.mark.performance
def test_orchestrator_handles_llm_failure() -> None:
    """
    Test orchestrator behavior when LLM fails mid-run.
    """
    llm = FailingLLM(fail_after=3)
    env = BenchmarkEnvironment()

    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    with pytest.raises(RuntimeError) as exc_info:
        orchestrator.completion(
            prompt="This will fail",
            max_iterations=10,
        )

    assert "Simulated failure" in str(exc_info.value)


@pytest.mark.performance
def test_orchestrator_handles_environment_failure() -> None:
    """
    Test orchestrator behavior when environment fails.
    """
    llm = BenchmarkLLM()  # Will produce code blocks
    env = FailingEnvironment(fail_after=2)

    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    with pytest.raises(RuntimeError) as exc_info:
        orchestrator.completion(
            prompt="This will fail during execution",
            max_iterations=10,
        )

    assert "Simulated execution failure" in str(exc_info.value)


@pytest.mark.performance
def test_broker_cleanup_on_stop() -> None:
    """
    Test that broker properly cleans up resources on stop.
    """
    llm = SlowLLM(delay_seconds=0.01)
    broker = TcpBrokerAdapter(llm)

    # Start and stop multiple times
    for _ in range(5):
        broker.start()
        broker.complete(LLMRequest(prompt="test"))
        broker.stop()

    # Final verification
    broker.start()
    result = broker.complete(LLMRequest(prompt="final"))
    broker.stop()

    assert result.response is not None


@pytest.mark.performance
def test_multiple_orchestrators_parallel() -> None:
    """
    Test running multiple orchestrators in parallel.
    """
    results = []
    errors = []

    def run_orchestrator(idx: int) -> str:
        llm = BenchmarkLLM(include_final=True, final_after=3)
        env = BenchmarkEnvironment()
        orchestrator = RLMOrchestrator(llm=llm, environment=env)

        result = orchestrator.completion(
            prompt=f"Parallel task {idx}",
            max_iterations=5,
        )
        return result.response

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_orchestrator, i) for i in range(10)]

        for future in as_completed(futures):
            try:
                response = future.result()
                results.append(response)
            except Exception as e:
                errors.append(str(e))

    assert len(errors) == 0, f"Errors: {errors}"
    assert len(results) == 10


@pytest.mark.performance
async def test_async_orchestrator_cancellation() -> None:
    """
    Test async orchestrator handles cancellation properly.
    """
    llm = SlowLLM(delay_seconds=0.1)
    env = BenchmarkEnvironment()
    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    async def run_with_timeout():
        return await asyncio.wait_for(
            orchestrator.acompletion(prompt="Slow task", max_iterations=100),
            timeout=0.5,
        )

    with pytest.raises(asyncio.TimeoutError):
        await run_with_timeout()


@pytest.mark.performance
def test_thread_safety_of_usage_tracking() -> None:
    """
    Verify usage tracking is thread-safe under concurrent access.
    """
    from rlm.adapters.llm.provider_base import UsageTracker

    tracker = UsageTracker()
    errors = []

    def record_usage(thread_id: int):
        try:
            for _i in range(100):
                tracker.record(
                    f"model-{thread_id}",
                    calls=1,
                    input_tokens=100,
                    output_tokens=50,
                )
                # Also read while writing
                _ = tracker.get_usage_summary()
        except Exception as e:
            errors.append((thread_id, str(e)))

    threads = [threading.Thread(target=record_usage, args=(i,)) for i in range(10)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread safety errors: {errors}"

    # Verify all records were captured
    summary = tracker.get_usage_summary()
    total_calls = sum(m.total_calls for m in summary.model_usage_summaries.values())
    assert total_calls == 1000  # 10 threads * 100 calls


@pytest.mark.performance
def test_environment_cleanup_idempotent() -> None:
    """
    Verify environment cleanup can be called multiple times safely.
    """
    from rlm.adapters.environments.local import LocalEnvironmentAdapter

    env = LocalEnvironmentAdapter()
    env.load_context({"data": "test"})
    env.execute_code("x = 1")

    # Call cleanup multiple times
    for _ in range(5):
        env.cleanup()

    # Should not raise


@pytest.mark.performance
def test_broker_timeout_handling() -> None:
    """
    Test broker timeout behavior with slow LLM.
    """

    class VerySlowLLM(SlowLLM):
        def __init__(self):
            super().__init__(delay_seconds=10.0)  # Very slow

    llm = VerySlowLLM()
    broker = TcpBrokerAdapter(
        llm,
        timeouts=BrokerTimeouts(batched_completion_timeout_s=0.1),
        cancellation=CancellationPolicy(grace_timeout_s=0.05),
    )
    broker.start()

    try:
        request = BatchedLLMRequest(prompts=["slow"], model=None)

        # Broker wraps TimeoutError as LLMError
        with pytest.raises(LLMError) as exc_info:
            broker.complete_batched(request)
        assert "timed out" in str(exc_info.value).lower()
    finally:
        broker.stop()


@pytest.mark.performance
def test_rapid_start_stop_cycles() -> None:
    """
    Test broker handles rapid start/stop cycles without resource leaks.
    """
    llm = BenchmarkLLM()

    for _ in range(20):
        broker = TcpBrokerAdapter(llm)
        broker.start()
        broker.stop()

    # Final cycle should work normally
    broker = TcpBrokerAdapter(llm)
    broker.start()
    try:
        result = broker.complete(LLMRequest(prompt="test"))
        assert result.response is not None
    finally:
        broker.stop()


@pytest.mark.performance
def test_large_batch_memory_stability() -> None:
    """
    Test that large batched requests don't cause memory issues.
    """
    import gc

    llm = BenchmarkLLM()
    broker = TcpBrokerAdapter(
        llm,
        timeouts=BrokerTimeouts(batched_completion_timeout_s=30.0),
    )
    broker.start()

    gc.collect()
    initial_objects = len(gc.get_objects())

    try:
        for _ in range(5):
            prompts = [f"Prompt {i}" for i in range(100)]
            request = BatchedLLMRequest(prompts=prompts, model=None)
            results = broker.complete_batched(request)
            assert len(results) == 100

        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count shouldn't grow significantly
        growth = final_objects - initial_objects
        # Allow some growth but not unbounded
        assert growth < 10000, f"Object count grew by {growth}"

    finally:
        broker.stop()
