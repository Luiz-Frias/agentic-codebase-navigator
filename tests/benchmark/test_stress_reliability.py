"""
Reliability and concurrency benchmarks using pytest-benchmark.
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
from tests.benchmark.bench_utils import run_pedantic_once
from tests.performance.perf_utils import BenchmarkEnvironment, BenchmarkLLM


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


@pytest.mark.benchmark
def test_concurrent_broker_requests(benchmark) -> None:
    def run():
        llm = SlowLLM(delay_seconds=0.01)
        broker = TcpBrokerAdapter(llm)
        broker.start()
        results = []
        errors = []

        def make_request(i: int) -> tuple[int, str]:
            try:
                cc = broker.complete(LLMRequest(prompt=f"Request {i}"))
                return (i, cc.response)
            except Exception as exc:  # pragma: no cover - defensive
                return (i, f"Error: {exc}")

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

        return results, errors

    results, errors = run_pedantic_once(benchmark, run)
    assert len(errors) == 0
    assert len(results) == 50


@pytest.mark.benchmark
def test_broker_batched_stress(benchmark) -> None:
    def run():
        llm = SlowLLM(delay_seconds=0.005)
        broker = TcpBrokerAdapter(
            llm,
            timeouts=BrokerTimeouts(batched_completion_timeout_s=5.0),
        )
        broker.start()

        try:
            prompts = [f"Prompt {i}" for i in range(100)]
            request = BatchedLLMRequest(prompts=prompts, model=None)

            start = time.perf_counter()
            results = broker.complete_batched(request)
            elapsed = time.perf_counter() - start
            return len(results), elapsed
        finally:
            broker.stop()

    result_count, elapsed = run_pedantic_once(benchmark, run)
    assert result_count == 100
    benchmark.extra_info["elapsed_seconds"] = elapsed


@pytest.mark.benchmark
def test_orchestrator_high_iteration_count(benchmark) -> None:
    def run():
        llm = BenchmarkLLM(include_final=True, final_after=50)
        env = BenchmarkEnvironment()
        orchestrator = RLMOrchestrator(llm=llm, environment=env)

        start = time.perf_counter()
        result = orchestrator.completion(prompt="Run many iterations", max_iterations=60)
        elapsed = time.perf_counter() - start
        return result.response, elapsed

    response, elapsed = run_pedantic_once(benchmark, run)
    assert "done after 50 iterations" in response
    benchmark.extra_info["elapsed_seconds"] = elapsed


@pytest.mark.benchmark
def test_orchestrator_handles_llm_failure(benchmark) -> None:
    def run() -> str:
        llm = FailingLLM(fail_after=3)
        env = BenchmarkEnvironment()
        orchestrator = RLMOrchestrator(llm=llm, environment=env)

        with pytest.raises(RuntimeError) as exc_info:
            orchestrator.completion(prompt="This will fail", max_iterations=10)
        return str(exc_info.value)

    error_message = run_pedantic_once(benchmark, run)
    assert "Simulated failure" in error_message


@pytest.mark.benchmark
def test_orchestrator_handles_environment_failure(benchmark) -> None:
    def run() -> str:
        llm = BenchmarkLLM()
        env = FailingEnvironment(fail_after=2)
        orchestrator = RLMOrchestrator(llm=llm, environment=env)

        with pytest.raises(RuntimeError) as exc_info:
            orchestrator.completion(prompt="This will fail during execution", max_iterations=10)
        return str(exc_info.value)

    error_message = run_pedantic_once(benchmark, run)
    assert "Simulated execution failure" in error_message


@pytest.mark.benchmark
def test_broker_cleanup_on_stop(benchmark) -> None:
    def run() -> str:
        llm = SlowLLM(delay_seconds=0.01)
        broker = TcpBrokerAdapter(llm)

        for _ in range(5):
            broker.start()
            broker.complete(LLMRequest(prompt="test"))
            broker.stop()

        broker.start()
        result = broker.complete(LLMRequest(prompt="final"))
        broker.stop()
        return result.response

    response = run_pedantic_once(benchmark, run)
    assert response is not None


@pytest.mark.benchmark
def test_multiple_orchestrators_parallel(benchmark) -> None:
    def run():
        results = []
        errors = []

        def run_orchestrator(idx: int) -> str:
            llm = BenchmarkLLM(include_final=True, final_after=3)
            env = BenchmarkEnvironment()
            orchestrator = RLMOrchestrator(llm=llm, environment=env)
            result = orchestrator.completion(prompt=f"Parallel task {idx}", max_iterations=5)
            return result.response

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_orchestrator, i) for i in range(10)]

            for future in as_completed(futures):
                try:
                    response = future.result()
                    results.append(response)
                except Exception as exc:  # pragma: no cover - defensive
                    errors.append(str(exc))

        return results, errors

    results, errors = run_pedantic_once(benchmark, run)
    assert len(errors) == 0
    assert len(results) == 10


@pytest.mark.benchmark
def test_async_orchestrator_cancellation(benchmark) -> None:
    def run() -> bool:
        llm = SlowLLM(delay_seconds=0.1)
        env = BenchmarkEnvironment()
        orchestrator = RLMOrchestrator(llm=llm, environment=env)

        async def run_with_timeout():
            return await asyncio.wait_for(
                orchestrator.acompletion(prompt="Slow task", max_iterations=100),
                timeout=0.5,
            )

        try:
            asyncio.run(run_with_timeout())
        except TimeoutError:
            return True
        return False

    cancelled = run_pedantic_once(benchmark, run)
    assert cancelled is True


@pytest.mark.benchmark
def test_thread_safety_of_usage_tracking(benchmark) -> None:
    from rlm.adapters.llm.provider_base import UsageTracker

    def run():
        tracker = UsageTracker()
        errors = []

        def record_usage(thread_id: int) -> None:
            try:
                for _i in range(100):
                    tracker.record(
                        f"model-{thread_id}",
                        calls=1,
                        input_tokens=100,
                        output_tokens=50,
                    )
                    _ = tracker.get_usage_summary()
            except Exception as exc:  # pragma: no cover - defensive
                errors.append((thread_id, str(exc)))

        threads = [threading.Thread(target=record_usage, args=(i,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        summary = tracker.get_usage_summary()
        total_calls = sum(m.total_calls for m in summary.model_usage_summaries.values())
        return errors, total_calls

    errors, total_calls = run_pedantic_once(benchmark, run)
    assert len(errors) == 0
    assert total_calls == 1000


@pytest.mark.benchmark
def test_environment_cleanup_idempotent(benchmark) -> None:
    from rlm.adapters.environments.local import LocalEnvironmentAdapter

    def run() -> None:
        env = LocalEnvironmentAdapter()
        env.load_context({"data": "test"})
        env.execute_code("x = 1")
        for _ in range(5):
            env.cleanup()

    run_pedantic_once(benchmark, run)


@pytest.mark.benchmark
def test_broker_timeout_handling(benchmark) -> None:
    class VerySlowLLM(SlowLLM):
        def __init__(self):
            super().__init__(delay_seconds=10.0)

    def run() -> str:
        llm = VerySlowLLM()
        broker = TcpBrokerAdapter(
            llm,
            timeouts=BrokerTimeouts(batched_completion_timeout_s=0.1),
            cancellation=CancellationPolicy(grace_timeout_s=0.05),
        )
        broker.start()
        try:
            request = BatchedLLMRequest(prompts=["slow"], model=None)
            with pytest.raises(LLMError) as exc_info:
                broker.complete_batched(request)
            return str(exc_info.value)
        finally:
            broker.stop()

    error_message = run_pedantic_once(benchmark, run)
    assert "timed out" in error_message.lower()


@pytest.mark.benchmark
def test_rapid_start_stop_cycles(benchmark) -> None:
    def run() -> str:
        llm = BenchmarkLLM()

        for _ in range(20):
            broker = TcpBrokerAdapter(llm)
            broker.start()
            broker.stop()

        broker = TcpBrokerAdapter(llm)
        broker.start()
        try:
            result = broker.complete(LLMRequest(prompt="test"))
            return result.response
        finally:
            broker.stop()

    response = run_pedantic_once(benchmark, run)
    assert response is not None


@pytest.mark.benchmark
def test_large_batch_memory_stability(benchmark) -> None:
    def run():
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
            return initial_objects, final_objects
        finally:
            broker.stop()

    initial_objects, final_objects = run_pedantic_once(benchmark, run)
    growth = final_objects - initial_objects
    benchmark.extra_info["object_growth"] = growth
