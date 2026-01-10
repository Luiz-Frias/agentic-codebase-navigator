from __future__ import annotations

import pytest

from rlm.application.use_cases.run_completion import (
    RunCompletionDeps,
    RunCompletionRequest,
    run_completion,
)
from rlm.domain.errors import BrokerError, ExecutionError, RLMError
from rlm.domain.models import BatchedLLMRequest, ChatCompletion, LLMRequest, UsageSummary
from rlm.domain.ports import BrokerPort, EnvironmentPort, LLMPort
from rlm.domain.types import ContextPayload
from tests.fakes_ports import InMemoryBroker, QueueEnvironment, QueueLLM


class _FailingBroker(BrokerPort):
    def __init__(self, *, exc: Exception) -> None:
        self._exc = exc
        self.stopped = False

    def register_llm(self, model_name: str, llm: LLMPort, /) -> None:
        raise AssertionError("not used")

    def start(self) -> tuple[str, int]:
        raise self._exc

    def stop(self) -> None:
        self.stopped = True

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        raise AssertionError("not used")

    def complete_batched(self, request: BatchedLLMRequest, /) -> list[ChatCompletion]:
        raise AssertionError("not used")

    def get_usage_summary(self) -> UsageSummary:
        raise AssertionError("not used")


class _FailingEnvFactory:
    def __init__(self, *, exc: Exception) -> None:
        self._exc = exc

    def build(self, _broker_address: tuple[str, int], /) -> EnvironmentPort:
        raise self._exc


class _SpyEnv(QueueEnvironment):
    def __init__(self) -> None:
        super().__init__(results=[])
        self.cleaned = False

    def cleanup(self) -> None:
        self.cleaned = True


class _SpyEnvFactory:
    def __init__(self, env: _SpyEnv) -> None:
        self.env = env

    def build(self, _broker_address: tuple[str, int], /) -> EnvironmentPort:
        return self.env


@pytest.mark.unit
def test_run_completion_maps_broker_start_errors_to_broker_error() -> None:
    boom = RuntimeError("broker boom")
    broker = _FailingBroker(exc=boom)
    llm = QueueLLM(responses=["FINAL(ok)"])
    env_factory = _FailingEnvFactory(exc=AssertionError("should not build env"))

    deps = RunCompletionDeps(llm=llm, broker=broker, environment_factory=env_factory)
    req = RunCompletionRequest(prompt="hello")

    with pytest.raises(BrokerError, match="Failed to start broker") as ei:
        run_completion(req, deps=deps)
    assert isinstance(ei.value.__cause__, RuntimeError)
    assert "broker boom" in str(ei.value.__cause__)
    assert broker.stopped is False


@pytest.mark.unit
def test_run_completion_maps_env_build_errors_to_execution_error_and_stops_broker() -> None:
    llm = QueueLLM(responses=["FINAL(ok)"])
    broker = InMemoryBroker(default_llm=llm)
    env_factory = _FailingEnvFactory(exc=RuntimeError("env boom"))

    deps = RunCompletionDeps(llm=llm, broker=broker, environment_factory=env_factory)
    req = RunCompletionRequest(prompt="hello")

    with pytest.raises(ExecutionError, match="Failed to build environment") as ei:
        run_completion(req, deps=deps)
    assert isinstance(ei.value.__cause__, RuntimeError)
    assert "env boom" in str(ei.value.__cause__)
    assert broker._started is False  # broker.stop() called in finally


@pytest.mark.unit
def test_run_completion_wraps_untyped_orchestrator_errors_as_rlm_error_and_cleans_up() -> None:
    llm = QueueLLM(responses=[RuntimeError("llm boom")])
    broker = InMemoryBroker(default_llm=llm)
    env = _SpyEnv()
    env_factory = _SpyEnvFactory(env)

    deps = RunCompletionDeps(llm=llm, broker=broker, environment_factory=env_factory)
    req = RunCompletionRequest(prompt="hello")

    with pytest.raises(RLMError, match="RLM run failed") as ei:
        run_completion(req, deps=deps)
    assert isinstance(ei.value.__cause__, RuntimeError)
    assert "llm boom" in str(ei.value.__cause__)
    assert env.cleaned is True
    assert broker._started is False


@pytest.mark.unit
def test_run_completion_does_not_double_wrap_domain_errors() -> None:
    class _ExplodingEnv(QueueEnvironment):
        def load_context(self, _context_payload: ContextPayload, /) -> None:
            raise ExecutionError("bad context")

    llm = QueueLLM(responses=["FINAL(ok)"])
    broker = InMemoryBroker(default_llm=llm)

    class _Factory:
        def build(self, _broker_address: tuple[str, int], /) -> EnvironmentPort:
            return _ExplodingEnv()

    deps = RunCompletionDeps(llm=llm, broker=broker, environment_factory=_Factory())
    req = RunCompletionRequest(prompt="hello")

    with pytest.raises(ExecutionError, match="bad context"):
        run_completion(req, deps=deps)
