from __future__ import annotations

import inspect

import pytest

from rlm.application.use_cases.run_completion import (
    RunCompletionDeps,
    RunCompletionRequest,
    _build_environment,
    _infer_environment_type,
    arun_completion,
)
from rlm.domain.errors import BrokerError, ExecutionError, RLMError
from rlm.domain.ports import BrokerPort, EnvironmentPort
from tests.fakes_ports import (
    CollectingLogger,
    InMemoryBroker,
    QueueEnvironment,
    QueueLLM,
)


@pytest.mark.unit
def test_build_environment_selects_call_shape_by_signature() -> None:
    broker = InMemoryBroker(default_llm=QueueLLM(responses=["FINAL(ok)"]))
    addr = ("127.0.0.1", 1234)
    cid = "cid"
    env: EnvironmentPort = QueueEnvironment()

    class VarargsFactory:
        def __init__(self) -> None:
            self.seen: tuple[object, ...] | None = None

        def build(self, *args: object) -> EnvironmentPort:
            self.seen = args
            return env

    fv = VarargsFactory()
    assert _build_environment(fv, broker, addr, cid) is env
    assert fv.seen == (broker, addr, cid)

    class ThreeArgFactory:
        def __init__(self) -> None:
            self.seen: tuple[object, ...] | None = None

        def build(
            self,
            broker: BrokerPort,
            broker_address: tuple[str, int],
            correlation_id: str | None,
            /,
        ) -> EnvironmentPort:
            self.seen = (broker, broker_address, correlation_id)
            return env

    f3 = ThreeArgFactory()
    assert _build_environment(f3, broker, addr, cid) is env
    assert f3.seen == (broker, addr, cid)

    class TwoArgFactory:
        def __init__(self) -> None:
            self.seen: tuple[object, ...] | None = None

        def build(self, broker: BrokerPort, broker_address: tuple[str, int], /) -> EnvironmentPort:
            self.seen = (broker, broker_address)
            return env

    f2 = TwoArgFactory()
    assert _build_environment(f2, broker, addr, cid) is env
    assert f2.seen == (broker, addr)

    class OneArgFactory:
        def __init__(self) -> None:
            self.seen: tuple[object, ...] | None = None

        def build(self, broker_address: tuple[str, int], /) -> EnvironmentPort:
            self.seen = (broker_address,)
            return env

    f1 = OneArgFactory()
    assert _build_environment(f1, broker, addr, cid) is env
    assert f1.seen == (addr,)


@pytest.mark.unit
def test_build_environment_fallbacks_when_signature_introspection_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = InMemoryBroker(default_llm=QueueLLM(responses=["FINAL(ok)"]))
    addr = ("127.0.0.1", 1234)
    cid = "cid"
    env: EnvironmentPort = QueueEnvironment()

    class OneArgFactory:
        def build(self, broker_address: tuple[str, int], /) -> EnvironmentPort:
            return env

    # Force the `except (TypeError, ValueError)` path inside _build_environment.
    monkeypatch.setattr(
        inspect,
        "signature",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError()),
    )
    assert _build_environment(OneArgFactory(), broker, addr, cid) is env


@pytest.mark.unit
def test_infer_environment_type_uses_inner_env_name_when_missing_attribute() -> None:
    class DockerREPL:
        pass

    class LocalREPL:
        pass

    class Wrapper:
        environment_type = None

        def __init__(self, inner: object) -> None:
            self._env = inner

    assert _infer_environment_type(Wrapper(DockerREPL())) == "docker"
    assert _infer_environment_type(Wrapper(LocalREPL())) == "local"


@pytest.mark.unit
async def test_arun_completion_maps_broker_start_errors_to_broker_error() -> None:
    class FailingBroker(InMemoryBroker):
        def start(self) -> tuple[str, int]:
            raise RuntimeError("boom")

    llm = QueueLLM(responses=["FINAL(ok)"])
    broker = FailingBroker(default_llm=llm)

    class _Factory:
        def build(self, _broker_address: tuple[str, int], /) -> EnvironmentPort:
            return QueueEnvironment()

    deps = RunCompletionDeps(llm=llm, broker=broker, environment_factory=_Factory())
    with pytest.raises(BrokerError, match="Failed to start broker"):
        await arun_completion(RunCompletionRequest(prompt="hi"), deps=deps)


@pytest.mark.unit
async def test_arun_completion_maps_env_build_errors_to_execution_error_and_stops_broker() -> None:
    llm = QueueLLM(responses=["FINAL(ok)"])
    broker = InMemoryBroker(default_llm=llm)

    class _Factory:
        def build(self, _broker_address: tuple[str, int], /) -> EnvironmentPort:
            raise RuntimeError("env boom")

    deps = RunCompletionDeps(llm=llm, broker=broker, environment_factory=_Factory())
    with pytest.raises(ExecutionError, match="Failed to build environment"):
        await arun_completion(RunCompletionRequest(prompt="hi"), deps=deps)
    assert broker._started is False


@pytest.mark.unit
async def test_arun_completion_logs_metadata_when_logger_present() -> None:
    llm = QueueLLM(responses=["FINAL(ok)"])
    broker = InMemoryBroker(default_llm=llm)
    logger = CollectingLogger()

    class _Factory:
        def build(self, _broker_address: tuple[str, int], /) -> EnvironmentPort:
            return QueueEnvironment()

    deps = RunCompletionDeps(llm=llm, broker=broker, environment_factory=_Factory(), logger=logger)
    cc = await arun_completion(RunCompletionRequest(prompt="hi"), deps=deps)
    assert cc.response == "ok"
    assert len(logger.metadata) == 1
    assert logger.metadata[0].correlation_id is not None


@pytest.mark.unit
async def test_arun_completion_wraps_untyped_orchestrator_errors_as_rlm_error() -> None:
    llm = QueueLLM(responses=[RuntimeError("llm boom")])
    broker = InMemoryBroker(default_llm=llm)

    class _Factory:
        def build(self, _broker_address: tuple[str, int], /) -> EnvironmentPort:
            return QueueEnvironment()

    deps = RunCompletionDeps(llm=llm, broker=broker, environment_factory=_Factory())
    with pytest.raises(RLMError, match="RLM run failed"):
        await arun_completion(RunCompletionRequest(prompt="hi"), deps=deps)


@pytest.mark.unit
async def test_arun_completion_does_not_double_wrap_domain_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlm.application.use_cases.run_completion as uc

    class _ExplodingOrchestrator:
        def __init__(self, **_kwargs):  # noqa: ANN003
            pass

        async def acompletion(self, *_args, **_kwargs):  # noqa: ANN001, ANN003
            raise ExecutionError("bad context")

    monkeypatch.setattr(uc, "RLMOrchestrator", _ExplodingOrchestrator)

    llm = QueueLLM(responses=["FINAL(ok)"])
    broker = InMemoryBroker(default_llm=llm)

    class _Factory:
        def build(self, _broker_address: tuple[str, int], /) -> EnvironmentPort:
            return QueueEnvironment()

    deps = RunCompletionDeps(llm=llm, broker=broker, environment_factory=_Factory())
    with pytest.raises(ExecutionError, match="bad context"):
        await arun_completion(RunCompletionRequest(prompt="hi"), deps=deps)
