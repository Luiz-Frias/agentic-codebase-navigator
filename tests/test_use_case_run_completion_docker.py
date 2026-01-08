from __future__ import annotations

import pytest

from rlm._legacy.environments.docker_repl import DockerREPL
from rlm.adapters.legacy.broker import LegacyBrokerAdapter
from rlm.adapters.legacy.environment import LegacyEnvironmentAdapter
from rlm.application.use_cases.run_completion import (
    RunCompletionDeps,
    RunCompletionRequest,
    run_completion,
)
from rlm.domain.ports import BrokerPort, EnvironmentPort
from rlm.domain.types import ContextPayload
from tests.fakes_ports import QueueLLM


class _SpyBroker(BrokerPort):
    def __init__(self, inner: BrokerPort) -> None:
        self._inner = inner
        self.started = False
        self.stopped = False

    def register_llm(self, model_name: str, llm, /) -> None:  # type: ignore[override]
        return self._inner.register_llm(model_name, llm)

    def start(self) -> tuple[str, int]:
        self.started = True
        return self._inner.start()

    def stop(self) -> None:
        self.stopped = True
        return self._inner.stop()

    def complete(self, request, /):  # type: ignore[override]
        return self._inner.complete(request)

    def complete_batched(self, request, /):  # type: ignore[override]
        return self._inner.complete_batched(request)

    def get_usage_summary(self):  # type: ignore[override]
        return self._inner.get_usage_summary()


class _WrappedEnv(EnvironmentPort):
    def __init__(self, inner: EnvironmentPort) -> None:
        self._inner = inner
        self.cleaned = False

    def load_context(self, context_payload: ContextPayload, /) -> None:
        return self._inner.load_context(context_payload)

    def execute_code(self, code: str, /):
        return self._inner.execute_code(code)

    def cleanup(self) -> None:
        self.cleaned = True
        return self._inner.cleanup()


class _DockerEnvFactory:
    def __init__(self, *, image: str = "python:3.12-slim") -> None:
        self._image = image
        self.last_env: _WrappedEnv | None = None

    def build(self, broker_address: tuple[str, int], /) -> EnvironmentPort:
        env = DockerREPL(image=self._image, lm_handler_address=broker_address)
        wrapped = _WrappedEnv(LegacyEnvironmentAdapter(env))
        self.last_env = wrapped
        return wrapped


@pytest.mark.integration
@pytest.mark.docker
def test_run_completion_use_case_runs_docker_env_and_cleans_up() -> None:
    llm = QueueLLM(
        responses=[
            "```repl\nprint('HELLO_DOCKER')\n```",
            "FINAL(ok)",
        ]
    )
    broker = _SpyBroker(LegacyBrokerAdapter(llm))
    env_factory = _DockerEnvFactory(image="python:3.12-slim")
    deps = RunCompletionDeps(llm=llm, broker=broker, environment_factory=env_factory, logger=None)
    req = RunCompletionRequest(prompt="hello", max_depth=1, max_iterations=3)

    try:
        cc = run_completion(req, deps=deps)
    except RuntimeError as e:
        # Docker can be present but image pulls may fail in restricted environments.
        if "Failed to start container" in str(e):
            pytest.skip(str(e))
        raise

    assert cc.response == "ok"
    assert broker.started is True
    assert broker.stopped is True
    assert env_factory.last_env is not None
    assert env_factory.last_env.cleaned is True
