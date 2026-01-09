from __future__ import annotations

import pytest


@pytest.mark.unit
def test_default_legacy_environment_factory_docker_defaults_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.api.rlm import _default_legacy_environment_factory
    from tests.fakes_ports import InMemoryBroker, QueueLLM

    class _FakeDockerREPL:
        def __init__(
            self,
            image: str = "python:3.12-slim",
            lm_handler_address=None,
            broker=None,
            *,
            subprocess_timeout_s: float = 300,
            **kwargs,
        ):
            self.image = image
            self.lm_handler_address = lm_handler_address
            self.broker = broker
            self.subprocess_timeout_s = subprocess_timeout_s
            self.kwargs = kwargs

        def load_context(self, _payload):
            raise NotImplementedError

        def execute_code(self, _code):
            raise NotImplementedError

        def cleanup(self):
            return None

    import rlm._legacy.environments.docker_repl as docker_repl_mod

    monkeypatch.setattr(docker_repl_mod, "DockerREPL", _FakeDockerREPL)

    factory = _default_legacy_environment_factory("docker", {})
    broker = InMemoryBroker(default_llm=QueueLLM())
    env = factory.build(broker, ("127.0.0.1", 12345))
    assert env is not None
    assert env._env.image == "python:3.12-slim"  # type: ignore[attr-defined]
    env.cleanup()


@pytest.mark.unit
def test_default_legacy_environment_factory_docker_passes_through_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.api.rlm import _default_legacy_environment_factory
    from tests.fakes_ports import InMemoryBroker, QueueLLM

    class _FakeDockerREPL:
        def __init__(
            self,
            image: str = "python:3.12-slim",
            lm_handler_address=None,
            broker=None,
            *,
            subprocess_timeout_s: float = 300,
            **kwargs,
        ):
            self.image = image
            self.lm_handler_address = lm_handler_address
            self.broker = broker
            self.subprocess_timeout_s = subprocess_timeout_s
            self.kwargs = kwargs

        def load_context(self, _payload):
            raise NotImplementedError

        def execute_code(self, _code):
            raise NotImplementedError

        def cleanup(self):
            return None

    import rlm._legacy.environments.docker_repl as docker_repl_mod

    monkeypatch.setattr(docker_repl_mod, "DockerREPL", _FakeDockerREPL)

    factory = _default_legacy_environment_factory(
        "docker",
        {
            "image": "python:3.12-alpine",
            "subprocess_timeout_s": 12.34,
        },
    )
    broker = InMemoryBroker(default_llm=QueueLLM())
    env = factory.build(broker, ("127.0.0.1", 12345))
    assert env is not None
    assert env._env.image == "python:3.12-alpine"  # type: ignore[attr-defined]
    assert env._env.subprocess_timeout_s == 12.34  # type: ignore[attr-defined]
    assert env._env.lm_handler_address == ("127.0.0.1", 12345)  # type: ignore[attr-defined]
    assert env._env.broker is broker  # type: ignore[attr-defined]
    env.cleanup()
