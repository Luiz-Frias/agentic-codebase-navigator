from __future__ import annotations

import pytest


@pytest.mark.unit
def test_default_environment_registry_docker_defaults_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.api.registries import DefaultEnvironmentRegistry
    from rlm.application.config import EnvironmentConfig
    from tests.fakes_ports import InMemoryBroker, QueueLLM

    created: list[dict[str, object]] = []

    class _FakeDockerEnv:
        def __init__(
            self,
            *,
            image: str = "python:3.12-slim",
            broker=None,
            broker_address=None,
            correlation_id=None,
            subprocess_timeout_s: float = 300,
            **kwargs,
        ) -> None:
            created.append(
                {
                    "image": image,
                    "broker": broker,
                    "broker_address": broker_address,
                    "correlation_id": correlation_id,
                    "subprocess_timeout_s": subprocess_timeout_s,
                    "kwargs": kwargs,
                }
            )

        def load_context(self, _payload):
            raise NotImplementedError

        def execute_code(self, _code):
            raise NotImplementedError

        def cleanup(self):
            return None

    import rlm.adapters.environments.docker as docker_mod

    monkeypatch.setattr(docker_mod, "DockerEnvironmentAdapter", _FakeDockerEnv)

    import rlm.api.registries as reg_mod

    monkeypatch.setattr(reg_mod, "ensure_docker_available", lambda *_, **__: None)

    factory = DefaultEnvironmentRegistry().build(EnvironmentConfig(environment="docker"))
    broker = InMemoryBroker(default_llm=QueueLLM())
    env = factory.build(broker, ("127.0.0.1", 12345))
    assert env is not None
    assert created and created[0]["image"] == "python:3.12-slim"
    env.cleanup()


@pytest.mark.unit
def test_default_environment_registry_docker_passes_through_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.api.registries import DefaultEnvironmentRegistry
    from rlm.application.config import EnvironmentConfig
    from tests.fakes_ports import InMemoryBroker, QueueLLM

    created: list[dict[str, object]] = []

    class _FakeDockerEnv:
        def __init__(
            self,
            *,
            image: str = "python:3.12-slim",
            broker=None,
            broker_address=None,
            correlation_id=None,
            subprocess_timeout_s: float = 300,
            **kwargs,
        ) -> None:
            created.append(
                {
                    "image": image,
                    "broker": broker,
                    "broker_address": broker_address,
                    "correlation_id": correlation_id,
                    "subprocess_timeout_s": subprocess_timeout_s,
                    "kwargs": kwargs,
                }
            )

        def load_context(self, _payload):
            raise NotImplementedError

        def execute_code(self, _code):
            raise NotImplementedError

        def cleanup(self):
            return None

    import rlm.adapters.environments.docker as docker_mod

    monkeypatch.setattr(docker_mod, "DockerEnvironmentAdapter", _FakeDockerEnv)

    import rlm.api.registries as reg_mod

    monkeypatch.setattr(reg_mod, "ensure_docker_available", lambda *_, **__: None)

    factory = DefaultEnvironmentRegistry().build(
        EnvironmentConfig(
            environment="docker",
            environment_kwargs={
                "image": "python:3.12-alpine",
                "subprocess_timeout_s": 12.34,
            },
        )
    )
    broker = InMemoryBroker(default_llm=QueueLLM())
    env = factory.build(broker, ("127.0.0.1", 12345))
    assert env is not None
    assert created
    info = created[0]
    assert info["image"] == "python:3.12-alpine"
    assert info["subprocess_timeout_s"] == 12.34
    assert info["broker_address"] == ("127.0.0.1", 12345)
    assert info["broker"] is broker
    env.cleanup()
