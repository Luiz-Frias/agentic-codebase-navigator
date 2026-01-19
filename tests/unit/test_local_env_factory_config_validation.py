from __future__ import annotations

import pytest


@pytest.mark.unit
def test_default_environment_registry_local_passes_broker_and_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.api.registries import DefaultEnvironmentRegistry
    from rlm.application.config import EnvironmentConfig
    from tests.fakes_ports import InMemoryBroker, QueueLLM

    created: list[object] = []

    class _FakeLocalEnv:
        def __init__(
            self,
            broker=None,
            broker_address=None,
            correlation_id=None,
            context_payload=None,
            setup_code=None,
            **kwargs,
        ) -> None:
            created.append(
                {
                    "broker": broker,
                    "broker_address": broker_address,
                    "context_payload": context_payload,
                    "setup_code": setup_code,
                    "kwargs": kwargs,
                },
            )

        def load_context(self, _context_payload) -> None:
            raise AssertionError("not used")

        def execute_code(self, _code: str):
            raise AssertionError("not used")

        def cleanup(self) -> None:
            return None

    import rlm.adapters.environments.local as local_env_mod

    monkeypatch.setattr(local_env_mod, "LocalEnvironmentAdapter", _FakeLocalEnv)

    broker = InMemoryBroker(default_llm=QueueLLM())
    factory = DefaultEnvironmentRegistry().build(
        EnvironmentConfig(
            environment="local",
            environment_kwargs={
                "context_payload": {"x": 1},
                "setup_code": "print('hi')",
            },
        ),
    )
    env = factory.build(broker, ("127.0.0.1", 12345))
    assert env is not None

    assert created, "expected LocalEnvironmentAdapter to be constructed"
    info = created[0]
    assert info["broker_address"] == ("127.0.0.1", 12345)
    assert info["broker"] is broker
    assert info["context_payload"] == {"x": 1}
    assert info["setup_code"] == "print('hi')"

    env.cleanup()
