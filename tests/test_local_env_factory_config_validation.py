from __future__ import annotations

import pytest


@pytest.mark.unit
def test_default_legacy_environment_factory_local_passes_broker_and_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.api.rlm import _default_legacy_environment_factory
    from tests.fakes_ports import InMemoryBroker, QueueLLM

    created: list[object] = []

    class _FakeLocalREPL:
        def __init__(
            self,
            lm_handler_address=None,
            broker=None,
            context_payload=None,
            setup_code=None,
            **kwargs,
        ) -> None:
            created.append(
                {
                    "lm_handler_address": lm_handler_address,
                    "broker": broker,
                    "context_payload": context_payload,
                    "setup_code": setup_code,
                    "kwargs": kwargs,
                }
            )

        def load_context(self, _context_payload) -> None:
            raise AssertionError("not used")

        def execute_code(self, _code: str):
            raise AssertionError("not used")

        def cleanup(self) -> None:
            return None

    import rlm._legacy.environments.local_repl as local_repl_mod

    monkeypatch.setattr(local_repl_mod, "LocalREPL", _FakeLocalREPL)

    broker = InMemoryBroker(default_llm=QueueLLM())
    factory = _default_legacy_environment_factory(
        "local",
        {"context_payload": {"x": 1}, "setup_code": "print('hi')", "extra": 123},
    )
    env = factory.build(broker, ("127.0.0.1", 12345))
    assert env is not None

    assert created, "expected LocalREPL to be constructed"
    info = created[0]
    assert info["lm_handler_address"] == ("127.0.0.1", 12345)
    assert info["broker"] is broker
    assert info["context_payload"] == {"x": 1}
    assert info["setup_code"] == "print('hi')"
    assert info["kwargs"]["extra"] == 123

    env.cleanup()
