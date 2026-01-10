from __future__ import annotations

import builtins

import pytest

from rlm.api.factory import create_rlm
from rlm.api.registries import DefaultEnvironmentRegistry
from rlm.application.config import EnvironmentConfig
from rlm.domain.errors import ExecutionError
from tests.fakes_ports import QueueLLM


@pytest.mark.unit
def test_default_environment_registry_does_not_import_modal_when_building_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Optional env adapters must not be imported unless selected.

    In particular: selecting the local env should not import the optional `modal` package.
    """

    orig_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "modal" or str(name).startswith("modal."):
            raise AssertionError("modal should not be imported when building the local environment")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    reg = DefaultEnvironmentRegistry()
    factory = reg.build(EnvironmentConfig(environment="local"))
    env = factory.build(("127.0.0.1", 0))
    env.cleanup()


@pytest.mark.unit
def test_selecting_modal_without_dependency_yields_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Selecting modal without optional deps should fail fast with a helpful message.
    """

    orig_import = builtins.__import__

    def _missing_modal(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "modal" or str(name).startswith("modal."):
            raise ImportError("No module named 'modal'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing_modal)

    # LLM won't be used because env construction fails first.
    rlm = create_rlm(
        QueueLLM(model_name="dummy", responses=[]), environment="modal", max_iterations=1
    )
    with pytest.raises(ExecutionError) as excinfo:
        rlm.completion("hello")

    cause = excinfo.value.__cause__
    assert cause is not None
    msg = str(cause)
    assert "optional dependency" in msg.lower()
    assert "pip install modal" in msg.lower()


@pytest.mark.unit
def test_selecting_prime_yields_helpful_not_implemented_error() -> None:
    rlm = create_rlm(
        QueueLLM(model_name="dummy", responses=[]), environment="prime", max_iterations=1
    )
    with pytest.raises(ExecutionError) as excinfo:
        rlm.completion("hello")

    cause = excinfo.value.__cause__
    assert cause is not None
    msg = str(cause).lower()
    assert "prime" in msg
    assert "not implemented" in msg
