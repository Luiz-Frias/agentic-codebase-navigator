from __future__ import annotations

import pytest

from rlm.api.registries import DefaultEnvironmentRegistry
from rlm.application.config import EnvironmentConfig


@pytest.mark.unit
def test_environment_registry_rejects_unknown_local_kwargs() -> None:
    reg = DefaultEnvironmentRegistry()
    with pytest.raises(ValueError, match="Unknown local environment kwargs"):
        reg.build(EnvironmentConfig(environment="local", environment_kwargs={"nope": 1}))


@pytest.mark.unit
def test_environment_registry_rejects_invalid_local_allowed_import_roots_type() -> None:
    reg = DefaultEnvironmentRegistry()
    with pytest.raises(ValueError, match="allowed_import_roots"):
        reg.build(
            EnvironmentConfig(
                environment="local", environment_kwargs={"allowed_import_roots": "nope"}
            )
        )


@pytest.mark.unit
def test_environment_registry_ignores_legacy_lm_handler_address_key() -> None:
    reg = DefaultEnvironmentRegistry()
    # Should not raise (legacy key is ignored).
    reg.build(
        EnvironmentConfig(
            environment="local",
            environment_kwargs={"lm_handler_address": ("127.0.0.1", 1234)},
        )
    )


@pytest.mark.unit
def test_environment_registry_rejects_unknown_docker_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlm.api.registries as reg_mod

    # Avoid requiring Docker for this unit test.
    monkeypatch.setattr(reg_mod, "ensure_docker_available", lambda *_, **__: None)

    reg = DefaultEnvironmentRegistry()
    with pytest.raises(ValueError, match="Unknown docker environment kwargs"):
        reg.build(EnvironmentConfig(environment="docker", environment_kwargs={"nope": 1}))


@pytest.mark.unit
def test_environment_registry_rejects_invalid_docker_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlm.api.registries as reg_mod

    monkeypatch.setattr(reg_mod, "ensure_docker_available", lambda *_, **__: None)

    reg = DefaultEnvironmentRegistry()
    with pytest.raises(ValueError, match="subprocess_timeout_s"):
        reg.build(
            EnvironmentConfig(environment="docker", environment_kwargs={"subprocess_timeout_s": -1})
        )


@pytest.mark.unit
def test_environment_registry_rejects_modal_kwargs() -> None:
    reg = DefaultEnvironmentRegistry()
    with pytest.raises(ValueError, match="modal environment does not accept kwargs"):
        reg.build(EnvironmentConfig(environment="modal", environment_kwargs={"x": 1}))
