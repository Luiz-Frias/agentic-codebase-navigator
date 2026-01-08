from __future__ import annotations

from typing import Any

import pytest

from rlm.api.registries import (
    DefaultEnvironmentRegistry,
    DefaultLoggerRegistry,
    DictLLMRegistry,
    ensure_docker_available,
)
from rlm.application.config import EnvironmentConfig, LLMConfig, LoggerConfig, RLMConfig
from rlm.domain.models import RunMetadata


class _NoopLLM:
    model_name = "noop"


@pytest.mark.unit
def test_llm_config_validates_backend_non_empty() -> None:
    with pytest.raises(ValueError, match="LLMConfig\\.backend"):
        LLMConfig(backend="")


@pytest.mark.unit
def test_llm_config_validates_model_name_non_empty_when_provided() -> None:
    with pytest.raises(ValueError, match="LLMConfig\\.model_name"):
        LLMConfig(backend="x", model_name="")


@pytest.mark.unit
def test_environment_config_validates_environment_name() -> None:
    with pytest.raises(ValueError, match="EnvironmentConfig\\.environment"):
        EnvironmentConfig(environment="nope")  # type: ignore[arg-type]


@pytest.mark.unit
def test_logger_config_validates_logger_name() -> None:
    with pytest.raises(ValueError, match="LoggerConfig\\.logger"):
        LoggerConfig(logger="nope")  # type: ignore[arg-type]


@pytest.mark.unit
def test_rlm_config_validates_max_iterations_and_depth() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        RLMConfig(llm=LLMConfig(backend="mock"), max_depth=-1)
    with pytest.raises(ValueError, match="max_iterations"):
        RLMConfig(llm=LLMConfig(backend="mock"), max_iterations=0)


@pytest.mark.unit
def test_dict_llm_registry_dispatches_by_backend() -> None:
    registry = DictLLMRegistry(builders={"noop": lambda cfg: _NoopLLM()})  # type: ignore[arg-type]
    llm = registry.build(LLMConfig(backend="noop"))
    assert llm.model_name == "noop"


@pytest.mark.unit
def test_dict_llm_registry_unknown_backend_has_helpful_error() -> None:
    registry = DictLLMRegistry(builders={"noop": lambda cfg: _NoopLLM()})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        registry.build(LLMConfig(backend="missing"))


@pytest.mark.unit
def test_default_logger_registry_none_returns_none() -> None:
    reg = DefaultLoggerRegistry()
    assert reg.build(LoggerConfig(logger="none")) is None


@pytest.mark.unit
def test_default_logger_registry_legacy_jsonl_requires_log_dir() -> None:
    reg = DefaultLoggerRegistry()
    with pytest.raises(ValueError, match="log_dir"):
        reg.build(LoggerConfig(logger="legacy_jsonl", logger_kwargs={}))


@pytest.mark.unit
def test_default_logger_registry_legacy_jsonl_builds_logger(tmp_path: Any) -> None:
    reg = DefaultLoggerRegistry()
    logger = reg.build(
        LoggerConfig(
            logger="legacy_jsonl",
            logger_kwargs={"log_dir": str(tmp_path), "file_name": "test"},
        )
    )
    assert logger is not None

    # Smoke: write a metadata record to prove the adapter is callable.
    logger.log_metadata(
        RunMetadata(
            root_model="dummy",
            max_depth=1,
            max_iterations=1,
            backend="dummy",
            backend_kwargs={},
            environment_type="local",
            environment_kwargs={},
            other_backends=[],
        )
    )


@pytest.mark.unit
def test_default_environment_registry_builds_local_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[object] = []

    class _FakeLocalREPL:
        def __init__(self, *args, **kwargs) -> None:
            created.append((args, kwargs))

        def load_context(self, _context_payload) -> None:
            pass

        def execute_code(self, _code: str):
            raise AssertionError("not used")

        def cleanup(self) -> None:
            pass

    import rlm._legacy.environments.local_repl as local_repl_mod

    monkeypatch.setattr(local_repl_mod, "LocalREPL", _FakeLocalREPL)

    env_reg = DefaultEnvironmentRegistry()
    env_factory = env_reg.build(EnvironmentConfig(environment="local"))
    env = env_factory.build(("127.0.0.1", 12345))

    assert created, "expected LocalREPL to be constructed"
    _args, kwargs = created[0]
    assert kwargs["lm_handler_address"] == ("127.0.0.1", 12345)
    env.cleanup()


@pytest.mark.unit
def test_ensure_docker_available_errors_when_docker_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlm.api.registries as reg_mod

    monkeypatch.setattr(reg_mod, "which", lambda _cmd: None)
    with pytest.raises(RuntimeError, match="not found on PATH"):
        ensure_docker_available(timeout_s=0.01)


@pytest.mark.unit
def test_ensure_docker_available_errors_when_daemon_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlm.api.registries as reg_mod

    monkeypatch.setattr(reg_mod, "which", lambda _cmd: "/usr/bin/docker")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(reg_mod.subprocess, "run", _boom)
    with pytest.raises(RuntimeError, match="daemon is not reachable"):
        ensure_docker_available(timeout_s=0.01)
