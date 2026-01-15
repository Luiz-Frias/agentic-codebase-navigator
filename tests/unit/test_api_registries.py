from __future__ import annotations

import sys
import types

import pytest

import rlm.api.registries as reg
from rlm.api.registries import (
    DefaultEnvironmentRegistry,
    DefaultLLMRegistry,
    DefaultLoggerRegistry,
    DictLLMRegistry,
    _validate_environment_kwargs,
    ensure_docker_available,
)
from rlm.application.config import EnvironmentConfig, LLMConfig, LoggerConfig


@pytest.mark.unit
def test_dict_llm_registry_raises_helpful_error_for_unknown_backend() -> None:
    r = DictLLMRegistry(builders={})
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        r.build(LLMConfig(backend="nope"))


@pytest.mark.unit
def test_default_llm_registry_raises_for_unknown_backend() -> None:
    r = DefaultLLMRegistry()
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        r.build(LLMConfig(backend="nope"))


@pytest.mark.unit
def test_validate_environment_kwargs_prunes_legacy_key_when_allowed() -> None:
    out = _validate_environment_kwargs(
        "local", {"lm_handler_address": "127.0.0.1:1"}, allow_legacy_keys=True
    )
    assert out == {}


@pytest.mark.unit
def test_validate_environment_kwargs_local_validations() -> None:
    with pytest.raises(ValueError, match="Unknown local environment kwargs"):
        _validate_environment_kwargs("local", {"nope": 1}, allow_legacy_keys=False)

    with pytest.raises(ValueError, match="allowed_import_roots.*set/list/tuple"):
        _validate_environment_kwargs(
            "local", {"allowed_import_roots": "nope"}, allow_legacy_keys=False
        )

    with pytest.raises(ValueError, match="allowed_import_roots.*non-empty strings"):
        _validate_environment_kwargs(
            "local", {"allowed_import_roots": {"", "x"}}, allow_legacy_keys=False
        )

    out_roots = _validate_environment_kwargs(
        "local", {"allowed_import_roots": ["math"]}, allow_legacy_keys=False
    )
    assert out_roots["allowed_import_roots"] == {"math"}

    out = _validate_environment_kwargs(
        "local", {"execute_timeout_s": None}, allow_legacy_keys=False
    )
    assert out["execute_timeout_s"] is None

    with pytest.raises(ValueError, match="broker_timeout_s.*number"):
        _validate_environment_kwargs("local", {"broker_timeout_s": True}, allow_legacy_keys=False)

    with pytest.raises(ValueError, match="context_payload.*one of str\\|dict\\|list"):
        _validate_environment_kwargs(
            "local", {"context_payload": object()}, allow_legacy_keys=False
        )

    with pytest.raises(ValueError, match="setup_code.*string"):
        _validate_environment_kwargs("local", {"setup_code": 123}, allow_legacy_keys=False)


@pytest.mark.unit
def test_validate_environment_kwargs_docker_and_modal_validations() -> None:
    with pytest.raises(ValueError, match="Unknown docker environment kwargs"):
        _validate_environment_kwargs("docker", {"nope": 1}, allow_legacy_keys=False)

    with pytest.raises(ValueError, match="docker environment requires 'image'"):
        _validate_environment_kwargs("docker", {"image": ""}, allow_legacy_keys=False)

    with pytest.raises(ValueError, match="stop_grace_s.*>= 0"):
        _validate_environment_kwargs("docker", {"stop_grace_s": -1}, allow_legacy_keys=False)

    with pytest.raises(ValueError, match="stop_grace_s.*int"):
        _validate_environment_kwargs("docker", {"stop_grace_s": True}, allow_legacy_keys=False)

    out = _validate_environment_kwargs(
        "docker",
        {
            "stop_grace_s": 1,
            "subprocess_timeout_s": 1,
            "proxy_http_timeout_s": 1,
            "cleanup_subprocess_timeout_s": 1,
            "thread_join_timeout_s": 1,
            "context_payload": "ctx",
            "setup_code": "print('hi')",
        },
        allow_legacy_keys=False,
    )
    assert out["stop_grace_s"] == 1
    assert out["subprocess_timeout_s"] == 1.0
    assert out["context_payload"] == "ctx"
    assert out["setup_code"] == "print('hi')"

    with pytest.raises(ValueError, match="modal environment does not accept kwargs"):
        _validate_environment_kwargs("modal", {"x": 1}, allow_legacy_keys=False)

    with pytest.raises(ValueError, match="Unknown environment"):
        _validate_environment_kwargs("nope", {}, allow_legacy_keys=False)


@pytest.mark.unit
def test_default_environment_registry_build_factory_accepts_supported_call_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure we don't touch the real Docker daemon during unit tests.
    monkeypatch.setattr(reg, "ensure_docker_available", lambda *args, **kwargs: None)

    r = DefaultEnvironmentRegistry()
    factory = r.build(EnvironmentConfig(environment="local"))

    env1 = factory.build(("127.0.0.1", 1234))
    try:
        assert getattr(env1, "environment_type", None) == "local"
    finally:
        env1.cleanup()

    env2 = factory.build(("127.0.0.1", 1234), "cid")
    try:
        assert getattr(env2, "_correlation_id", None) == "cid"
    finally:
        env2.cleanup()

    with pytest.raises(TypeError, match="EnvironmentFactory\\.build"):
        factory.build(1)  # type: ignore[arg-type]


@pytest.mark.unit
def test_default_environment_registry_defensive_unknown_environment_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bypass EnvironmentConfig + validation to hit the defensive default branch.
    monkeypatch.setattr(reg, "_validate_environment_kwargs", lambda *_args, **_kwargs: {})
    cfg = EnvironmentConfig(environment="local")
    object.__setattr__(cfg, "environment", "nope")

    factory = DefaultEnvironmentRegistry().build(cfg)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown environment"):
        factory.build(("127.0.0.1", 1))


@pytest.mark.unit
def test_default_logger_registry_validations() -> None:
    r = DefaultLoggerRegistry()

    assert r.build(LoggerConfig(logger="none")) is None

    with pytest.raises(ValueError, match="requires logger_kwargs\\['log_dir'\\]"):
        r.build(LoggerConfig(logger="jsonl", logger_kwargs={}))

    with pytest.raises(ValueError, match="file_name.*non-empty"):
        r.build(LoggerConfig(logger="jsonl", logger_kwargs={"log_dir": "x", "file_name": ""}))

    with pytest.raises(ValueError, match="rotate_per_run.*bool"):
        r.build(LoggerConfig(logger="jsonl", logger_kwargs={"log_dir": "x", "rotate_per_run": 1}))

    with pytest.raises(ValueError, match="enabled.*bool"):
        r.build(LoggerConfig(logger="console", logger_kwargs={"enabled": 1}))

    console = r.build(LoggerConfig(logger="console", logger_kwargs={"enabled": False}))
    assert console is not None
    assert console.enabled is False

    # Defensive branch: LoggerConfig validation should prevent this, so we bypass it.
    cfg = LoggerConfig(logger="none")
    object.__setattr__(cfg, "logger", "nope")
    with pytest.raises(ValueError, match="Unknown logger"):
        r.build(cfg)


@pytest.mark.unit
def test_ensure_docker_available_raises_helpful_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reg, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="docker.*not found on PATH"):
        ensure_docker_available(timeout_s=0.001)

    monkeypatch.setattr(reg, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(
        reg.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="Docker daemon is not reachable"):
        ensure_docker_available(timeout_s=0.001)


@pytest.mark.unit
def test_default_llm_registry_model_name_required_paths_can_be_hit_with_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # These branches validate model_name after import, so we stub modules to avoid optional deps.
    anthropic_mod = types.ModuleType("rlm.adapters.llm.anthropic")
    anthropic_mod.build_anthropic_adapter = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rlm.adapters.llm.anthropic", anthropic_mod)

    gemini_mod = types.ModuleType("rlm.adapters.llm.gemini")
    gemini_mod.build_gemini_adapter = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rlm.adapters.llm.gemini", gemini_mod)

    portkey_mod = types.ModuleType("rlm.adapters.llm.portkey")
    portkey_mod.build_portkey_adapter = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rlm.adapters.llm.portkey", portkey_mod)

    litellm_mod = types.ModuleType("rlm.adapters.llm.litellm")
    litellm_mod.build_litellm_adapter = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rlm.adapters.llm.litellm", litellm_mod)

    azure_mod = types.ModuleType("rlm.adapters.llm.azure_openai")
    azure_mod.build_azure_openai_adapter = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rlm.adapters.llm.azure_openai", azure_mod)

    r = DefaultLLMRegistry()
    with pytest.raises(ValueError, match="requires LLMConfig\\.model_name"):
        r.build(LLMConfig(backend="anthropic", model_name=None))
    with pytest.raises(ValueError, match="requires LLMConfig\\.model_name"):
        r.build(LLMConfig(backend="gemini", model_name=None))
    with pytest.raises(ValueError, match="requires LLMConfig\\.model_name"):
        r.build(LLMConfig(backend="portkey", model_name=None))
    with pytest.raises(ValueError, match="requires LLMConfig\\.model_name"):
        r.build(LLMConfig(backend="litellm", model_name=None))
    with pytest.raises(ValueError, match="requires LLMConfig\\.model_name"):
        r.build(LLMConfig(backend="azure_openai", model_name=None))
