from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from shutil import which
from typing import Protocol

from rlm.application.config import EnvironmentConfig, LLMConfig, LoggerConfig
from rlm.application.use_cases.run_completion import EnvironmentFactory
from rlm.domain.policies.timeouts import DEFAULT_DOCKER_DAEMON_PROBE_TIMEOUT_S
from rlm.domain.ports import BrokerPort, EnvironmentPort, LLMPort, LoggerPort
from rlm.domain.types import ContextPayload


class LLMRegistry(Protocol):
    """Select/build an `LLMPort` from `LLMConfig`."""

    def build(self, config: LLMConfig, /) -> LLMPort: ...


class EnvironmentRegistry(Protocol):
    """Select/build an `EnvironmentFactory` from `EnvironmentConfig`."""

    def build(self, config: EnvironmentConfig, /) -> EnvironmentFactory: ...


class LoggerRegistry(Protocol):
    """Select/build a `LoggerPort` (or None) from `LoggerConfig`."""

    def build(self, config: LoggerConfig, /) -> LoggerPort | None: ...


@dataclass(frozen=True, slots=True)
class DictLLMRegistry(LLMRegistry):
    """
    A tiny registry that dispatches on `LLMConfig.backend`.

    This is intentionally generic and is useful for tests and embedding.
    Provider-specific registries/adapters arrive in later phases.
    """

    builders: Mapping[str, Callable[[LLMConfig], LLMPort]]

    def build(self, config: LLMConfig, /) -> LLMPort:
        try:
            builder = self.builders[config.backend]
        except KeyError as e:
            raise ValueError(
                f"Unknown LLM backend {config.backend!r}. Available: {sorted(self.builders)}"
            ) from e
        return builder(config)


@dataclass(frozen=True, slots=True)
class DefaultLLMRegistry(LLMRegistry):
    """
    Default provider registry (Phase 4).

    Keeps optional provider dependencies behind lazy imports and provides a
    consistent place to map `LLMConfig` -> concrete `LLMPort`.
    """

    def build(self, config: LLMConfig, /) -> LLMPort:
        match config.backend:
            case "mock":
                from rlm.adapters.llm.mock import MockLLMAdapter

                return MockLLMAdapter(
                    model=config.model_name or "mock-model", **config.backend_kwargs
                )
            case "openai":
                from rlm.adapters.llm.openai import build_openai_adapter

                model = config.model_name or "gpt-4o-mini"
                return build_openai_adapter(model=model, **config.backend_kwargs)
            case "anthropic":
                from rlm.adapters.llm.anthropic import build_anthropic_adapter

                model = config.model_name
                if model is None:
                    raise ValueError("LLM backend 'anthropic' requires LLMConfig.model_name")
                return build_anthropic_adapter(model=model, **config.backend_kwargs)
            case "gemini":
                from rlm.adapters.llm.gemini import build_gemini_adapter

                model = config.model_name
                if model is None:
                    raise ValueError("LLM backend 'gemini' requires LLMConfig.model_name")
                return build_gemini_adapter(model=model, **config.backend_kwargs)
            case "portkey":
                from rlm.adapters.llm.portkey import build_portkey_adapter

                model = config.model_name
                if model is None:
                    raise ValueError("LLM backend 'portkey' requires LLMConfig.model_name")
                return build_portkey_adapter(model=model, **config.backend_kwargs)
            case "litellm":
                from rlm.adapters.llm.litellm import build_litellm_adapter

                model = config.model_name
                if model is None:
                    raise ValueError("LLM backend 'litellm' requires LLMConfig.model_name")
                return build_litellm_adapter(model=model, **config.backend_kwargs)
            case "azure_openai":
                from rlm.adapters.llm.azure_openai import build_azure_openai_adapter

                deployment = config.model_name
                if deployment is None:
                    raise ValueError("LLM backend 'azure_openai' requires LLMConfig.model_name")
                return build_azure_openai_adapter(deployment=deployment, **config.backend_kwargs)
            case _:
                raise ValueError(
                    f"Unknown LLM backend {config.backend!r}. "
                    "Available: ['mock','openai','anthropic','gemini','portkey','litellm','azure_openai']"
                )


@dataclass(frozen=True, slots=True)
class DefaultEnvironmentRegistry(EnvironmentRegistry):
    """
    Phase 05 environment registry.

    Builds an `EnvironmentFactory` from `EnvironmentConfig` and keeps optional
    environment dependencies behind lazy imports.
    """

    def build(self, config: EnvironmentConfig, /) -> EnvironmentFactory:
        if config.environment == "docker":
            ensure_docker_available()

        env_name = config.environment
        env_kwargs = _validate_environment_kwargs(
            env_name, dict(config.environment_kwargs), allow_legacy_keys=True
        )

        def _build(
            broker: BrokerPort | None,
            broker_address: tuple[str, int],
            correlation_id: str | None,
            /,
        ) -> EnvironmentPort:
            match env_name:
                case "local":
                    from rlm.adapters.environments.local import LocalEnvironmentAdapter

                    return LocalEnvironmentAdapter(
                        broker=broker,
                        broker_address=broker_address,
                        correlation_id=correlation_id,
                        **env_kwargs,
                    )
                case "docker":
                    from rlm.adapters.environments.docker import DockerEnvironmentAdapter

                    return DockerEnvironmentAdapter(
                        broker=broker,
                        broker_address=broker_address,
                        correlation_id=correlation_id,
                        **env_kwargs,
                    )
                case "modal":
                    from rlm.adapters.environments.modal import ModalEnvironmentAdapter

                    return ModalEnvironmentAdapter(**env_kwargs)
                case "prime":
                    from rlm.adapters.environments.prime import PrimeEnvironmentAdapter

                    return PrimeEnvironmentAdapter(**env_kwargs)
                case _:
                    raise ValueError(f"Unknown environment: {env_name!r}")

        class _Factory:
            def build(self, *args: object) -> object:  # noqa: ANN401 - migration-compatible facade
                """
                Build an environment for a run.

                Supported call shapes during migration:
                - build(broker_address)
                - build(broker, broker_address)
                - build(broker, broker_address, correlation_id)
                """

                match args:
                    case ((str() as host, int() as port),):
                        return _build(None, (host, port), None)
                    case (broker, (str() as host, int() as port)):
                        return _build(broker, (host, port), None)  # type: ignore[arg-type]
                    case (broker, (str() as host, int() as port), cid) if cid is None or isinstance(
                        cid, str
                    ):
                        return _build(broker, (host, port), cid)  # type: ignore[arg-type]
                    case ((str() as host, int() as port), cid) if isinstance(cid, str):
                        return _build(None, (host, port), cid)
                    case _:
                        raise TypeError(
                            "EnvironmentFactory.build() expects (broker_address) or (broker, broker_address[, correlation_id])"
                        )

        return _Factory()


def _validate_environment_kwargs(
    env: str,
    kwargs: dict[str, object],
    /,
    *,
    allow_legacy_keys: bool,
) -> dict[str, object]:
    """
    Validate and normalize environment-specific kwargs.

    This intentionally lives in the composition root layer (api) because it:
    - is boundary validation (user-provided config)
    - maps directly to adapter constructor kwargs
    """

    if allow_legacy_keys:
        # Historical key: used when environments were wired via `_legacy`.
        kwargs.pop("lm_handler_address", None)

    def _expect_str(key: str) -> str:
        v = kwargs.get(key)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"{env} environment requires {key!r} to be a non-empty string")
        return v

    def _expect_float(key: str, *, allow_none: bool = False) -> float | None:
        v = kwargs.get(key)
        if v is None and allow_none:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{env} environment requires {key!r} to be a number")
        f = float(v)
        if f <= 0:
            raise ValueError(f"{env} environment requires {key!r} to be > 0")
        return f

    def _expect_int(key: str) -> int:
        v = kwargs.get(key)
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{env} environment requires {key!r} to be an int")
        if v < 0:
            raise ValueError(f"{env} environment requires {key!r} to be >= 0")
        return v

    def _expect_setup_code() -> str | None:
        v = kwargs.get("setup_code")
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError(
                f"{env} environment requires 'setup_code' to be a string when provided"
            )
        return v

    def _expect_context_payload() -> ContextPayload | None:
        v = kwargs.get("context_payload")
        if v is None:
            return None
        if not isinstance(v, (str, dict, list)):
            raise ValueError(
                f"{env} environment requires 'context_payload' to be one of str|dict|list when provided"
            )
        return v  # type: ignore[return-value]

    match env:
        case "local":
            allowed = {
                "execute_timeout_s",
                "broker_timeout_s",
                "allowed_import_roots",
                "context_payload",
                "setup_code",
            }
            unknown = set(kwargs) - allowed
            if unknown:
                raise ValueError(
                    f"Unknown local environment kwargs: {sorted(unknown)}. Allowed: {sorted(allowed)}"
                )

            out: dict[str, object] = {}
            if "execute_timeout_s" in kwargs:
                out["execute_timeout_s"] = _expect_float("execute_timeout_s", allow_none=True)
            if "broker_timeout_s" in kwargs:
                out["broker_timeout_s"] = _expect_float("broker_timeout_s", allow_none=False)
            if "allowed_import_roots" in kwargs:
                v = kwargs.get("allowed_import_roots")
                if isinstance(v, set):
                    roots = v
                elif isinstance(v, (list, tuple)):
                    roots = set(v)
                else:
                    raise ValueError(
                        "local environment requires 'allowed_import_roots' to be a set/list/tuple of strings"
                    )
                if not all(isinstance(x, str) and x.strip() for x in roots):
                    raise ValueError(
                        "local environment requires 'allowed_import_roots' to contain only non-empty strings"
                    )
                out["allowed_import_roots"] = roots
            if (ctx := _expect_context_payload()) is not None:
                out["context_payload"] = ctx
            if (sc := _expect_setup_code()) is not None:
                out["setup_code"] = sc
            return out
        case "docker":
            allowed = {
                "image",
                "subprocess_timeout_s",
                "proxy_http_timeout_s",
                "stop_grace_s",
                "cleanup_subprocess_timeout_s",
                "thread_join_timeout_s",
                "context_payload",
                "setup_code",
            }
            unknown = set(kwargs) - allowed
            if unknown:
                raise ValueError(
                    f"Unknown docker environment kwargs: {sorted(unknown)}. Allowed: {sorted(allowed)}"
                )

            out: dict[str, object] = {}
            if "image" in kwargs:
                out["image"] = _expect_str("image")
            if "subprocess_timeout_s" in kwargs:
                out["subprocess_timeout_s"] = _expect_float(
                    "subprocess_timeout_s", allow_none=False
                )
            if "proxy_http_timeout_s" in kwargs:
                out["proxy_http_timeout_s"] = _expect_float(
                    "proxy_http_timeout_s", allow_none=False
                )
            if "stop_grace_s" in kwargs:
                out["stop_grace_s"] = _expect_int("stop_grace_s")
            if "cleanup_subprocess_timeout_s" in kwargs:
                out["cleanup_subprocess_timeout_s"] = _expect_float(
                    "cleanup_subprocess_timeout_s", allow_none=False
                )
            if "thread_join_timeout_s" in kwargs:
                out["thread_join_timeout_s"] = _expect_float(
                    "thread_join_timeout_s", allow_none=False
                )
            if (ctx := _expect_context_payload()) is not None:
                out["context_payload"] = ctx
            if (sc := _expect_setup_code()) is not None:
                out["setup_code"] = sc
            return out
        case "modal" | "prime":
            if kwargs:
                raise ValueError(
                    f"{env} environment does not accept kwargs in Phase 05 (got {sorted(kwargs)})"
                )
            return {}
        case _:
            raise ValueError(f"Unknown environment: {env!r}")


@dataclass(frozen=True, slots=True)
class DefaultLoggerRegistry(LoggerRegistry):
    """
    Phase 2 logger registry.

    Supported values:
    - logger='none': disables logging
    - logger='jsonl': JSONL logger adapter (requires `log_dir`)
    - logger='console': minimal stdout logger (optional; `enabled` flag)
    """

    def build(self, config: LoggerConfig, /) -> LoggerPort | None:
        match config.logger:
            case "none":
                return None
            case "jsonl":
                log_dir = config.logger_kwargs.get("log_dir")
                if not isinstance(log_dir, str) or not log_dir.strip():
                    raise ValueError("LoggerConfig for 'jsonl' requires logger_kwargs['log_dir']")
                file_name = config.logger_kwargs.get("file_name", "rlm")
                if not isinstance(file_name, str) or not file_name.strip():
                    raise ValueError(
                        "LoggerConfig.logger_kwargs['file_name'] must be a non-empty string"
                    )

                rotate_per_run = config.logger_kwargs.get("rotate_per_run", True)
                if not isinstance(rotate_per_run, bool):
                    raise ValueError(
                        "LoggerConfig.logger_kwargs['rotate_per_run'] must be a bool when provided"
                    )

                from rlm.adapters.logger.jsonl import JsonlLoggerAdapter

                return JsonlLoggerAdapter(
                    log_dir=log_dir, file_name=file_name, rotate_per_run=rotate_per_run
                )
            case "console":
                enabled = config.logger_kwargs.get("enabled", True)
                if not isinstance(enabled, bool):
                    raise ValueError(
                        "LoggerConfig.logger_kwargs['enabled'] must be a bool when provided"
                    )

                from rlm.adapters.logger.console import ConsoleLoggerAdapter

                return ConsoleLoggerAdapter(enabled=enabled)
            case _:
                # Should be prevented by LoggerConfig validation, but keep a defensive
                # error here since this is a composition root.
                raise ValueError(f"Unknown logger: {config.logger!r}")


def ensure_docker_available(*, timeout_s: float = DEFAULT_DOCKER_DAEMON_PROBE_TIMEOUT_S) -> None:
    """
    Raise a helpful error if Docker isn't available.

    This is a best-effort check intended for composition root UX, not strict
    environment validation.
    """

    if which("docker") is None:
        raise RuntimeError(
            "Docker environment selected but 'docker' was not found on PATH. "
            "Install Docker Desktop (macOS) or the Docker Engine (Linux) and retry."
        )
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )
    except Exception as e:
        raise RuntimeError(
            "Docker environment selected but the Docker daemon is not reachable. "
            "Make sure Docker is running (e.g., Docker Desktop) and retry."
        ) from e
