from __future__ import annotations

import math
import subprocess  # nosec B404 - required for Docker daemon health check
from dataclasses import dataclass
from shutil import which
from typing import TYPE_CHECKING, Protocol, TypeGuard

from rlm.adapters.tools import InMemoryToolRegistry
from rlm.domain.models.result import Err
from rlm.domain.models.validation import Validator
from rlm.domain.policies.timeouts import (
    DEFAULT_DOCKER_DAEMON_PROBE_TIMEOUT_S,
    DEFAULT_LOCAL_EXECUTE_TIMEOUT_CAP_S,
    DEFAULT_LOCAL_EXECUTE_TIMEOUT_S,
    MAX_LOCAL_EXECUTE_TIMEOUT_CAP_S,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from rlm.application.config import EnvironmentConfig, LLMConfig, LoggerConfig
    from rlm.application.use_cases.run_completion import EnvironmentFactory
    from rlm.domain.ports import BrokerPort, EnvironmentPort, LLMPort, LoggerPort


# =============================================================================
# Environment Kwargs Validation (Data-Driven with Validator Pattern)
# =============================================================================
#
# Per-environment validation schemas using Validator pattern from domain.
# Each schema defines: allowed_keys, per-key validators, and transformers.


def _is_positive_number(v: object) -> TypeGuard[int | float]:
    """Check if value is a positive number (int or float, not bool)."""
    return (
        not isinstance(v, bool)
        and isinstance(v, (int, float))
        and math.isfinite(float(v))
        and float(v) > 0
    )


def _is_non_negative_int(v: object) -> bool:
    """Check if value is a non-negative integer (not bool)."""
    return not isinstance(v, bool) and isinstance(v, int) and v >= 0


def _is_non_empty_string(v: object) -> bool:
    """Check if value is a non-empty string."""
    return isinstance(v, str) and bool(v.strip())


def _is_string_or_none(v: object) -> bool:
    """Check if value is a string or None."""
    return v is None or isinstance(v, str)


def _is_context_payload(v: object) -> bool:
    """Check if value is a valid context payload type."""
    return v is None or isinstance(v, (str, dict, list))


def _is_import_roots(v: object) -> bool:
    """Check if value is a valid import roots collection."""
    if not isinstance(v, (set, list, tuple)):
        return False
    return all(isinstance(x, str) and x.strip() for x in v)


def _is_local_execute_timeout_cap(v: object) -> bool:
    """Check if local execute timeout cap is within the allowed max."""
    if not _is_positive_number(v):
        return False
    return float(v) <= MAX_LOCAL_EXECUTE_TIMEOUT_CAP_S


# Pre-built validators for common environment kwargs patterns
# Error messages complete the sentence: "{env} environment requires '{key}' {message}"
_positive_float_validator: Validator[object] = Validator[object]().satisfies(
    _is_positive_number, "to be a number > 0"
)
_non_negative_int_validator: Validator[object] = Validator[object]().satisfies(
    _is_non_negative_int, "to be an int >= 0"
)
_non_empty_string_validator: Validator[object] = Validator[object]().satisfies(
    _is_non_empty_string, "to be a non-empty string"
)
_string_or_none_validator: Validator[object] = Validator[object]().satisfies(
    _is_string_or_none, "to be a string when provided"
)
_context_payload_validator: Validator[object] = Validator[object]().satisfies(
    _is_context_payload, "to be one of str|dict|list when provided"
)
_import_roots_validator: Validator[object] = Validator[object]().satisfies(
    _is_import_roots, "to be a set/list/tuple of non-empty strings"
)
_local_execute_timeout_cap_validator: Validator[object] = Validator[object]().satisfies(
    _is_local_execute_timeout_cap,
    f"to be a number > 0 and <= {MAX_LOCAL_EXECUTE_TIMEOUT_CAP_S}",
)


# Environment kwargs schemas: {env_name: {allowed_keys, validators, transformers}}
_ENV_KWARGS_SCHEMAS: dict[str, dict[str, object]] = {
    "local": {
        "allowed_keys": {
            "execute_timeout_s",
            "execute_timeout_cap_s",
            "broker_timeout_s",
            "allowed_import_roots",
            "context_payload",
            "setup_code",
        },
        "validators": {
            "execute_timeout_s": _positive_float_validator,
            "execute_timeout_cap_s": _local_execute_timeout_cap_validator,
            "broker_timeout_s": _positive_float_validator,
            "allowed_import_roots": _import_roots_validator,
            "context_payload": _context_payload_validator,
            "setup_code": _string_or_none_validator,
        },
        "transformers": {
            # Convert list/tuple to set for allowed_import_roots
            "allowed_import_roots": lambda v: set(v) if isinstance(v, (list, tuple)) else v,
        },
    },
    "docker": {
        "allowed_keys": {
            "image",
            "subprocess_timeout_s",
            "proxy_http_timeout_s",
            "stop_grace_s",
            "cleanup_subprocess_timeout_s",
            "thread_join_timeout_s",
            "context_payload",
            "setup_code",
        },
        "validators": {
            "image": _non_empty_string_validator,
            "subprocess_timeout_s": _positive_float_validator,
            "proxy_http_timeout_s": _positive_float_validator,
            "stop_grace_s": _non_negative_int_validator,
            "cleanup_subprocess_timeout_s": _positive_float_validator,
            "thread_join_timeout_s": _positive_float_validator,
            "context_payload": _context_payload_validator,
            "setup_code": _string_or_none_validator,
        },
        "transformers": {},
    },
    "modal": {"allowed_keys": set(), "validators": {}, "transformers": {}},
    "prime": {"allowed_keys": set(), "validators": {}, "transformers": {}},
}


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
    Provider-specific registries/adapters arrive in future implementations.
    """

    builders: Mapping[str, Callable[[LLMConfig], LLMPort]]

    def build(self, config: LLMConfig, /) -> LLMPort:
        try:
            builder = self.builders[config.backend]
        except KeyError as e:
            raise ValueError(
                f"Unknown LLM backend {config.backend!r}. Available: {sorted(self.builders)}",
            ) from e
        return builder(config)


@dataclass(frozen=True, slots=True)
class DefaultLLMRegistry(LLMRegistry):
    """
    Default provider registry.

    Keeps optional provider dependencies behind lazy imports and provides a
    consistent place to map `LLMConfig` -> concrete `LLMPort`.
    """

    def build(self, config: LLMConfig, /) -> LLMPort:
        match config.backend:
            case "mock":
                from rlm.adapters.llm.mock import MockLLMAdapter

                return MockLLMAdapter(
                    model=config.model_name or "mock-model",
                    **config.backend_kwargs,
                )
            case "openai":
                from rlm.adapters.llm.openai import build_openai_adapter

                model = config.model_name or "gpt-5-nano"
                return build_openai_adapter(model=model, **config.backend_kwargs)
            case "anthropic":
                from rlm.adapters.llm.anthropic import build_anthropic_adapter

                anthropic_model = config.model_name
                if anthropic_model is None:
                    raise ValueError("LLM backend 'anthropic' requires LLMConfig.model_name")
                return build_anthropic_adapter(model=anthropic_model, **config.backend_kwargs)
            case "gemini":
                from rlm.adapters.llm.gemini import build_gemini_adapter

                gemini_model = config.model_name
                if gemini_model is None:
                    raise ValueError("LLM backend 'gemini' requires LLMConfig.model_name")
                return build_gemini_adapter(model=gemini_model, **config.backend_kwargs)
            case "portkey":
                from rlm.adapters.llm.portkey import build_portkey_adapter

                portkey_model = config.model_name
                if portkey_model is None:
                    raise ValueError("LLM backend 'portkey' requires LLMConfig.model_name")
                return build_portkey_adapter(model=portkey_model, **config.backend_kwargs)
            case "litellm":
                from rlm.adapters.llm.litellm import build_litellm_adapter

                litellm_model = config.model_name
                if litellm_model is None:
                    raise ValueError("LLM backend 'litellm' requires LLMConfig.model_name")
                return build_litellm_adapter(model=litellm_model, **config.backend_kwargs)
            case "azure_openai":
                from rlm.adapters.llm.azure_openai import build_azure_openai_adapter

                deployment = config.model_name
                if deployment is None:
                    raise ValueError("LLM backend 'azure_openai' requires LLMConfig.model_name")
                return build_azure_openai_adapter(deployment=deployment, **config.backend_kwargs)
            case _:
                raise ValueError(
                    f"Unknown LLM backend {config.backend!r}. "
                    "Available: ['mock','openai','anthropic','gemini','portkey','litellm','azure_openai']",
                )


@dataclass(frozen=True, slots=True)
class DefaultEnvironmentRegistry(EnvironmentRegistry):
    """
    Environment Registry:

    Builds an `EnvironmentFactory` from `EnvironmentConfig` and keeps optional
    environment dependencies behind lazy imports.
    """

    def build(self, config: EnvironmentConfig, /) -> EnvironmentFactory:
        if config.environment == "docker":
            ensure_docker_available()

        env_name = config.environment
        env_kwargs = _validate_environment_kwargs(
            env_name,
            dict(config.environment_kwargs),
            allow_legacy_keys=True,
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
                        **env_kwargs,  # type: ignore[arg-type]  # validated by _validate_environment_kwargs
                    )
                case "docker":
                    from rlm.adapters.environments.docker import (
                        DockerEnvironmentAdapter,
                    )

                    return DockerEnvironmentAdapter(
                        broker=broker,
                        broker_address=broker_address,
                        correlation_id=correlation_id,
                        **env_kwargs,  # type: ignore[arg-type]  # validated by _validate_environment_kwargs
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
            def build(self, *args: object) -> EnvironmentPort:
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
                        cid,
                        str,
                    ):
                        return _build(broker, (host, port), cid)  # type: ignore[arg-type]
                    case ((str() as host, int() as port), cid) if isinstance(cid, str):
                        return _build(None, (host, port), cid)
                    case _:
                        raise TypeError(
                            "EnvironmentFactory.build() expects (broker_address) or (broker, broker_address[, correlation_id])",
                        )

        return _Factory()  # type: ignore[return-value]  # _Factory implements EnvironmentFactory protocol


def _validate_environment_kwargs(
    env: str,
    kwargs: dict[str, object],
    /,
    *,
    allow_legacy_keys: bool,
) -> dict[str, object]:
    """
    Validate and normalize environment-specific kwargs using data-driven schemas.

    This intentionally lives in the composition root layer (api) because it:
    - is boundary validation (user-provided config)
    - maps directly to adapter constructor kwargs

    Uses the Validator pattern from domain for composable field validation.
    """
    if allow_legacy_keys:
        # Historical key: used when environments were wired via `_legacy`.
        kwargs.pop("lm_handler_address", None)

    # Get schema for this environment
    schema = _ENV_KWARGS_SCHEMAS.get(env)
    if schema is None:
        raise ValueError(f"Unknown environment: {env!r}")

    allowed_keys = schema["allowed_keys"]
    validators = schema["validators"]
    transformers = schema["transformers"]

    # Type narrow for pyright
    if not isinstance(allowed_keys, set):
        raise TypeError("Schema allowed_keys must be a set")
    if not isinstance(validators, dict):
        raise TypeError("Schema validators must be a dict")
    if not isinstance(transformers, dict):
        raise TypeError("Schema transformers must be a dict")

    # Check for unknown keys
    unknown = set(kwargs) - allowed_keys
    if unknown:
        # Special message for environments that don't accept any kwargs
        if not allowed_keys:
            raise ValueError(
                f"{env} environment does not accept kwargs currently (got {sorted(kwargs)})"
            )
        raise ValueError(
            f"Unknown {env} environment kwargs: {sorted(unknown)}. Allowed: {sorted(allowed_keys)}"
        )

    # Validate and transform each provided kwarg
    out: dict[str, object] = {}
    for key, value in kwargs.items():
        # Validate using schema validator if present
        validator = validators.get(key)
        if validator is not None and isinstance(validator, Validator):
            result = validator.validate_to_result(value)
            match result:
                case Err(error=e):
                    raise ValueError(f"{env} environment requires {key!r} {e}")
                case _:
                    pass  # Ok - validation passed

        # Apply transformer if present (e.g., list → set conversion)
        transformer = transformers.get(key)
        transformed_value = (
            transformer(value) if transformer is not None and callable(transformer) else value
        )

        out[key] = transformed_value

    if env == "local":
        effective_execute_timeout = out.get("execute_timeout_s", DEFAULT_LOCAL_EXECUTE_TIMEOUT_S)
        effective_cap = out.get("execute_timeout_cap_s", DEFAULT_LOCAL_EXECUTE_TIMEOUT_CAP_S)
        if (
            isinstance(effective_execute_timeout, (int, float))
            and isinstance(effective_cap, (int, float))
            and float(effective_execute_timeout) > float(effective_cap)
        ):
            raise ValueError(
                "local environment requires 'execute_timeout_s' to be <= "
                f"'execute_timeout_cap_s' ({effective_cap}s)",
            )

    return out


@dataclass(frozen=True, slots=True)
class DefaultLoggerRegistry(LoggerRegistry):
    """
    Logger Registry:

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
                        "LoggerConfig.logger_kwargs['file_name'] must be a non-empty string",
                    )

                rotate_per_run = config.logger_kwargs.get("rotate_per_run", True)
                if not isinstance(rotate_per_run, bool):
                    raise ValueError(
                        "LoggerConfig.logger_kwargs['rotate_per_run'] must be a bool when provided",
                    )

                from rlm.adapters.logger.jsonl import JsonlLoggerAdapter

                return JsonlLoggerAdapter(
                    log_dir=log_dir,
                    file_name=file_name,
                    rotate_per_run=rotate_per_run,
                )
            case "console":
                enabled = config.logger_kwargs.get("enabled", True)
                if not isinstance(enabled, bool):
                    raise ValueError(
                        "LoggerConfig.logger_kwargs['enabled'] must be a bool when provided",
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
            "Install Docker Desktop (macOS) or the Docker Engine (Linux) and retry.",
        )
    try:
        subprocess.run(  # nosec B603 B607 - safe list-form command, Docker CLI health check
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )
    except Exception as e:
        raise RuntimeError(
            "Docker environment selected but the Docker daemon is not reachable. "
            "Make sure Docker is running (e.g., Docker Desktop) and retry.",
        ) from e


# -----------------------------------------------------------------------------
# Tool Registry (Agent Capabilities)
# -----------------------------------------------------------------------------
# Re-export for convenience. The InMemoryToolRegistry is the default
# implementation of ToolRegistryPort.

__all__ = [
    "DefaultEnvironmentRegistry",
    "DefaultLLMRegistry",
    "DefaultLoggerRegistry",
    "DictLLMRegistry",
    "EnvironmentRegistry",
    "InMemoryToolRegistry",
    "LLMRegistry",
    "LoggerRegistry",
    "ensure_docker_available",
]
