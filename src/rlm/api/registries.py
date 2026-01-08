from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from shutil import which
from typing import Protocol

from rlm.application.config import EnvironmentConfig, LLMConfig, LoggerConfig
from rlm.application.use_cases.run_completion import EnvironmentFactory
from rlm.domain.ports import LLMPort, LoggerPort


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
class DefaultEnvironmentRegistry(EnvironmentRegistry):
    """
    Phase 2 environment registry.

    Today this returns legacy Local/Docker REPL adapters via the same helper
    used by the `RLM` facade defaults.
    """

    def build(self, config: EnvironmentConfig, /) -> EnvironmentFactory:
        if config.environment == "docker":
            ensure_docker_available()

        # Keep this import lazy so importing the API doesn't pull legacy env code
        # unless the registry is actually used.
        from rlm.api.rlm import _default_legacy_environment_factory

        return _default_legacy_environment_factory(
            config.environment, dict(config.environment_kwargs)
        )


@dataclass(frozen=True, slots=True)
class DefaultLoggerRegistry(LoggerRegistry):
    """
    Phase 2 logger registry.

    Supported values:
    - logger='none': disables logging
    - logger='legacy_jsonl': legacy JSONL logger adapter (requires `log_dir`)
    """

    def build(self, config: LoggerConfig, /) -> LoggerPort | None:
        match config.logger:
            case "none":
                return None
            case "legacy_jsonl":
                log_dir = config.logger_kwargs.get("log_dir")
                if not isinstance(log_dir, str) or not log_dir.strip():
                    raise ValueError(
                        "LoggerConfig for 'legacy_jsonl' requires logger_kwargs['log_dir']"
                    )
                file_name = config.logger_kwargs.get("file_name", "rlm")
                if not isinstance(file_name, str) or not file_name.strip():
                    raise ValueError(
                        "LoggerConfig.logger_kwargs['file_name'] must be a non-empty string"
                    )

                from rlm._legacy.logger.rlm_logger import RLMLogger
                from rlm.adapters.legacy.logger import LegacyLoggerAdapter

                return LegacyLoggerAdapter(RLMLogger(log_dir=log_dir, file_name=file_name))
            case _:
                # Should be prevented by LoggerConfig validation, but keep a defensive
                # error here since this is a composition root.
                raise ValueError(f"Unknown logger: {config.logger!r}")


def ensure_docker_available(*, timeout_s: float = 2.0) -> None:
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
