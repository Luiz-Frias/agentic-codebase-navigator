from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EnvironmentName = Literal["local", "modal", "docker", "prime"]
LoggerName = Literal["none", "legacy_jsonl"]


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """
    Minimal LLM configuration (Phase 1).

    In later phases this becomes validated and mapped to concrete adapters via
    registries in the composition root.
    """

    backend: str
    model_name: str | None = None
    backend_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("LLMConfig.backend must be a non-empty string")
        if self.model_name is not None and (
            not isinstance(self.model_name, str) or not self.model_name.strip()
        ):
            raise ValueError("LLMConfig.model_name must be a non-empty string when provided")
        if not isinstance(self.backend_kwargs, dict):
            raise ValueError("LLMConfig.backend_kwargs must be a dict")


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Minimal environment configuration (Phase 1)."""

    environment: EnvironmentName
    environment_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.environment not in ("local", "docker", "modal", "prime"):
            raise ValueError(
                "EnvironmentConfig.environment must be one of "
                "['local','docker','modal','prime'], "
                f"got {self.environment!r}"
            )
        if not isinstance(self.environment_kwargs, dict):
            raise ValueError("EnvironmentConfig.environment_kwargs must be a dict")


@dataclass(frozen=True, slots=True)
class LoggerConfig:
    """Logger configuration (Phase 2)."""

    logger: LoggerName = "none"
    logger_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.logger not in ("none", "legacy_jsonl"):
            raise ValueError(
                f"LoggerConfig.logger must be one of ['none','legacy_jsonl'], got {self.logger!r}"
            )
        if not isinstance(self.logger_kwargs, dict):
            raise ValueError("LoggerConfig.logger_kwargs must be a dict")


@dataclass(frozen=True, slots=True)
class RLMConfig:
    """Minimal RLM facade configuration (Phase 1)."""

    llm: LLMConfig
    env: EnvironmentConfig = field(default_factory=lambda: EnvironmentConfig(environment="local"))
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    max_depth: int = 1
    max_iterations: int = 30
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("RLMConfig.max_depth must be >= 0")
        if self.max_iterations < 1:
            raise ValueError("RLMConfig.max_iterations must be >= 1")
