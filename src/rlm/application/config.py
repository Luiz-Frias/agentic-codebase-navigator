from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Minimal environment configuration (Phase 1)."""

    environment: str
    environment_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RLMConfig:
    """Minimal RLM facade configuration (Phase 1)."""

    llm: LLMConfig
    env: EnvironmentConfig = field(default_factory=lambda: EnvironmentConfig(environment="local"))
    max_depth: int = 1
    max_iterations: int = 30
    verbose: bool = False
