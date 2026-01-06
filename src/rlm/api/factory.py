from __future__ import annotations

from typing import Any

from rlm.api.rlm import RLM
from rlm.application.config import EnvironmentName, RLMConfig
from rlm.domain.ports import LLMPort


def create_rlm(
    llm: LLMPort,
    *,
    environment: EnvironmentName = "local",
    environment_kwargs: dict[str, Any] | None = None,
    max_depth: int = 1,
    max_iterations: int = 30,
    verbose: bool = False,
) -> RLM:
    """Convenience factory for the public `RLM` facade."""
    return RLM(
        llm,
        environment=environment,
        environment_kwargs=environment_kwargs,
        max_depth=max_depth,
        max_iterations=max_iterations,
        verbose=verbose,
    )


def create_rlm_from_config(config: RLMConfig, *, llm: LLMPort | None = None) -> RLM:
    """
    Construct an `RLM` from config.

    Phase 1 requires passing a concrete `LLMPort` (provider registries arrive in Phase 4).
    """
    if llm is None:
        raise NotImplementedError(
            "Phase 1 factory requires an explicit `llm` instance. "
            "Provider selection by `LLMConfig` will be implemented in Phase 4 adapters."
        )

    return create_rlm(
        llm,
        environment=config.env.environment,
        environment_kwargs=config.env.environment_kwargs,
        max_depth=config.max_depth,
        max_iterations=config.max_iterations,
        verbose=config.verbose,
    )
