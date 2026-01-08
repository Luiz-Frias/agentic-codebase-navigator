from __future__ import annotations

from typing import Any, Protocol

from rlm.api.rlm import RLM
from rlm.application.config import EnvironmentName, LLMConfig, RLMConfig
from rlm.domain.ports import LLMPort


class LLMRegistry(Protocol):
    """
    Registry for selecting/building an `LLMPort` from configuration.

    This is intentionally minimal in Phase 2; provider-specific registries and
    lazy optional-dependency adapters arrive in later phases.
    """

    def build(self, config: LLMConfig, /) -> LLMPort: ...


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


def create_rlm_from_config(
    config: RLMConfig,
    *,
    llm: LLMPort | None = None,
    llm_registry: LLMRegistry | None = None,
) -> RLM:
    """
    Construct an `RLM` from config.

    Phase 2 allows optionally providing an `llm_registry` to build an `LLMPort`
    from `config.llm`. If neither `llm` nor `llm_registry` is provided, we fail
    fast with a helpful message.
    """
    if llm is None:
        if llm_registry is None:
            raise NotImplementedError(
                "Phase 2 factory requires either an explicit `llm` instance or an "
                "`llm_registry` to build one from `RLMConfig.llm`."
            )
        llm = llm_registry.build(config.llm)

    return create_rlm(
        llm,
        environment=config.env.environment,
        environment_kwargs=config.env.environment_kwargs,
        max_depth=config.max_depth,
        max_iterations=config.max_iterations,
        verbose=config.verbose,
    )
