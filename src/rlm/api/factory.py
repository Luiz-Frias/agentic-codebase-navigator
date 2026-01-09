from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rlm.api.registries import (
    DefaultEnvironmentRegistry,
    DefaultLLMRegistry,
    DefaultLoggerRegistry,
    EnvironmentRegistry,
    LLMRegistry,
    LoggerRegistry,
)
from rlm.api.rlm import RLM
from rlm.application.config import EnvironmentName, RLMConfig
from rlm.application.use_cases.run_completion import EnvironmentFactory
from rlm.domain.ports import BrokerPort, LLMPort, LoggerPort


def create_rlm(
    llm: LLMPort,
    *,
    other_llms: list[LLMPort] | None = None,
    environment: EnvironmentName = "local",
    environment_kwargs: dict[str, Any] | None = None,
    max_depth: int = 1,
    max_iterations: int = 30,
    verbose: bool = False,
    broker_factory: Callable[[LLMPort], BrokerPort] | None = None,
    environment_factory: EnvironmentFactory | None = None,
    logger: LoggerPort | None = None,
    system_prompt: str | None = None,
) -> RLM:
    """Convenience factory for the public `RLM` facade."""
    return RLM(
        llm,
        other_llms=other_llms,
        environment=environment,
        environment_kwargs=environment_kwargs,
        max_depth=max_depth,
        max_iterations=max_iterations,
        verbose=verbose,
        broker_factory=broker_factory,
        environment_factory=environment_factory,
        logger=logger,
        system_prompt=system_prompt,
    )


def create_rlm_from_config(
    config: RLMConfig,
    *,
    llm: LLMPort | None = None,
    llm_registry: LLMRegistry | None = None,
    environment_registry: EnvironmentRegistry | None = None,
    logger_registry: LoggerRegistry | None = None,
) -> RLM:
    """
    Construct an `RLM` from config.

    Phase 2 allows optionally providing an `llm_registry` to build an `LLMPort`
    from `config.llm`. If neither `llm` nor `llm_registry` is provided, we fail
    fast with a helpful message.
    """
    if llm is None:
        if llm_registry is None:
            llm_registry = DefaultLLMRegistry()
        llm = llm_registry.build(config.llm)
    else:
        # If the caller provided the root LLM but not a registry, we may still
        # need a registry for `config.other_llms`.
        if llm_registry is None:
            llm_registry = DefaultLLMRegistry()

    other_llms: list[LLMPort] = [llm_registry.build(c) for c in config.other_llms]

    if environment_registry is None:
        environment_registry = DefaultEnvironmentRegistry()
    if logger_registry is None:
        logger_registry = DefaultLoggerRegistry()

    environment_factory = environment_registry.build(config.env)
    logger = logger_registry.build(config.logger)

    return create_rlm(
        llm,
        other_llms=other_llms,
        environment=config.env.environment,
        environment_kwargs=config.env.environment_kwargs,
        max_depth=config.max_depth,
        max_iterations=config.max_iterations,
        verbose=config.verbose,
        environment_factory=environment_factory,
        logger=logger,
    )
