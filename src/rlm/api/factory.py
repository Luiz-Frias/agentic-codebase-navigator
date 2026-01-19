from __future__ import annotations

from typing import TYPE_CHECKING

from rlm.api.registries import (
    DefaultEnvironmentRegistry,
    DefaultLLMRegistry,
    DefaultLoggerRegistry,
    EnvironmentRegistry,
    LLMRegistry,
    LoggerRegistry,
)
from rlm.api.rlm import RLM

if TYPE_CHECKING:
    from collections.abc import Callable

    from rlm.application.config import AgentModeName, EnvironmentName, RLMConfig
    from rlm.application.use_cases.run_completion import EnvironmentFactory
    from rlm.domain.agent_ports import ToolPort
    from rlm.domain.ports import BrokerPort, LLMPort, LoggerPort


def create_rlm(
    llm: LLMPort,
    *,
    other_llms: list[LLMPort] | None = None,
    environment: EnvironmentName = "local",
    environment_kwargs: dict[str, object] | None = None,
    max_depth: int = 1,
    max_iterations: int = 30,
    verbose: bool = False,
    broker_factory: Callable[[LLMPort], BrokerPort] | None = None,
    environment_factory: EnvironmentFactory | None = None,
    logger: LoggerPort | None = None,
    system_prompt: str | None = None,
    # Agent capability extensions
    tools: list[ToolPort | Callable[..., object]] | None = None,
    agent_mode: AgentModeName = "code",
) -> RLM:
    """
    Convenience factory for the public `RLM` facade.

    Args:
        llm: Primary LLM adapter.
        other_llms: Additional LLM adapters for multi-backend routing.
        environment: Execution environment name.
        environment_kwargs: Environment-specific configuration.
        max_depth: Maximum recursion depth for nested completions.
        max_iterations: Maximum iterations in code execution loop.
        verbose: Enable verbose logging.
        broker_factory: Custom broker factory for LLM routing.
        environment_factory: Custom environment factory.
        logger: Logger port for iteration logging.
        system_prompt: Custom system prompt override.
        tools: List of tools (ToolPort or callable) for function calling.
        agent_mode: Either "code" (default) or "tools" for function calling.

    Returns:
        Configured RLM facade instance.

    """
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
        tools=tools,
        agent_mode=agent_mode,
    )


def create_rlm_from_config(
    config: RLMConfig,
    *,
    llm: LLMPort | None = None,
    llm_registry: LLMRegistry | None = None,
    environment_registry: EnvironmentRegistry | None = None,
    logger_registry: LoggerRegistry | None = None,
    # Runtime tool injection (tools cannot be serialized to config)
    tools: list[ToolPort | Callable[..., object]] | None = None,
) -> RLM:
    """
    Construct an `RLM` from config.

    Args:
        config: RLM configuration object.
        llm: Optional pre-built LLM adapter (overrides config.llm).
        llm_registry: Registry for building LLM adapters from config.
        environment_registry: Registry for building environment factories.
        logger_registry: Registry for building loggers.
        tools: List of tools for function calling (runtime injection, since
            tools are Python callables and cannot be serialized to config).

    Returns:
        Configured RLM facade instance.

    Note:
        The `agent_mode` is read from config. If agent_mode="tools", you must
        provide tools via the `tools` parameter.

    """
    if llm is None:
        if llm_registry is None:
            llm_registry = DefaultLLMRegistry()
        llm = llm_registry.build(config.llm)
    # If the caller provided the root LLM but not a registry, we may still
    # need a registry for `config.other_llms`.
    elif llm_registry is None:
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
        tools=tools,
        agent_mode=config.agent_mode,
    )
