from __future__ import annotations

from typing import TYPE_CHECKING

from rlm.adapters.broker.tcp import TcpBrokerAdapter
from rlm.adapters.tools import InMemoryToolRegistry
from rlm.api.registries import DefaultEnvironmentRegistry
from rlm.application.config import EnvironmentConfig
from rlm.application.use_cases.run_completion import (
    RunCompletionDeps,
    RunCompletionRequest,
    arun_completion,
    run_completion,
)
from rlm.domain.errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from rlm.application.config import EnvironmentName
    from rlm.application.use_cases.run_completion import EnvironmentFactory
    from rlm.domain.agent_ports import (
        ContextCompressor,
        NestedCallPolicy,
        StoppingPolicy,
        ToolPort,
        ToolRegistryPort,
    )
    from rlm.domain.models import ChatCompletion
    from rlm.domain.models.llm_request import ToolChoice
    from rlm.domain.ports import BrokerPort, LLMPort, LoggerPort
    from rlm.domain.services.rlm_orchestrator import AgentMode
    from rlm.domain.types import Prompt

__all__ = ["RLM"]


class RLM:
    """
    Public RLM facade (Phase 1).

    This facade is intentionally small while we migrate from the upstream legacy
    implementation. In Phase 2 it delegates to the domain orchestrator via the
    `run_completion` application use case.

    Agent Capabilities (Phase 1 - Core):
        tools: List of ToolPort implementations or plain callables. When provided
            with agent_mode="tools", these are registered and offered to the LLM.
        output_type: Target type for structured output validation (Pydantic model,
            dataclass, etc.). Currently reserved for future use.
        agent_mode: Either "code" (default, code execution) or "tools" (function
            calling). Tool mode requires tools to be provided.

    Extension Protocols (Phase 2.7-2.8):
        stopping_policy: Custom policy for controlling iteration loop termination.
            Allows EIG-gated stopping, entropy-based termination, etc.
        context_compressor: Compresses nested call results before returning to parent.
            Allows summarization, truncation, or extraction strategies.
        nested_call_policy: Controls whether nested llm_query() calls spawn
            sub-orchestrators vs simple LLM calls.

    Note:
        Tool calling mode (agent_mode="tools") is infrastructure-ready but not
        yet implemented. Use the default "code" mode for now.

    """

    def __init__(
        self,
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
        # Agent capability extensions (Phase 1 - Core)
        tools: list[ToolPort | Callable[..., object]] | None = None,
        output_type: type | None = None,
        agent_mode: AgentMode = "code",
        # Extension protocols (Phase 2.7-2.8)
        stopping_policy: StoppingPolicy | None = None,
        context_compressor: ContextCompressor | None = None,
        nested_call_policy: NestedCallPolicy | None = None,
    ) -> None:
        self._llm = llm
        self._other_llms = list(other_llms or [])
        self._max_depth = max_depth
        self._max_iterations = max_iterations
        self._verbose = verbose
        self._logger = logger

        # Agent capability storage
        self._tools = tools
        self._output_type = output_type
        self._agent_mode: AgentMode = agent_mode
        self._tool_registry: ToolRegistryPort | None = None

        # Extension protocols storage
        self._stopping_policy = stopping_policy
        self._context_compressor = context_compressor
        self._nested_call_policy = nested_call_policy

        # Build tool registry if tools provided
        if tools:
            registry = InMemoryToolRegistry()
            for tool in tools:
                registry.register(tool)
            self._tool_registry = registry

        self._broker_factory = broker_factory or _default_tcp_broker_factory
        if environment_factory is None:
            environment_factory = DefaultEnvironmentRegistry().build(
                EnvironmentConfig(
                    environment=environment,
                    environment_kwargs=environment_kwargs or {},
                ),
            )
        self._environment_factory = environment_factory
        self._system_prompt = system_prompt

    def completion(
        self,
        prompt: Prompt,
        *,
        root_prompt: str | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> ChatCompletion:
        broker = self._broker_factory(self._llm)
        # Register additional models for subcalls (Phase 4 multi-backend).
        seen = {self._llm.model_name}
        for other in self._other_llms:
            name = other.model_name
            if name in seen:
                raise ValidationError(f"Duplicate model registered: {name!r}")
            seen.add(name)
            broker.register_llm(name, other)
        # `RunCompletionDeps.system_prompt` is a dataclass field, not a class-level
        # constant (and under `slots=True` it resolves to a `member_descriptor`).
        # So: only pass a system prompt if the user explicitly provided one;
        # otherwise rely on the dataclass default (`RLM_SYSTEM_PROMPT`).
        if self._system_prompt is None:
            deps = RunCompletionDeps(
                llm=self._llm,
                broker=broker,
                environment_factory=self._environment_factory,
                logger=self._logger,
                agent_mode=self._agent_mode,
                tool_registry=self._tool_registry,
                stopping_policy=self._stopping_policy,
                context_compressor=self._context_compressor,
                nested_call_policy=self._nested_call_policy,
            )
        else:
            deps = RunCompletionDeps(
                llm=self._llm,
                broker=broker,
                environment_factory=self._environment_factory,
                logger=self._logger,
                system_prompt=self._system_prompt,
                agent_mode=self._agent_mode,
                tool_registry=self._tool_registry,
                stopping_policy=self._stopping_policy,
                context_compressor=self._context_compressor,
                nested_call_policy=self._nested_call_policy,
            )
        req = RunCompletionRequest(
            prompt=prompt,
            root_prompt=root_prompt,
            max_depth=self._max_depth,
            max_iterations=self._max_iterations,
            tool_choice=tool_choice,
        )
        return run_completion(req, deps=deps)

    async def acompletion(
        self,
        prompt: Prompt,
        *,
        root_prompt: str | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> ChatCompletion:
        broker = self._broker_factory(self._llm)
        # Register additional models for subcalls (Phase 4 multi-backend).
        seen = {self._llm.model_name}
        for other in self._other_llms:
            name = other.model_name
            if name in seen:
                raise ValidationError(f"Duplicate model registered: {name!r}")
            seen.add(name)
            broker.register_llm(name, other)
        if self._system_prompt is None:
            deps = RunCompletionDeps(
                llm=self._llm,
                broker=broker,
                environment_factory=self._environment_factory,
                logger=self._logger,
                agent_mode=self._agent_mode,
                tool_registry=self._tool_registry,
                stopping_policy=self._stopping_policy,
                context_compressor=self._context_compressor,
                nested_call_policy=self._nested_call_policy,
            )
        else:
            deps = RunCompletionDeps(
                llm=self._llm,
                broker=broker,
                environment_factory=self._environment_factory,
                logger=self._logger,
                system_prompt=self._system_prompt,
                agent_mode=self._agent_mode,
                tool_registry=self._tool_registry,
                stopping_policy=self._stopping_policy,
                context_compressor=self._context_compressor,
                nested_call_policy=self._nested_call_policy,
            )
        req = RunCompletionRequest(
            prompt=prompt,
            root_prompt=root_prompt,
            max_depth=self._max_depth,
            max_iterations=self._max_iterations,
            tool_choice=tool_choice,
        )
        return await arun_completion(req, deps=deps)


def _default_tcp_broker_factory(llm: LLMPort, /) -> BrokerPort:
    """
    Default broker: TCP broker speaking the infra wire protocol.

    This is used so environments can call `llm_query()` during code execution.
    """
    return TcpBrokerAdapter(llm)
