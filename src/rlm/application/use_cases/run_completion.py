from __future__ import annotations

import asyncio
import inspect
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, overload

from rlm.domain.errors import BrokerError, ExecutionError, RLMError
from rlm.domain.models import ChatCompletion, RunMetadata
from rlm.domain.models.usage import merge_usage_summaries
from rlm.domain.services.prompts import RLM_SYSTEM_PROMPT
from rlm.domain.services.rlm_orchestrator import AgentMode, RLMOrchestrator

if TYPE_CHECKING:
    from rlm.domain.agent_ports import (
        ContextCompressor,
        NestedCallPolicy,
        StoppingPolicy,
        ToolRegistryPort,
    )
    from rlm.domain.models.llm_request import ToolChoice
    from rlm.domain.ports import BrokerPort, EnvironmentPort, LLMPort, LoggerPort
    from rlm.domain.types import Prompt


class EnvironmentFactory(Protocol):
    """
    Builds an EnvironmentPort for a single run.

    The factory is responsible for binding any broker address into the environment
    implementation (e.g., legacy LocalREPL/DockerREPL need an LMHandler address for
    `llm_query()`).
    """

    @overload
    def build(self, broker_address: tuple[str, int], /) -> EnvironmentPort: ...

    @overload
    def build(self, broker: BrokerPort, broker_address: tuple[str, int], /) -> EnvironmentPort: ...

    @overload
    def build(
        self,
        broker: BrokerPort,
        broker_address: tuple[str, int],
        correlation_id: str | None,
        /,
    ) -> EnvironmentPort: ...

    def build(self, *args: object) -> EnvironmentPort: ...


# Factory signature shapes for backwards-compatible environment building.
# These represent the expected number of positional parameters for different factory versions:
# - FULL_SIGNATURE (3): build(broker, broker_address, correlation_id)
# - PARTIAL_SIGNATURE (2): build(broker, broker_address)
# - MINIMAL_SIGNATURE (1): build(broker_address)
_FULL_SIGNATURE_PARAMS = 3
_PARTIAL_SIGNATURE_PARAMS = 2


def _try_build_with_fallback(
    factory: EnvironmentFactory,
    broker: BrokerPort,
    broker_address: tuple[str, int],
    correlation_id: str | None,
) -> EnvironmentPort:
    """Try factory call shapes in order from richest to minimal."""
    try:
        return factory.build(broker, broker_address, correlation_id)  # type: ignore[misc]
    except TypeError:
        pass
    try:
        return factory.build(broker, broker_address)  # type: ignore[misc]
    except TypeError:
        return factory.build(broker_address)  # type: ignore[misc]


def _build_environment(
    factory: EnvironmentFactory,
    broker: BrokerPort,
    broker_address: tuple[str, int],
    correlation_id: str | None,
    /,
) -> EnvironmentPort:
    """
    Call `EnvironmentFactory.build()` in a backwards-compatible way.

    During the migration, some factories expose:
    - `build(broker_address)`
    - `build(broker, broker_address)`
    - `build(broker, broker_address, correlation_id)`

    We select the call shape via signature introspection so we don't accidentally
    swallow `TypeError` raised *inside* the factory.
    """
    try:
        sig = inspect.signature(factory.build)
    except (TypeError, ValueError):
        # Fallback: try call shapes in order from richest to minimal.
        return _try_build_with_fallback(factory, broker, broker_address, correlation_id)

    params = list(sig.parameters.values())
    has_var_positional = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params)

    # inspect.Parameter.default is typed as Any by stdlib - use helper to avoid Any propagation
    empty_sentinel: object = inspect.Parameter.empty

    def _has_no_default(p: inspect.Parameter) -> bool:
        default_val: object = p.default  # pyright: ignore[reportAny] - stdlib boundary
        return default_val is empty_sentinel

    required_count = sum(
        1
        for p in params
        if _has_no_default(p)
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )

    # Select call shape based on signature analysis.
    # Varargs or 3+ params: use full signature with correlation_id for tracing.
    # 2 params: use broker + address.
    # Otherwise: minimal with just address.
    if has_var_positional or required_count >= _FULL_SIGNATURE_PARAMS:
        return factory.build(broker, broker_address, correlation_id)  # type: ignore[misc]
    if required_count >= _PARTIAL_SIGNATURE_PARAMS:
        return factory.build(broker, broker_address)  # type: ignore[misc]
    return factory.build(broker_address)  # type: ignore[misc]


@dataclass(frozen=True, slots=True)
class RunCompletionDeps:
    """
    Dependencies for running a completion.

    Notes:
    - `broker` is started/stopped per run.
    - `environment_factory` is invoked per run.

    Agent Capabilities (Phase 1 - Core):
    - `agent_mode`: "code" (default) or "tools" for function calling.
    - `tool_registry`: Required when agent_mode="tools".

    Extension Protocols (Phase 2.7):
    - `stopping_policy`: Custom stopping criteria for iteration loops.
    - `context_compressor`: Compress nested call returns.
    - `nested_call_policy`: Control nested orchestrator spawning.

    """

    llm: LLMPort
    broker: BrokerPort
    environment_factory: EnvironmentFactory
    logger: LoggerPort | None = None
    system_prompt: str = RLM_SYSTEM_PROMPT

    # Agent capability extensions (Phase 1 - Core)
    agent_mode: AgentMode = "code"
    tool_registry: ToolRegistryPort | None = None

    # Extension protocols (Phase 2.7-2.8)
    stopping_policy: StoppingPolicy | None = None
    context_compressor: ContextCompressor | None = None
    nested_call_policy: NestedCallPolicy | None = None


@dataclass(frozen=True, slots=True)
class RunCompletionRequest:
    prompt: Prompt
    root_prompt: str | None = None
    max_depth: int = 1
    max_iterations: int = 30
    tool_choice: ToolChoice | None = None


def _infer_environment_type(env: EnvironmentPort, /) -> str:
    # Best-effort environment type inference without importing adapters/legacy.
    env_type: str = "unknown"
    # getattr returns Any at SDK boundary - narrow with isinstance before use
    declared: object = getattr(env, "environment_type", None)
    if isinstance(declared, str) and declared.strip():
        env_type = declared
    else:
        inner: object = getattr(env, "_env", None)
        # Narrow inner to get its type name safely
        inner_name = type(inner).__name__ if inner is not None else type(env).__name__
        if "DockerREPL" in inner_name:
            env_type = "docker"
        elif "LocalREPL" in inner_name:
            env_type = "local"
    return env_type


def run_completion(request: RunCompletionRequest, *, deps: RunCompletionDeps) -> ChatCompletion:
    """
    Use case: run an RLM completion using the domain orchestrator.

    This function:
    - starts the broker (for env `llm_query()` subcalls)
    - builds an EnvironmentPort bound to the broker address
    - runs the domain orchestrator
    - ensures cleanup (env + broker)
    """
    correlation_id = uuid.uuid4().hex
    try:
        broker_addr = deps.broker.start()
    except Exception as e:
        raise BrokerError("Failed to start broker") from e

    try:
        try:
            env = _build_environment(
                deps.environment_factory,
                deps.broker,
                broker_addr,
                correlation_id,
            )
        except Exception as e:
            raise ExecutionError("Failed to build environment") from e

        try:
            if deps.logger is not None:
                env_type = _infer_environment_type(env)
                deps.logger.log_metadata(
                    RunMetadata(
                        root_model=deps.llm.model_name,
                        max_depth=request.max_depth,
                        max_iterations=request.max_iterations,
                        backend=deps.llm.model_name,
                        backend_kwargs={},
                        environment_type=env_type,
                        environment_kwargs={},
                        other_backends=None,
                        correlation_id=correlation_id,
                    ),
                )

            orch = RLMOrchestrator(
                llm=deps.llm,
                environment=env,
                logger=deps.logger,
                system_prompt=deps.system_prompt,
                agent_mode=deps.agent_mode,
                tool_registry=deps.tool_registry,
                stopping_policy=deps.stopping_policy,
                context_compressor=deps.context_compressor,
                nested_call_policy=deps.nested_call_policy,
            )
            try:
                cc = orch.completion(
                    request.prompt,
                    root_prompt=request.root_prompt,
                    max_depth=request.max_depth,
                    depth=0,
                    max_iterations=request.max_iterations,
                    correlation_id=correlation_id,
                    tool_choice=request.tool_choice,
                )
                # Merge orchestrator usage (root calls) with broker usage (env subcalls).
                merged_usage = merge_usage_summaries(
                    [cc.usage_summary, deps.broker.get_usage_summary()],
                )
                return ChatCompletion(
                    root_model=cc.root_model,
                    prompt=cc.prompt,
                    response=cc.response,
                    usage_summary=merged_usage,
                    execution_time=cc.execution_time,
                    tool_calls=cc.tool_calls,
                    finish_reason=cc.finish_reason,
                )
            except RLMError:
                raise
            except Exception as e:
                raise RLMError("RLM run failed") from e
        finally:
            env.cleanup()
    finally:
        deps.broker.stop()


async def arun_completion(
    request: RunCompletionRequest,
    *,
    deps: RunCompletionDeps,
) -> ChatCompletion:
    """
    Async use case: run an RLM completion using the domain orchestrator.

    Notes:
    - `BrokerPort` / `EnvironmentPort` are sync; we execute their blocking methods via
      `asyncio.to_thread` so callers can safely run this in an event loop.
    - Cleanup is cancellation-safe via `asyncio.shield(...)`.

    """
    correlation_id = uuid.uuid4().hex
    try:
        broker_addr = await asyncio.to_thread(deps.broker.start)
    except Exception as e:
        raise BrokerError("Failed to start broker") from e

    try:
        try:
            env = await asyncio.to_thread(
                _build_environment,
                deps.environment_factory,
                deps.broker,
                broker_addr,
                correlation_id,
            )
        except Exception as e:
            raise ExecutionError("Failed to build environment") from e

        try:
            if deps.logger is not None:
                env_type = _infer_environment_type(env)
                await asyncio.to_thread(
                    deps.logger.log_metadata,
                    RunMetadata(
                        root_model=deps.llm.model_name,
                        max_depth=request.max_depth,
                        max_iterations=request.max_iterations,
                        backend=deps.llm.model_name,
                        backend_kwargs={},
                        environment_type=env_type,
                        environment_kwargs={},
                        other_backends=None,
                        correlation_id=correlation_id,
                    ),
                )

            orch = RLMOrchestrator(
                llm=deps.llm,
                environment=env,
                logger=deps.logger,
                system_prompt=deps.system_prompt,
                agent_mode=deps.agent_mode,
                tool_registry=deps.tool_registry,
                stopping_policy=deps.stopping_policy,
                context_compressor=deps.context_compressor,
                nested_call_policy=deps.nested_call_policy,
            )
            try:
                cc = await orch.acompletion(
                    request.prompt,
                    root_prompt=request.root_prompt,
                    max_depth=request.max_depth,
                    depth=0,
                    max_iterations=request.max_iterations,
                    correlation_id=correlation_id,
                    tool_choice=request.tool_choice,
                )
                broker_usage = await asyncio.to_thread(deps.broker.get_usage_summary)
                merged_usage = merge_usage_summaries([cc.usage_summary, broker_usage])
                return ChatCompletion(
                    root_model=cc.root_model,
                    prompt=cc.prompt,
                    response=cc.response,
                    usage_summary=merged_usage,
                    execution_time=cc.execution_time,
                    tool_calls=cc.tool_calls,
                    finish_reason=cc.finish_reason,
                )
            except RLMError:
                raise
            except Exception as e:
                raise RLMError("RLM run failed") from e
        finally:
            await asyncio.shield(asyncio.to_thread(env.cleanup))
    finally:
        await asyncio.shield(asyncio.to_thread(deps.broker.stop))
