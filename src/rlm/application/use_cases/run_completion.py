from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rlm.domain.models import ChatCompletion
from rlm.domain.ports import BrokerPort, EnvironmentPort, LLMPort, LoggerPort
from rlm.domain.services.rlm_orchestrator import RLMOrchestrator
from rlm.domain.types import Prompt


class EnvironmentFactory(Protocol):
    """
    Builds an EnvironmentPort for a single run.

    The factory is responsible for binding any broker address into the environment
    implementation (e.g., legacy LocalREPL/DockerREPL need an LMHandler address for
    `llm_query()`).
    """

    def build(self, broker_address: tuple[str, int], /) -> EnvironmentPort: ...


@dataclass(frozen=True, slots=True)
class RunCompletionDeps:
    """
    Dependencies for running a completion.

    Notes:
    - `broker` is started/stopped per run.
    - `environment_factory` is invoked per run.
    """

    llm: LLMPort
    broker: BrokerPort
    environment_factory: EnvironmentFactory
    logger: LoggerPort | None = None
    system_prompt: str = (
        "You are a recursive language model (RLM). "
        "You can use a `repl` tool to execute code and get results. "
        "When you have a final answer, respond with FINAL(your answer)."
    )


@dataclass(frozen=True, slots=True)
class RunCompletionRequest:
    prompt: Prompt
    root_prompt: str | None = None
    max_depth: int = 1
    max_iterations: int = 30


def run_completion(request: RunCompletionRequest, *, deps: RunCompletionDeps) -> ChatCompletion:
    """
    Use case: run an RLM completion using the domain orchestrator.

    This function:
    - starts the broker (for env `llm_query()` subcalls)
    - builds an EnvironmentPort bound to the broker address
    - runs the domain orchestrator
    - ensures cleanup (env + broker)
    """
    broker_addr = deps.broker.start()
    try:
        env = deps.environment_factory.build(broker_addr)
        try:
            orch = RLMOrchestrator(
                llm=deps.llm,
                environment=env,
                logger=deps.logger,
                system_prompt=deps.system_prompt,
            )
            return orch.completion(
                request.prompt,
                root_prompt=request.root_prompt,
                max_depth=request.max_depth,
                depth=0,
                max_iterations=request.max_iterations,
            )
        finally:
            env.cleanup()
    finally:
        deps.broker.stop()
