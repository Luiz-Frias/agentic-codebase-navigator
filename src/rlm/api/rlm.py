from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rlm.application.config import EnvironmentConfig, EnvironmentName
from rlm.application.use_cases.run_completion import (
    EnvironmentFactory,
    RunCompletionDeps,
    RunCompletionRequest,
    run_completion,
)
from rlm.domain.errors import ValidationError
from rlm.domain.models import ChatCompletion
from rlm.domain.ports import BrokerPort, LLMPort, LoggerPort
from rlm.domain.types import Prompt


class RLM:
    """
    Public RLM facade (Phase 1).

    This facade is intentionally small while we migrate from the upstream legacy
    implementation. In Phase 2 it delegates to the domain orchestrator via the
    `run_completion` application use case.
    """

    def __init__(
        self,
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
    ) -> None:
        self._llm = llm
        self._other_llms = list(other_llms or [])
        self._max_depth = max_depth
        self._max_iterations = max_iterations
        self._verbose = verbose
        self._logger = logger

        self._broker_factory = broker_factory or _default_tcp_broker_factory
        if environment_factory is None:
            from rlm.api.registries import DefaultEnvironmentRegistry

            environment_factory = DefaultEnvironmentRegistry().build(
                EnvironmentConfig(
                    environment=environment, environment_kwargs=environment_kwargs or {}
                )
            )
        self._environment_factory = environment_factory
        self._system_prompt = system_prompt

    def completion(self, prompt: Prompt, *, root_prompt: str | None = None) -> ChatCompletion:
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
            )
        else:
            deps = RunCompletionDeps(
                llm=self._llm,
                broker=broker,
                environment_factory=self._environment_factory,
                logger=self._logger,
                system_prompt=self._system_prompt,
            )
        req = RunCompletionRequest(
            prompt=prompt,
            root_prompt=root_prompt,
            max_depth=self._max_depth,
            max_iterations=self._max_iterations,
        )
        return run_completion(req, deps=deps)


def _default_tcp_broker_factory(llm: LLMPort, /) -> BrokerPort:
    """
    Default broker: TCP broker speaking the infra wire protocol.

    This is used so environments can call `llm_query()` during code execution.
    """
    from rlm.adapters.broker.tcp import TcpBrokerAdapter

    return TcpBrokerAdapter(llm)
