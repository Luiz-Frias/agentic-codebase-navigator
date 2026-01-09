from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rlm.application.config import EnvironmentName
from rlm.application.use_cases.run_completion import (
    EnvironmentFactory,
    RunCompletionDeps,
    RunCompletionRequest,
    run_completion,
)
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
        self._max_depth = max_depth
        self._max_iterations = max_iterations
        self._verbose = verbose
        self._logger = logger

        self._broker_factory = broker_factory or _default_tcp_broker_factory
        self._environment_factory = environment_factory or _default_legacy_environment_factory(
            environment, environment_kwargs or {}
        )
        self._system_prompt = system_prompt

    def completion(self, prompt: Prompt, *, root_prompt: str | None = None) -> ChatCompletion:
        broker = self._broker_factory(self._llm)
        deps = RunCompletionDeps(
            llm=self._llm,
            broker=broker,
            environment_factory=self._environment_factory,
            logger=self._logger,
            system_prompt=self._system_prompt or RunCompletionDeps.system_prompt,  # default text
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


def _default_legacy_environment_factory(
    environment: EnvironmentName,
    environment_kwargs: dict[str, Any],
) -> EnvironmentFactory:
    """
    Default environment factory: legacy LocalREPL/DockerREPL adapters (Phase 2).

    We keep imports lazy so default installs can still import `rlm` without
    pulling optional dependencies unless selected.
    """

    def _build(broker: BrokerPort | None, broker_address: tuple[str, int], /):
        from rlm.adapters.legacy.environment import LegacyEnvironmentAdapter

        match environment:
            case "local":
                from rlm._legacy.environments.local_repl import LocalREPL

                kwargs = dict(environment_kwargs)
                kwargs.pop("lm_handler_address", None)
                # Prefer injecting the BrokerPort explicitly (no runtime monkey patching).
                # Fallback is legacy socket transport (lm_handler_address).
                if broker is not None:
                    env = LocalREPL(lm_handler_address=broker_address, broker=broker, **kwargs)
                else:
                    env = LocalREPL(lm_handler_address=broker_address, **kwargs)
                return LegacyEnvironmentAdapter(env)
            case "docker":
                from rlm._legacy.environments.docker_repl import DockerREPL

                kwargs = dict(environment_kwargs)
                kwargs.pop("lm_handler_address", None)
                if broker is not None:
                    env = DockerREPL(lm_handler_address=broker_address, broker=broker, **kwargs)
                else:
                    env = DockerREPL(lm_handler_address=broker_address, **kwargs)
                return LegacyEnvironmentAdapter(env)
            case "modal" | "prime":
                raise NotImplementedError(
                    f"Environment '{environment}' is not supported in Phase 2."
                )
            case _:
                raise ValueError(f"Unknown environment: {environment!r}")

    class _Factory:
        def build(self, *args: object) -> object:  # noqa: ANN401 - migration-compatible facade
            """
            Build an environment for a run.

            Supported call shapes during migration:
            - build(broker_address)
            - build(broker, broker_address)
            """

            match args:
                case ((str() as host, int() as port),):
                    return _build(None, (host, port))
                case (broker, (str() as host, int() as port)):
                    return _build(broker, (host, port))  # type: ignore[arg-type]
                case _:
                    raise TypeError(
                        "EnvironmentFactory.build() expects (broker_address) or (broker, broker_address)"
                    )

    return _Factory()
