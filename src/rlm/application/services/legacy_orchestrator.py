from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rlm.adapters.legacy.environment import LegacyEnvironmentAdapter
from rlm.adapters.legacy.llm import _as_legacy_client
from rlm.application.config import EnvironmentName
from rlm.domain.models import ChatCompletion
from rlm.domain.ports import LLMPort
from rlm.domain.services.rlm_orchestrator import RLMOrchestrator
from rlm.domain.types import Prompt


class LegacyOrchestratorService:
    """
    Temporary application service that runs the *domain* orchestrator while still
    using legacy broker/environment implementations for execution.

    Phase 2 goal: stop depending on the upstream legacy core loop (`rlm._legacy.core.rlm`).
    We keep the legacy LMHandler + envs only as adapters until Phase 3+ replaces
    the broker/protocol and Phase 5 replaces environments.

    Notes:
    - Root LLM calls go directly through `LLMPort` (domain orchestrator).
    - Subcalls from REPL environments (via `llm_query`) go through legacy `LMHandler`,
      backed by the same `LLMPort` via an adapter.
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
    ) -> None:
        self._llm = llm
        self._environment = environment
        self._environment_kwargs = environment_kwargs or {}
        self._max_depth = max_depth
        self._max_iterations = max_iterations
        self._verbose = verbose

    @contextmanager
    def _spawn_broker(self) -> Iterator[tuple[str, int]]:
        """
        Start the legacy TCP broker (LMHandler) so environments can call `llm_query()`.
        """
        from rlm._legacy.core.lm_handler import LMHandler

        handler = LMHandler(_as_legacy_client(self._llm), host="127.0.0.1", port=0)
        address = handler.start()
        try:
            yield address
        finally:
            handler.stop()

    @contextmanager
    def _spawn_environment(
        self, lm_handler_address: tuple[str, int]
    ) -> Iterator[LegacyEnvironmentAdapter]:
        """
        Construct and initialize the selected legacy environment, wrapped as an EnvironmentPort.

        For Docker, we must also run the environment-specific spawn context so the
        container + proxy are active before executing code.
        """
        match self._environment:
            case "local":
                from rlm._legacy.environments.local_repl import LocalREPL

                # Force `lm_handler_address` to our spawned broker.
                kwargs = dict(self._environment_kwargs)
                kwargs.pop("lm_handler_address", None)
                env = LocalREPL(lm_handler_address=lm_handler_address, **kwargs)
                try:
                    yield LegacyEnvironmentAdapter(env)
                finally:
                    env.cleanup()
            case "docker":
                from rlm._legacy.environments.docker_repl import DockerREPL

                kwargs = dict(self._environment_kwargs)
                # Force `lm_handler_address` to our spawned broker.
                kwargs.pop("lm_handler_address", None)
                env = DockerREPL(lm_handler_address=lm_handler_address, **kwargs)
                try:
                    yield LegacyEnvironmentAdapter(env)
                finally:
                    env.cleanup()
            case "modal" | "prime":
                raise NotImplementedError(
                    f"Environment '{self._environment}' is not supported in Phase 2."
                )
            case _:
                raise ValueError(f"Unknown environment: {self._environment!r}")

    def completion(self, prompt: Prompt, *, root_prompt: str | None = None) -> ChatCompletion:
        with self._spawn_broker() as addr:
            with self._spawn_environment(addr) as env_port:
                orch = RLMOrchestrator(
                    llm=self._llm,
                    environment=env_port,
                    logger=None,
                )
                return orch.completion(
                    prompt,
                    root_prompt=root_prompt,
                    max_depth=self._max_depth,
                    depth=0,
                    max_iterations=self._max_iterations,
                )
