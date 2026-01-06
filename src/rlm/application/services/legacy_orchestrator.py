from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from rlm.adapters.legacy.llm import _as_legacy_client
from rlm.domain.ports import LLMPort, Prompt


class LegacyOrchestratorService:
    """
    Temporary application service that delegates to the upstream legacy orchestrator.

    This is a bridge during Phase 1/2: we run the proven legacy loop while we
    build the new domain/application orchestrator in later phases.

    The legacy orchestrator expects a `get_client()` router; we patch it
    *locally* during the call to inject the selected `LLMPort`.
    """

    def __init__(
        self,
        llm: LLMPort,
        *,
        environment: str = "local",
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
    def _patched_legacy_client_router(self):
        import rlm._legacy.core.rlm as legacy_rlm_mod

        original_get_client = legacy_rlm_mod.get_client

        def _get_client(_backend, _backend_kwargs):
            return _as_legacy_client(self._llm)

        legacy_rlm_mod.get_client = _get_client  # type: ignore[assignment]
        try:
            yield
        finally:
            legacy_rlm_mod.get_client = original_get_client  # type: ignore[assignment]

    def completion(self, prompt: Prompt, *, root_prompt: str | None = None) -> str:
        import rlm._legacy.core.rlm as legacy_rlm_mod

        with self._patched_legacy_client_router():
            legacy = legacy_rlm_mod.RLM(
                backend="openai",
                backend_kwargs={"model_name": self._llm.model_name},
                environment=self._environment,  # type: ignore[arg-type]
                environment_kwargs=self._environment_kwargs,
                max_depth=self._max_depth,
                max_iterations=self._max_iterations,
                verbose=self._verbose,
            )
            cc = legacy.completion(prompt, root_prompt=root_prompt)
            return cc.response
