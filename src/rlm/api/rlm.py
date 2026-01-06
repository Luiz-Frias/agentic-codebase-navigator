from __future__ import annotations

from typing import Any

from rlm.application.services.legacy_orchestrator import LegacyOrchestratorService
from rlm.domain.ports import LLMPort, Prompt


class RLM:
    """
    Public RLM facade (Phase 1).

    This facade is intentionally small while we migrate from the upstream legacy
    implementation. Today it delegates to the legacy orchestrator via the
    `LegacyOrchestratorService`.
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
        self._service = LegacyOrchestratorService(
            llm,
            environment=environment,
            environment_kwargs=environment_kwargs,
            max_depth=max_depth,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    def completion(self, prompt: Prompt, *, root_prompt: str | None = None) -> str:
        return self._service.completion(prompt, root_prompt=root_prompt)
