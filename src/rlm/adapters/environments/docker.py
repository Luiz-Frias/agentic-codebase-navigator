from __future__ import annotations

from typing import Any

from rlm._legacy.environments.docker_repl import DockerREPL
from rlm.domain.ports import ContextPayload


class LegacyDockerEnvironmentAdapter:
    """
    Adapter: legacy `DockerREPL` -> domain `EnvironmentPort`.

    This is a thin wrapper (Phase 1). The legacy Docker env:
    - runs code in a container
    - exposes `llm_query()` to the container via a host HTTP proxy that forwards
      to the legacy TCP broker (`LMHandler`).
    """

    def __init__(self, **docker_kwargs: Any) -> None:
        docker_kwargs.setdefault("image", "python:3.12-slim")
        self._env = DockerREPL(**docker_kwargs)

    def load_context(self, context_payload: ContextPayload, /) -> None:
        self._env.load_context(context_payload)  # type: ignore[arg-type]

    def execute_code(self, code: str, /) -> Any:
        return self._env.execute_code(code)

    def cleanup(self) -> None:
        self._env.cleanup()
