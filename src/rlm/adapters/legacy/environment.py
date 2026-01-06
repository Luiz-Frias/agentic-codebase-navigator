from __future__ import annotations

from typing import Any

from rlm._legacy.environments.base_env import BaseEnv
from rlm.domain.ports import ContextPayload


class LegacyEnvironmentAdapter:
    """Adapter: legacy `BaseEnv` -> domain `EnvironmentPort`."""

    def __init__(self, env: BaseEnv):
        self._env = env

    def load_context(self, context_payload: ContextPayload, /) -> None:
        self._env.load_context(context_payload)  # type: ignore[arg-type]

    def execute_code(self, code: str, /) -> Any:
        return self._env.execute_code(code)

    def cleanup(self) -> None:
        if hasattr(self._env, "cleanup"):
            self._env.cleanup()
