from __future__ import annotations

from rlm._legacy.environments.base_env import BaseEnv
from rlm.adapters.base import BaseEnvironmentAdapter
from rlm.adapters.legacy.mappers import legacy_repl_result_to_domain
from rlm.domain.models import ReplResult
from rlm.domain.types import ContextPayload


class LegacyEnvironmentAdapter(BaseEnvironmentAdapter):
    """Adapter: legacy `BaseEnv` -> domain `EnvironmentPort`."""

    def __init__(self, env: BaseEnv):
        self._env = env

    def load_context(self, context_payload: ContextPayload, /) -> None:
        self._env.load_context(context_payload)  # type: ignore[arg-type]

    def execute_code(self, code: str, /) -> ReplResult:
        legacy_result = self._env.execute_code(code)
        return legacy_repl_result_to_domain(legacy_result)

    def cleanup(self) -> None:
        if hasattr(self._env, "cleanup"):
            self._env.cleanup()
