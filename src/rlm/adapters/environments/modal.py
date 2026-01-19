from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from rlm.adapters.base import BaseEnvironmentAdapter

if TYPE_CHECKING:
    from rlm.domain.models import ReplResult
    from rlm.domain.types import ContextPayload


class ModalEnvironmentAdapter(BaseEnvironmentAdapter):
    """
    Modal environment adapter (optional dependency).

    Phase 05:
    - Keep imports lazy so the base package can be imported without `modal` installed.
    - Provide a helpful error when selected without optional deps.
    - Full remote execution/polling is implemented in later phases.
    """

    environment_type: str = "modal"

    def __init__(self, **_kwargs: object) -> None:
        if importlib.util.find_spec("modal") is None:
            raise RuntimeError(
                "Environment 'modal' was selected but the optional dependency 'modal' is not installed. "
                "Install it (and any future extras) and retry: `pip install modal`.",
            )

        raise NotImplementedError(
            "Environment 'modal' is not implemented yet in Phase 05. "
            "Use environment='local' or environment='docker' for now.",
        )

    def load_context(self, context_payload: ContextPayload, /) -> None:  # pragma: no cover
        raise NotImplementedError

    def execute_code(self, code: str, /) -> ReplResult:  # pragma: no cover
        raise NotImplementedError

    def cleanup(self) -> None:  # pragma: no cover
        return None
