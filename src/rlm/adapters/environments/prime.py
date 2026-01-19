from __future__ import annotations

from typing import TYPE_CHECKING

from rlm.adapters.base import BaseEnvironmentAdapter

if TYPE_CHECKING:
    from rlm.domain.models import ReplResult
    from rlm.domain.types import ContextPayload


class PrimeEnvironmentAdapter(BaseEnvironmentAdapter):
    """
    Placeholder adapter for the (future) "prime" environment.

    Phase 05:
    - We register a concrete adapter so configuration/selection can be validated.
    - Execution is intentionally unimplemented, with a helpful error message.
    """

    environment_type: str = "prime"

    def __init__(self, **_kwargs: object) -> None:
        raise NotImplementedError(
            "Environment 'prime' is not implemented yet. "
            "Use environment='local' or environment='docker' instead.",
        )

    def load_context(self, context_payload: ContextPayload, /) -> None:  # pragma: no cover
        raise NotImplementedError

    def execute_code(self, code: str, /) -> ReplResult:  # pragma: no cover
        raise NotImplementedError

    def cleanup(self) -> None:  # pragma: no cover
        return None
