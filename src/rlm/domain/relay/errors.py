from __future__ import annotations

from typing import Literal

ErrorType = Literal["transient", "fatal"]


class StateError(Exception):
    def __init__(
        self,
        *,
        error_type: ErrorType,
        message: str,
        retry_hint: str | None = None,
    ) -> None:
        self.error_type = error_type
        self.message = message
        self.retry_hint = retry_hint
        super().__init__(message)


class PipelineDefinitionError(Exception):
    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))
