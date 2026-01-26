from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ErrorType = Literal["transient", "fatal"]


@dataclass(frozen=True, slots=True)
class StateError(Exception):
    error_type: ErrorType
    message: str
    retry_hint: str | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)


class PipelineDefinitionError(Exception):
    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))
