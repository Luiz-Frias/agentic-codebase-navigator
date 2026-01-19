from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Uuid4IdGenerator:
    """
    Default correlation-id generator.

    This implements the `IdGeneratorPort` protocol via duck typing.
    """

    prefix: str | None = None

    def new_id(self) -> str:
        value = uuid.uuid4().hex
        if self.prefix:
            return f"{self.prefix}_{value}"
        return value
