from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

JoinMode = Literal["all", "race"]


@dataclass(frozen=True, slots=True)
class JoinSpec:
    mode: JoinMode = "all"
    timeout_seconds: float | None = 30.0
