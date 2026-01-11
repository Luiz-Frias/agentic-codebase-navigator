from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Ok[T]:
    value: T


@dataclass(slots=True, frozen=True)
class Err:
    error: str


type Result[T] = Ok[T] | Err
