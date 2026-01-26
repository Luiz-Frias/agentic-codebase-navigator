from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryStrategy:
    max_attempts: int = 1
    backoff_seconds: float = 0.0
    retry_on: tuple[type[Exception], ...] = (Exception,)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("RetryStrategy.max_attempts must be >= 1")
        if self.backoff_seconds < 0:
            raise ValueError("RetryStrategy.backoff_seconds must be >= 0")
