from __future__ import annotations

import secrets
from dataclasses import dataclass

_RETRYABLE_OPENAI_ERROR_NAMES: set[str] = {
    "APIConnectionError",
    "APIError",
    "APITimeoutError",
    "InternalServerError",
    "RateLimitError",
    "ServiceUnavailableError",
}


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 4.0
    jitter_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("RetryConfig.max_attempts must be >= 1")
        if self.base_delay_seconds < 0:
            raise ValueError("RetryConfig.base_delay_seconds must be >= 0")
        if self.max_delay_seconds < 0:
            raise ValueError("RetryConfig.max_delay_seconds must be >= 0")
        if self.jitter_seconds < 0:
            raise ValueError("RetryConfig.jitter_seconds must be >= 0")


def is_retryable_openai_error(exc: BaseException, /) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return type(exc).__name__ in _RETRYABLE_OPENAI_ERROR_NAMES


def compute_retry_delay(config: RetryConfig, attempt: int, /) -> float:
    backoff = config.base_delay_seconds * (2 ** max(attempt - 1, 0))
    delay = min(config.max_delay_seconds, backoff)
    if config.jitter_seconds:
        jitter = secrets.randbits(53) / (1 << 53)
        delay += jitter * config.jitter_seconds
    return delay
