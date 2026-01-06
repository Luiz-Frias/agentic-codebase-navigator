from __future__ import annotations


class RLMError(Exception):
    """Base exception for domain-level failures."""


class ValidationError(RLMError):
    """Raised when user/config input is invalid."""


class ExecutionError(RLMError):
    """Raised when code execution in an environment fails."""


class BrokerError(RLMError):
    """Raised when broker transport/protocol fails."""


class LLMError(RLMError):
    """Raised when an LLM provider call fails."""
