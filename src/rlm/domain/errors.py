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


class ToolNotFoundError(RLMError):
    """Raised when a tool requested by the LLM is not in the registry."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' not found in registry")


class ToolExecutionError(RLMError):
    """Raised when tool execution fails and preserves original exception."""

    def __init__(
        self,
        tool_name: str,
        message: str,
        original_exception: Exception | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.original_exception = original_exception
        if original_exception is not None:
            self.__cause__ = original_exception
        super().__init__(f"Tool '{tool_name}' execution failed: {message}")
