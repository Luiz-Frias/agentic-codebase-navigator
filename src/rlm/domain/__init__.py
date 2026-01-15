"""
Domain layer (hexagonal core).

Pure business logic and ports (no adapters, no infrastructure, no third-party deps).
"""

from __future__ import annotations

from rlm.domain.agent_ports import (
    StructuredOutputPort,
    ToolCallRequest,
    ToolCallResult,
    ToolDefinition,
    ToolPort,
    ToolRegistryPort,
)
from rlm.domain.errors import (
    BrokerError,
    ExecutionError,
    LLMError,
    RLMError,
    ValidationError,
)
from rlm.domain.ports import BrokerPort, EnvironmentPort, LLMPort, LoggerPort

__all__ = [
    # Errors
    "BrokerError",
    "ExecutionError",
    "LLMError",
    "RLMError",
    "ValidationError",
    # Core Ports
    "BrokerPort",
    "EnvironmentPort",
    "LLMPort",
    "LoggerPort",
    # Agent Ports
    "StructuredOutputPort",
    "ToolCallRequest",
    "ToolCallResult",
    "ToolDefinition",
    "ToolPort",
    "ToolRegistryPort",
]
