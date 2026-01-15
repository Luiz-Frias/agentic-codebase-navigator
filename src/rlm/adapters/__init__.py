"""
Adapters layer (hexagonal).

Concrete implementations of ports for external systems (LLM providers, envs, logging).
"""

from __future__ import annotations

from rlm.adapters.base import (
    BaseBrokerAdapter,
    BaseEnvironmentAdapter,
    BaseLLMAdapter,
    BaseLoggerAdapter,
    BaseStructuredOutputAdapter,
    BaseToolAdapter,
    BaseToolRegistryAdapter,
)
from rlm.adapters.tools import (
    InMemoryToolRegistry,
    NativeToolAdapter,
    PydanticOutputAdapter,
)

__all__ = [
    # Base classes
    "BaseBrokerAdapter",
    "BaseEnvironmentAdapter",
    "BaseLLMAdapter",
    "BaseLoggerAdapter",
    "BaseStructuredOutputAdapter",
    "BaseToolAdapter",
    "BaseToolRegistryAdapter",
    # Tool adapters
    "InMemoryToolRegistry",
    "NativeToolAdapter",
    "PydanticOutputAdapter",
]
