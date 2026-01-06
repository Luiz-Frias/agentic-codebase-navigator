"""
Legacy adapters (temporary).

These adapters wrap `rlm._legacy.*` implementations and expose the hexagonal
domain ports. They exist only to bridge Phase 1/2 while we migrate behavior into
domain/application layers.
"""

from __future__ import annotations

from rlm.adapters.legacy.broker import LegacyBrokerAdapter
from rlm.adapters.legacy.environment import LegacyEnvironmentAdapter
from rlm.adapters.legacy.llm import LegacyLLMPortAdapter
from rlm.adapters.legacy.logger import LegacyLoggerAdapter

__all__ = [
    "LegacyBrokerAdapter",
    "LegacyEnvironmentAdapter",
    "LegacyLLMPortAdapter",
    "LegacyLoggerAdapter",
]
