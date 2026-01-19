"""
Tool adapters for agent capabilities.

This module provides concrete implementations of ToolPort and ToolRegistryPort
for use with RLM's tool calling mode.
"""

from __future__ import annotations

from rlm.adapters.tools.native import NativeToolAdapter
from rlm.adapters.tools.pydantic_output import PydanticOutputAdapter
from rlm.adapters.tools.registry import InMemoryToolRegistry

__all__ = [
    "InMemoryToolRegistry",
    "NativeToolAdapter",
    "PydanticOutputAdapter",
]
