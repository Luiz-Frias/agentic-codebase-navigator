"""
LLM provider adapters.

Phase 4 starts by introducing a deterministic, dependency-free mock adapter that
implements the `LLMPort` contract and can be used by tests and examples.
"""

from __future__ import annotations

from rlm.adapters.llm.anthropic import AnthropicAdapter
from rlm.adapters.llm.azure_openai import AzureOpenAIAdapter
from rlm.adapters.llm.gemini import GeminiAdapter
from rlm.adapters.llm.litellm import LiteLLMAdapter
from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.adapters.llm.openai import OpenAIAdapter
from rlm.adapters.llm.portkey import PortkeyAdapter

__all__ = [
    "AnthropicAdapter",
    "AzureOpenAIAdapter",
    "GeminiAdapter",
    "LiteLLMAdapter",
    "MockLLMAdapter",
    "OpenAIAdapter",
    "PortkeyAdapter",
]
