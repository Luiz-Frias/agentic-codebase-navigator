"""
Unit tests for passthrough LLM adapters (Azure, LiteLLM, Portkey) tool calling support.

These adapters use OpenAI-compatible formats for tools.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rlm.adapters.llm.azure_openai as azure_mod
import rlm.adapters.llm.litellm as litellm_mod
import rlm.adapters.llm.portkey as portkey_mod
from rlm.adapters.llm.azure_openai import AzureOpenAIAdapter
from rlm.adapters.llm.litellm import LiteLLMAdapter
from rlm.adapters.llm.portkey import PortkeyAdapter
from rlm.domain.agent_ports import ToolDefinition
from rlm.domain.models import LLMRequest

# =============================================================================
# Azure OpenAI Adapter Tests
# =============================================================================


@pytest.mark.unit
def test_azure_adapter_supports_tools_property() -> None:
    """AzureOpenAIAdapter should report supports_tools=True."""
    adapter = AzureOpenAIAdapter(deployment="my-deployment")
    assert adapter.supports_tools is True


@pytest.mark.unit
def test_azure_adapter_complete_with_tools_passes_to_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AzureOpenAIAdapter should pass tools to the API in OpenAI format."""
    captured_kwargs: list[dict] = []

    class _Client:
        def __init__(self, **_k):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            captured_kwargs.append(kwargs)
            return {
                "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    dummy_openai = SimpleNamespace(AzureOpenAI=_Client, AsyncAzureOpenAI=None)
    monkeypatch.setattr(azure_mod, "_require_openai", lambda: dummy_openai)

    tools: list[ToolDefinition] = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    ]

    adapter = AzureOpenAIAdapter(deployment="my-deploy")
    cc = adapter.complete(LLMRequest(prompt="Weather?", tools=tools, tool_choice="auto"))

    assert cc.response == "Hello!"
    assert len(captured_kwargs) == 1
    assert "tools" in captured_kwargs[0]
    assert captured_kwargs[0]["tools"][0]["type"] == "function"
    assert captured_kwargs[0]["tool_choice"] == "auto"


@pytest.mark.unit
def test_azure_adapter_complete_extracts_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AzureOpenAIAdapter should extract tool_calls from API response."""

    class _Client:
        def __init__(self, **_k):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_azure_123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Seattle"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    },
                ],
                "usage": {},
            }

    dummy_openai = SimpleNamespace(AzureOpenAI=_Client, AsyncAzureOpenAI=None)
    monkeypatch.setattr(azure_mod, "_require_openai", lambda: dummy_openai)

    adapter = AzureOpenAIAdapter(deployment="my-deploy")
    cc = adapter.complete(LLMRequest(prompt="Weather?"))

    assert cc.finish_reason == "tool_calls"
    assert cc.tool_calls is not None
    assert len(cc.tool_calls) == 1
    assert cc.tool_calls[0]["id"] == "call_azure_123"
    assert cc.tool_calls[0]["name"] == "get_weather"
    assert cc.tool_calls[0]["arguments"] == {"city": "Seattle"}


@pytest.mark.unit
async def test_azure_adapter_acomplete_with_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AzureOpenAIAdapter async path should handle tools."""
    captured_kwargs: list[dict] = []

    class _AsyncCompletions:
        async def create(self, **kwargs):
            captured_kwargs.append(kwargs)
            return {
                "choices": [{"message": {"content": "Async response"}, "finish_reason": "stop"}],
                "usage": {},
            }

    class _AsyncClient:
        def __init__(self, **_k):
            self.chat = SimpleNamespace(completions=_AsyncCompletions())

    dummy_openai = SimpleNamespace(AzureOpenAI=None, AsyncAzureOpenAI=_AsyncClient)
    monkeypatch.setattr(azure_mod, "_require_openai", lambda: dummy_openai)

    tools: list[ToolDefinition] = [
        {"name": "search", "description": "Search", "parameters": {"type": "object"}},
    ]

    adapter = AzureOpenAIAdapter(deployment="my-deploy")
    cc = await adapter.acomplete(LLMRequest(prompt="Search", tools=tools))

    assert cc.response == "Async response"
    assert "tools" in captured_kwargs[0]


# =============================================================================
# LiteLLM Adapter Tests
# =============================================================================


@pytest.mark.unit
def test_litellm_adapter_supports_tools_property() -> None:
    """LiteLLMAdapter should report supports_tools=True."""
    adapter = LiteLLMAdapter(model="gpt-4")
    assert adapter.supports_tools is True


@pytest.mark.unit
def test_litellm_adapter_complete_with_tools_passes_to_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LiteLLMAdapter should pass tools to the underlying library."""
    captured_kwargs: list[dict] = []

    def mock_completion(**kwargs):
        captured_kwargs.append(kwargs)
        return {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    dummy_litellm = SimpleNamespace(completion=mock_completion, acompletion=None)
    monkeypatch.setattr(litellm_mod, "_require_litellm", lambda: dummy_litellm)

    tools: list[ToolDefinition] = [
        {
            "name": "calculate",
            "description": "Do math",
            "parameters": {"type": "object", "properties": {}},
        },
    ]

    adapter = LiteLLMAdapter(model="gpt-4")
    cc = adapter.complete(LLMRequest(prompt="Calculate 2+2", tools=tools, tool_choice="required"))

    assert cc.response == "Hello!"
    assert len(captured_kwargs) == 1
    assert "tools" in captured_kwargs[0]
    assert captured_kwargs[0]["tool_choice"] == "required"


@pytest.mark.unit
def test_litellm_adapter_complete_extracts_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LiteLLMAdapter should extract tool_calls from response."""

    def mock_completion(**_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_litellm_1",
                                "type": "function",
                                "function": {
                                    "name": "calculate",
                                    "arguments": '{"a": 2, "b": 3}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
            "usage": {},
        }

    dummy_litellm = SimpleNamespace(completion=mock_completion, acompletion=None)
    monkeypatch.setattr(litellm_mod, "_require_litellm", lambda: dummy_litellm)

    adapter = LiteLLMAdapter(model="gpt-4")
    cc = adapter.complete(LLMRequest(prompt="Calculate"))

    assert cc.finish_reason == "tool_calls"
    assert cc.tool_calls is not None
    assert len(cc.tool_calls) == 1
    assert cc.tool_calls[0]["name"] == "calculate"
    assert cc.tool_calls[0]["arguments"] == {"a": 2, "b": 3}


@pytest.mark.unit
async def test_litellm_adapter_acomplete_with_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LiteLLMAdapter async path should handle tools."""
    captured_kwargs: list[dict] = []

    async def mock_acompletion(**kwargs):
        captured_kwargs.append(kwargs)
        return {
            "choices": [{"message": {"content": "Async calc"}, "finish_reason": "stop"}],
            "usage": {},
        }

    dummy_litellm = SimpleNamespace(completion=None, acompletion=mock_acompletion)
    monkeypatch.setattr(litellm_mod, "_require_litellm", lambda: dummy_litellm)

    tools: list[ToolDefinition] = [
        {"name": "calc", "description": "Math", "parameters": {"type": "object"}},
    ]

    adapter = LiteLLMAdapter(model="gpt-4")
    cc = await adapter.acomplete(LLMRequest(prompt="Calculate", tools=tools))

    assert cc.response == "Async calc"
    assert "tools" in captured_kwargs[0]


# =============================================================================
# Portkey Adapter Tests
# =============================================================================


@pytest.mark.unit
def test_portkey_adapter_supports_tools_property() -> None:
    """PortkeyAdapter should report supports_tools=True."""
    adapter = PortkeyAdapter(model="gpt-4")
    assert adapter.supports_tools is True


@pytest.mark.unit
def test_portkey_adapter_complete_with_tools_passes_to_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PortkeyAdapter should pass tools to the API."""
    captured_kwargs: list[dict] = []

    class _Client:
        def __init__(self, **_k):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            captured_kwargs.append(kwargs)
            return {
                "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    dummy_portkey = SimpleNamespace(Portkey=_Client, AsyncPortkey=None)
    monkeypatch.setattr(portkey_mod, "_require_portkey", lambda: dummy_portkey)

    tools: list[ToolDefinition] = [
        {
            "name": "search",
            "description": "Search",
            "parameters": {"type": "object", "properties": {}},
        },
    ]

    adapter = PortkeyAdapter(model="gpt-4")
    cc = adapter.complete(LLMRequest(prompt="Search", tools=tools, tool_choice="auto"))

    assert cc.response == "Hello!"
    assert len(captured_kwargs) == 1
    assert "tools" in captured_kwargs[0]
    assert captured_kwargs[0]["tool_choice"] == "auto"


@pytest.mark.unit
def test_portkey_adapter_complete_extracts_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PortkeyAdapter should extract tool_calls from response."""

    class _Client:
        def __init__(self, **_k):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_portkey_1",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query": "python"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    },
                ],
                "usage": {},
            }

    dummy_portkey = SimpleNamespace(Portkey=_Client, AsyncPortkey=None)
    monkeypatch.setattr(portkey_mod, "_require_portkey", lambda: dummy_portkey)

    adapter = PortkeyAdapter(model="gpt-4")
    cc = adapter.complete(LLMRequest(prompt="Search"))

    assert cc.finish_reason == "tool_calls"
    assert cc.tool_calls is not None
    assert len(cc.tool_calls) == 1
    assert cc.tool_calls[0]["id"] == "call_portkey_1"
    assert cc.tool_calls[0]["name"] == "search"
    assert cc.tool_calls[0]["arguments"] == {"query": "python"}


@pytest.mark.unit
async def test_portkey_adapter_acomplete_with_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PortkeyAdapter async path should handle tools."""
    captured_kwargs: list[dict] = []

    class _AsyncCompletions:
        async def create(self, **kwargs):
            captured_kwargs.append(kwargs)
            return {
                "choices": [{"message": {"content": "Async search"}, "finish_reason": "stop"}],
                "usage": {},
            }

    class _AsyncClient:
        def __init__(self, **_k):
            self.chat = SimpleNamespace(completions=_AsyncCompletions())

    dummy_portkey = SimpleNamespace(Portkey=None, AsyncPortkey=_AsyncClient)
    monkeypatch.setattr(portkey_mod, "_require_portkey", lambda: dummy_portkey)

    tools: list[ToolDefinition] = [
        {"name": "search", "description": "Search", "parameters": {"type": "object"}},
    ]

    adapter = PortkeyAdapter(model="gpt-4")
    cc = await adapter.acomplete(LLMRequest(prompt="Search", tools=tools))

    assert cc.response == "Async search"
    assert "tools" in captured_kwargs[0]
