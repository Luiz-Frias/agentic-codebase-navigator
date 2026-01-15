from __future__ import annotations

from types import SimpleNamespace

import pytest

import rlm.adapters.llm.openai as openai_mod
from rlm.adapters.llm.openai import (
    OpenAIAdapter,
    _safe_openai_error_message,
    build_openai_adapter,
)
from rlm.domain.errors import LLMError
from rlm.domain.models import LLMRequest


@pytest.mark.unit
def test_safe_openai_error_message_classifies_common_errors() -> None:
    assert _safe_openai_error_message(TimeoutError()) == "OpenAI request timed out"
    assert _safe_openai_error_message(ConnectionError()) == "OpenAI connection error"
    assert _safe_openai_error_message(OSError()) == "OpenAI connection error"

    class AuthenticationError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    class BadRequestError(Exception):
        pass

    class UnprocessableEntityError(Exception):
        pass

    assert (
        _safe_openai_error_message(AuthenticationError())
        == "OpenAI authentication failed (check OPENAI_API_KEY)"
    )
    assert _safe_openai_error_message(RateLimitError()) == "OpenAI rate limit exceeded"
    assert _safe_openai_error_message(BadRequestError()) == "OpenAI request rejected"
    assert _safe_openai_error_message(UnprocessableEntityError()) == "OpenAI request rejected"
    assert _safe_openai_error_message(RuntimeError("boom")) == "OpenAI request failed"


@pytest.mark.unit
def test_build_openai_adapter_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty 'model'"):
        build_openai_adapter(model="")

    with pytest.raises(ValueError, match="api_key must be a non-empty string"):
        build_openai_adapter(model="m", api_key=" ")


@pytest.mark.unit
def test_openai_adapter_get_client_requires_expected_sdk_api() -> None:
    adapter = OpenAIAdapter(model="m")
    with pytest.raises(ImportError, match="expected `openai.OpenAI`"):
        adapter._get_client(SimpleNamespace())

    with pytest.raises(ImportError, match="expected `openai.AsyncOpenAI`"):
        adapter._get_async_client(SimpleNamespace())


@pytest.mark.unit
def test_openai_adapter_complete_success_and_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resp_ok = {
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }

    class _Client:
        def __init__(self, *, resp, exc: Exception | None = None) -> None:
            self._resp = resp
            self._exc = exc
            self.chat = SimpleNamespace(completions=self)

        def create(self, **_kwargs):
            if self._exc is not None:
                raise self._exc
            return self._resp

    created: list[dict] = []

    class OpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=_Client(resp=resp_ok).create)
            )

    dummy_openai = SimpleNamespace(OpenAI=OpenAI, AsyncOpenAI=None)
    monkeypatch.setattr(openai_mod, "_require_openai", lambda: dummy_openai)

    adapter = OpenAIAdapter(model="m", api_key="k", base_url="u")
    cc = adapter.complete(LLMRequest(prompt="hi"))
    assert cc.response == "hello"
    assert adapter.get_usage_summary().total_calls == 1
    assert created == [{"api_key": "k", "base_url": "u"}]

    # Provider exception => mapped LLMError.
    class OpenAI2:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=_Client(resp=resp_ok, exc=TimeoutError()).create)
            )

    dummy_openai2 = SimpleNamespace(OpenAI=OpenAI2, AsyncOpenAI=None)
    monkeypatch.setattr(openai_mod, "_require_openai", lambda: dummy_openai2)

    with pytest.raises(LLMError, match="timed out"):
        OpenAIAdapter(model="m").complete(LLMRequest(prompt="hi"))

    # Invalid response => mapped LLMError.
    class OpenAI3:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_Client(resp={}).create))

    dummy_openai3 = SimpleNamespace(OpenAI=OpenAI3, AsyncOpenAI=None)
    monkeypatch.setattr(openai_mod, "_require_openai", lambda: dummy_openai3)

    with pytest.raises(LLMError, match="response invalid"):
        OpenAIAdapter(model="m").complete(LLMRequest(prompt="hi"))


@pytest.mark.unit
async def test_openai_adapter_acomplete_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resp_ok = {
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }

    class _AsyncCompletions:
        async def create(self, **_kwargs):
            return resp_ok

    class _AsyncClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=_AsyncCompletions())

    dummy_openai = SimpleNamespace(OpenAI=None, AsyncOpenAI=_AsyncClient)
    monkeypatch.setattr(openai_mod, "_require_openai", lambda: dummy_openai)

    adapter = OpenAIAdapter(model="m")
    cc = await adapter.acomplete(LLMRequest(prompt="hi"))
    assert cc.response == "hello"
    assert adapter.get_last_usage().total_calls == 1


@pytest.mark.unit
async def test_openai_adapter_acomplete_error_invalid_response_and_client_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_kwargs: list[dict] = []

    class _AsyncCompletionsOK:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **_kwargs):
            self.calls += 1
            return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    class _AsyncClientOK:
        def __init__(self, **kwargs):
            created_kwargs.append(kwargs)
            self.completions = _AsyncCompletionsOK()
            self.chat = SimpleNamespace(completions=self.completions)

    dummy_openai_ok = SimpleNamespace(OpenAI=None, AsyncOpenAI=_AsyncClientOK)
    monkeypatch.setattr(openai_mod, "_require_openai", lambda: dummy_openai_ok)

    adapter = OpenAIAdapter(model="m", api_key="k", base_url="u")
    cc1 = await adapter.acomplete(LLMRequest(prompt="hi"))
    cc2 = await adapter.acomplete(LLMRequest(prompt="hi"))
    assert cc1.response == "ok"
    assert cc2.response == "ok"
    # Client constructed once and cached.
    assert created_kwargs == [{"api_key": "k", "base_url": "u"}]

    class _AsyncCompletionsTimeout:
        async def create(self, **_kwargs):
            raise TimeoutError()

    class _AsyncClientTimeout:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=_AsyncCompletionsTimeout())

    dummy_openai_timeout = SimpleNamespace(OpenAI=None, AsyncOpenAI=_AsyncClientTimeout)
    monkeypatch.setattr(openai_mod, "_require_openai", lambda: dummy_openai_timeout)

    with pytest.raises(LLMError, match="timed out"):
        await OpenAIAdapter(model="m").acomplete(LLMRequest(prompt="hi"))

    class _AsyncCompletionsBad:
        async def create(self, **_kwargs):
            return {}

    class _AsyncClientBad:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=_AsyncCompletionsBad())

    dummy_openai_bad = SimpleNamespace(OpenAI=None, AsyncOpenAI=_AsyncClientBad)
    monkeypatch.setattr(openai_mod, "_require_openai", lambda: dummy_openai_bad)

    with pytest.raises(LLMError, match="response invalid"):
        await OpenAIAdapter(model="m").acomplete(LLMRequest(prompt="hi"))


# =============================================================================
# Phase 2: Tool Calling Tests
# =============================================================================


@pytest.mark.unit
def test_openai_adapter_supports_tools_property() -> None:
    """OpenAIAdapter should report supports_tools=True."""
    adapter = OpenAIAdapter(model="gpt-4")
    assert adapter.supports_tools is True


@pytest.mark.unit
def test_openai_adapter_complete_with_tools_passes_to_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAIAdapter should pass tools to the API in OpenAI format."""
    from rlm.domain.agent_ports import ToolDefinition

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

    dummy_openai = SimpleNamespace(OpenAI=_Client, AsyncOpenAI=None)
    monkeypatch.setattr(openai_mod, "_require_openai", lambda: dummy_openai)

    tools: list[ToolDefinition] = [
        {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]

    adapter = OpenAIAdapter(model="gpt-4")
    cc = adapter.complete(LLMRequest(prompt="Weather in NYC?", tools=tools, tool_choice="auto"))

    assert cc.response == "Hello!"
    assert cc.finish_reason == "stop"
    assert cc.tool_calls is None

    # Verify tools were passed to API in OpenAI format
    assert len(captured_kwargs) == 1
    api_call = captured_kwargs[0]
    assert "tools" in api_call
    assert len(api_call["tools"]) == 1
    assert api_call["tools"][0] == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    assert api_call["tool_choice"] == "auto"


@pytest.mark.unit
def test_openai_adapter_complete_extracts_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAIAdapter should extract tool_calls from API response."""

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
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "NYC"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    dummy_openai = SimpleNamespace(OpenAI=_Client, AsyncOpenAI=None)
    monkeypatch.setattr(openai_mod, "_require_openai", lambda: dummy_openai)

    adapter = OpenAIAdapter(model="gpt-4")
    cc = adapter.complete(LLMRequest(prompt="Weather?"))

    assert cc.finish_reason == "tool_calls"
    assert cc.tool_calls is not None
    assert len(cc.tool_calls) == 1
    assert cc.tool_calls[0]["id"] == "call_abc123"
    assert cc.tool_calls[0]["name"] == "get_weather"
    assert cc.tool_calls[0]["arguments"] == {"city": "NYC"}
    assert cc.response == ""  # Empty when tool_calls present


@pytest.mark.unit
async def test_openai_adapter_acomplete_with_tools_passes_to_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAIAdapter async path should pass tools to the API in OpenAI format."""
    from rlm.domain.agent_ports import ToolDefinition

    captured_kwargs: list[dict] = []

    class _AsyncCompletions:
        async def create(self, **kwargs):
            captured_kwargs.append(kwargs)
            return {
                "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    class _AsyncClient:
        def __init__(self, **_k):
            self.chat = SimpleNamespace(completions=_AsyncCompletions())

    dummy_openai = SimpleNamespace(OpenAI=None, AsyncOpenAI=_AsyncClient)
    monkeypatch.setattr(openai_mod, "_require_openai", lambda: dummy_openai)

    tools: list[ToolDefinition] = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    adapter = OpenAIAdapter(model="gpt-4")
    cc = await adapter.acomplete(LLMRequest(prompt="hi", tools=tools, tool_choice="required"))

    assert cc.response == "Hello!"
    assert len(captured_kwargs) == 1
    assert "tools" in captured_kwargs[0]
    assert captured_kwargs[0]["tool_choice"] == "required"


@pytest.mark.unit
async def test_openai_adapter_acomplete_extracts_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAIAdapter async path should extract tool_calls from API response."""

    class _AsyncCompletions:
        async def create(self, **_kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_xyz789",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query": "python"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {},
            }

    class _AsyncClient:
        def __init__(self, **_k):
            self.chat = SimpleNamespace(completions=_AsyncCompletions())

    dummy_openai = SimpleNamespace(OpenAI=None, AsyncOpenAI=_AsyncClient)
    monkeypatch.setattr(openai_mod, "_require_openai", lambda: dummy_openai)

    adapter = OpenAIAdapter(model="gpt-4")
    cc = await adapter.acomplete(LLMRequest(prompt="search"))

    assert cc.finish_reason == "tool_calls"
    assert cc.tool_calls is not None
    assert len(cc.tool_calls) == 1
    assert cc.tool_calls[0]["name"] == "search"
    assert cc.tool_calls[0]["arguments"] == {"query": "python"}
