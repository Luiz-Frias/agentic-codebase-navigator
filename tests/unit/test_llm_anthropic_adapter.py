"""Unit tests for AnthropicAdapter tool calling support."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import rlm.adapters.llm.anthropic as anthropic_mod
from rlm.adapters.llm.anthropic import AnthropicAdapter, build_anthropic_adapter
from rlm.domain.errors import LLMError
from rlm.domain.models import LLMRequest


@pytest.mark.unit
def test_build_anthropic_adapter_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty 'model'"):
        build_anthropic_adapter(model="")

    with pytest.raises(ValueError, match="api_key must be a non-empty string"):
        build_anthropic_adapter(model="m", api_key=" ")


@pytest.mark.unit
def test_anthropic_adapter_get_client_requires_expected_sdk_api() -> None:
    adapter = AnthropicAdapter(model="m")
    with pytest.raises(ImportError, match="expected `anthropic.Anthropic`"):
        adapter._get_client(SimpleNamespace())

    with pytest.raises(ImportError, match="expected `anthropic.AsyncAnthropic`"):
        adapter._get_async_client(SimpleNamespace())


@pytest.mark.unit
def test_anthropic_adapter_complete_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """AnthropicAdapter should handle basic completion."""

    class _Messages:
        def create(self, **_kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(text="Hello from Claude!")],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                stop_reason="end_turn",
            )

    class _Client:
        def __init__(self, **_kwargs):
            self.messages = _Messages()

    dummy_anthropic = SimpleNamespace(Anthropic=_Client, AsyncAnthropic=None)
    monkeypatch.setattr(anthropic_mod, "_require_anthropic", lambda: dummy_anthropic)

    adapter = AnthropicAdapter(model="claude-3")
    cc = adapter.complete(LLMRequest(prompt="Hello"))

    assert cc.response == "Hello from Claude!"
    assert cc.finish_reason == "stop"  # "end_turn" normalized to "stop"
    assert adapter.get_usage_summary().total_calls == 1


@pytest.mark.unit
def test_anthropic_adapter_complete_error_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """AnthropicAdapter should map SDK exceptions to LLMError."""

    class _Messages:
        def create(self, **_kwargs):
            raise TimeoutError()

    class _Client:
        def __init__(self, **_kwargs):
            self.messages = _Messages()

    dummy_anthropic = SimpleNamespace(Anthropic=_Client, AsyncAnthropic=None)
    monkeypatch.setattr(anthropic_mod, "_require_anthropic", lambda: dummy_anthropic)

    with pytest.raises(LLMError, match="timed out"):
        AnthropicAdapter(model="claude-3").complete(LLMRequest(prompt="hi"))


# =============================================================================
# Tool Calling Tests
# =============================================================================


@pytest.mark.unit
def test_anthropic_adapter_supports_tools_property() -> None:
    """AnthropicAdapter should report supports_tools=True."""
    adapter = AnthropicAdapter(model="claude-3")
    assert adapter.supports_tools is True


@pytest.mark.unit
def test_anthropic_adapter_complete_with_tools_passes_to_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AnthropicAdapter should pass tools to the API in Anthropic format."""
    from rlm.domain.agent_ports import ToolDefinition

    captured_kwargs: list[dict] = []

    class _Messages:
        def create(self, **kwargs):
            captured_kwargs.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(text="Hello!")],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                stop_reason="end_turn",
            )

    class _Client:
        def __init__(self, **_kwargs):
            self.messages = _Messages()

    dummy_anthropic = SimpleNamespace(Anthropic=_Client, AsyncAnthropic=None)
    monkeypatch.setattr(anthropic_mod, "_require_anthropic", lambda: dummy_anthropic)

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

    adapter = AnthropicAdapter(model="claude-3")
    cc = adapter.complete(LLMRequest(prompt="Weather in NYC?", tools=tools, tool_choice="required"))

    assert cc.response == "Hello!"
    assert cc.finish_reason == "stop"

    # Verify tools were passed to API in Anthropic format
    assert len(captured_kwargs) == 1
    api_call = captured_kwargs[0]
    assert "tools" in api_call
    assert api_call["tool_choice"] == {"type": "any"}
    assert len(api_call["tools"]) == 1
    assert api_call["tools"][0] == {
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }


@pytest.mark.unit
def test_anthropic_adapter_complete_extracts_tool_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AnthropicAdapter should extract tool_use blocks from response."""

    class _Messages:
        def create(self, **_kwargs):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="toolu_abc123",
                        name="get_weather",
                        input={"city": "NYC"},
                    )
                ],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                stop_reason="tool_use",
            )

    class _Client:
        def __init__(self, **_kwargs):
            self.messages = _Messages()

    dummy_anthropic = SimpleNamespace(Anthropic=_Client, AsyncAnthropic=None)
    monkeypatch.setattr(anthropic_mod, "_require_anthropic", lambda: dummy_anthropic)

    adapter = AnthropicAdapter(model="claude-3")
    cc = adapter.complete(LLMRequest(prompt="Weather?"))

    assert cc.finish_reason == "tool_calls"  # "tool_use" normalized to "tool_calls"
    assert cc.tool_calls is not None
    assert len(cc.tool_calls) == 1
    assert cc.tool_calls[0]["id"] == "toolu_abc123"
    assert cc.tool_calls[0]["name"] == "get_weather"
    assert cc.tool_calls[0]["arguments"] == {"city": "NYC"}
    assert cc.response == ""  # Empty when tool_calls present


@pytest.mark.unit
def test_anthropic_adapter_complete_mixed_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AnthropicAdapter should handle mixed text and tool_use content."""

    class _Messages:
        def create(self, **_kwargs):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="text", text="Let me check the weather."),
                    SimpleNamespace(
                        type="tool_use",
                        id="toolu_xyz",
                        name="get_weather",
                        input={"city": "London"},
                    ),
                ],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                stop_reason="tool_use",
            )

    class _Client:
        def __init__(self, **_kwargs):
            self.messages = _Messages()

    dummy_anthropic = SimpleNamespace(Anthropic=_Client, AsyncAnthropic=None)
    monkeypatch.setattr(anthropic_mod, "_require_anthropic", lambda: dummy_anthropic)

    adapter = AnthropicAdapter(model="claude-3")
    cc = adapter.complete(LLMRequest(prompt="Weather in London?"))

    assert cc.tool_calls is not None
    assert len(cc.tool_calls) == 1
    assert cc.tool_calls[0]["name"] == "get_weather"
    # Text content should still be extracted
    assert cc.response == "Let me check the weather."


@pytest.mark.unit
async def test_anthropic_adapter_acomplete_with_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AnthropicAdapter async path should handle tools."""
    from rlm.domain.agent_ports import ToolDefinition

    captured_kwargs: list[dict] = []

    class _AsyncMessages:
        async def create(self, **kwargs):
            captured_kwargs.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(text="Async response!")],
                usage=SimpleNamespace(input_tokens=5, output_tokens=3),
                stop_reason="end_turn",
            )

    class _AsyncClient:
        def __init__(self, **_kwargs):
            self.messages = _AsyncMessages()

    dummy_anthropic = SimpleNamespace(Anthropic=None, AsyncAnthropic=_AsyncClient)
    monkeypatch.setattr(anthropic_mod, "_require_anthropic", lambda: dummy_anthropic)

    tools: list[ToolDefinition] = [
        {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    adapter = AnthropicAdapter(model="claude-3")
    cc = await adapter.acomplete(LLMRequest(prompt="Search", tools=tools, tool_choice="required"))

    assert cc.response == "Async response!"
    assert len(captured_kwargs) == 1
    assert "tools" in captured_kwargs[0]
    assert captured_kwargs[0]["tool_choice"] == {"type": "any"}


@pytest.mark.unit
async def test_anthropic_adapter_acomplete_extracts_tool_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AnthropicAdapter async path should extract tool_use blocks."""

    class _AsyncMessages:
        async def create(self, **_kwargs):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="async_call_1",
                        name="calculate",
                        input={"a": 5, "b": 3},
                    )
                ],
                usage=SimpleNamespace(input_tokens=5, output_tokens=3),
                stop_reason="tool_use",
            )

    class _AsyncClient:
        def __init__(self, **_kwargs):
            self.messages = _AsyncMessages()

    dummy_anthropic = SimpleNamespace(Anthropic=None, AsyncAnthropic=_AsyncClient)
    monkeypatch.setattr(anthropic_mod, "_require_anthropic", lambda: dummy_anthropic)

    adapter = AnthropicAdapter(model="claude-3")
    cc = await adapter.acomplete(LLMRequest(prompt="Calculate"))

    assert cc.tool_calls is not None
    assert len(cc.tool_calls) == 1
    assert cc.tool_calls[0]["id"] == "async_call_1"
    assert cc.tool_calls[0]["name"] == "calculate"
    assert cc.tool_calls[0]["arguments"] == {"a": 5, "b": 3}
