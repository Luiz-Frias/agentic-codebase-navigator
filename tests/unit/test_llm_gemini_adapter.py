from __future__ import annotations

from types import SimpleNamespace

import pytest

import rlm.adapters.llm.gemini as gemini_mod
from rlm.adapters.llm.gemini import (
    GeminiAdapter,
    _extract_text,
    _extract_usage_tokens,
    _prompt_to_gemini_contents,
    build_gemini_adapter,
)
from rlm.domain.errors import LLMError
from rlm.domain.models import LLMRequest


@pytest.mark.unit
def test_build_gemini_adapter_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty 'model'"):
        build_gemini_adapter(model="")
    with pytest.raises(ValueError, match="api_key must be a non-empty string"):
        build_gemini_adapter(model="m", api_key=" ")


@pytest.mark.unit
def test_gemini_adapter_reports_tool_prompt_format() -> None:
    adapter = GeminiAdapter(model="m")
    assert adapter.tool_prompt_format == "gemini"


@pytest.mark.unit
def test_extract_text_supports_multiple_shapes() -> None:
    assert _extract_text({"text": "hi"}) == "hi"

    class Resp:
        text = "hello"

    assert _extract_text(Resp()) == "hello"

    payload = {"candidates": [{"content": {"parts": [{"text": "p0"}]}}]}
    assert _extract_text(payload) == "p0"

    with pytest.raises(ValueError, match="missing text"):
        _extract_text({})


@pytest.mark.unit
def test_extract_usage_tokens_supports_multiple_shapes() -> None:
    assert _extract_usage_tokens({}) == (0, 0)

    assert _extract_usage_tokens(
        {"usage_metadata": {"prompt_token_count": 1, "candidates_token_count": 2}}
    ) == (
        1,
        2,
    )
    assert _extract_usage_tokens(
        {"usage_metadata": {"input_token_count": "3", "output_token_count": "4"}}
    ) == (
        3,
        4,
    )

    class Usage:
        prompt_token_count = 5
        candidates_token_count = 6

    class Resp:
        usage_metadata = Usage()

    assert _extract_usage_tokens(Resp()) == (5, 6)

    class BadUsage:
        input_token_count = "nope"
        output_token_count = None

    class Resp2:
        usage_metadata = BadUsage()

    assert _extract_usage_tokens(Resp2()) == (0, 0)


@pytest.mark.unit
def test_prompt_to_gemini_contents_maps_tool_messages() -> None:
    prompt = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 72}'},
    ]

    contents = _prompt_to_gemini_contents(prompt)

    assert isinstance(contents, list)
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"
    func_call = contents[1]["parts"][-1]["function_call"]
    assert func_call["name"] == "get_weather"
    assert func_call["args"] == {"city": "NYC"}
    func_resp = contents[2]["parts"][0]["function_response"]
    assert func_resp["name"] == "get_weather"
    assert func_resp["response"] == {"temp": 72}


@pytest.mark.unit
def test_gemini_adapter_complete_success_error_and_client_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict] = []

    class _Models:
        def __init__(self, *, resp, exc: Exception | None = None) -> None:
            self._resp = resp
            self._exc = exc

        def generate_content(self, **_kwargs):
            if self._exc is not None:
                raise self._exc
            return self._resp

    class Client:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.models = _Models(
                resp={
                    "text": "ok",
                    "usage_metadata": {
                        "prompt_token_count": 1,
                        "candidates_token_count": 2,
                    },
                }
            )

    dummy_genai = SimpleNamespace(Client=Client)
    monkeypatch.setattr(gemini_mod, "_require_google_genai", lambda: dummy_genai)

    adapter = GeminiAdapter(model="m", api_key="k")
    cc1 = adapter.complete(LLMRequest(prompt="hi"))
    cc2 = adapter.complete(LLMRequest(prompt="hi"))
    assert cc1.response == "ok"
    assert cc2.response == "ok"
    # Client created once and cached.
    assert created == [{"api_key": "k"}]

    class ClientTimeout:
        def __init__(self, **_kwargs):
            self.models = _Models(resp={}, exc=TimeoutError())

    dummy_genai2 = SimpleNamespace(Client=ClientTimeout)
    monkeypatch.setattr(gemini_mod, "_require_google_genai", lambda: dummy_genai2)

    with pytest.raises(LLMError, match="Gemini request timed out"):
        GeminiAdapter(model="m").complete(LLMRequest(prompt="hi"))

    class ClientBadText:
        def __init__(self, **_kwargs):
            self.models = _Models(resp={})

    dummy_genai3 = SimpleNamespace(Client=ClientBadText)
    monkeypatch.setattr(gemini_mod, "_require_google_genai", lambda: dummy_genai3)

    with pytest.raises(ValueError, match="missing text"):
        # _extract_text raises ValueError; adapter does not wrap it (intentional).
        GeminiAdapter(model="m").complete(LLMRequest(prompt="hi"))


@pytest.mark.unit
def test_gemini_adapter_get_client_requires_expected_sdk_api() -> None:
    adapter = GeminiAdapter(model="m")
    with pytest.raises(ImportError, match="expected `google\\.genai\\.Client`"):
        adapter._get_client(SimpleNamespace())


@pytest.mark.unit
async def test_gemini_adapter_acomplete_runs_sync_path_in_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        def __init__(self, **_kwargs):
            self.models = SimpleNamespace(
                generate_content=lambda **_k: {"text": "ok"}  # noqa: ARG005
            )

    dummy_genai = SimpleNamespace(Client=Client)
    monkeypatch.setattr(gemini_mod, "_require_google_genai", lambda: dummy_genai)

    cc = await GeminiAdapter(model="m").acomplete(LLMRequest(prompt="hi"))
    assert cc.response == "ok"


# =============================================================================
# Phase 2: Tool Calling Tests
# =============================================================================


@pytest.mark.unit
def test_gemini_adapter_supports_tools_property() -> None:
    """GeminiAdapter should report supports_tools=True."""
    adapter = GeminiAdapter(model="gemini-pro")
    assert adapter.supports_tools is True


@pytest.mark.unit
def test_gemini_adapter_complete_with_tools_passes_to_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GeminiAdapter should pass tools to the API in Gemini format."""
    from rlm.domain.agent_ports import ToolDefinition

    captured_kwargs: list[dict] = []

    class _Models:
        def generate_content(self, **kwargs):
            captured_kwargs.append(kwargs)
            return {
                "text": "Hello!",
                "usage_metadata": {"prompt_token_count": 10, "candidates_token_count": 5},
            }

    class Client:
        def __init__(self, **_kwargs):
            self.models = _Models()

    dummy_genai = SimpleNamespace(Client=Client)
    monkeypatch.setattr(gemini_mod, "_require_google_genai", lambda: dummy_genai)

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

    adapter = GeminiAdapter(model="gemini-pro")
    cc = adapter.complete(LLMRequest(prompt="Weather in NYC?", tools=tools))

    assert cc.response == "Hello!"

    # Verify tools were passed to API in Gemini format (wrapped in function_declarations)
    assert len(captured_kwargs) == 1
    api_call = captured_kwargs[0]
    assert "tools" in api_call
    assert len(api_call["tools"]) == 1
    assert "function_declarations" in api_call["tools"][0]
    func_decls = api_call["tools"][0]["function_declarations"]
    assert len(func_decls) == 1
    assert func_decls[0] == {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }


@pytest.mark.unit
def test_gemini_adapter_complete_extracts_function_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GeminiAdapter should extract functionCall parts from response."""

    class _Models:
        def generate_content(self, **_kwargs):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "function_call": {
                                        "name": "get_weather",
                                        "args": {"city": "NYC"},
                                    }
                                }
                            ]
                        },
                        "finish_reason": "STOP",
                    }
                ],
                "usage_metadata": {"prompt_token_count": 10, "candidates_token_count": 5},
            }

    class Client:
        def __init__(self, **_kwargs):
            self.models = _Models()

    dummy_genai = SimpleNamespace(Client=Client)
    monkeypatch.setattr(gemini_mod, "_require_google_genai", lambda: dummy_genai)

    adapter = GeminiAdapter(model="gemini-pro")
    cc = adapter.complete(LLMRequest(prompt="Weather?"))

    assert cc.finish_reason == "stop"  # "STOP" normalized to "stop"
    assert cc.tool_calls is not None
    assert len(cc.tool_calls) == 1
    # Gemini doesn't provide IDs, so we generate them
    assert cc.tool_calls[0]["id"].startswith("gemini_call_")
    assert cc.tool_calls[0]["name"] == "get_weather"
    assert cc.tool_calls[0]["arguments"] == {"city": "NYC"}
    assert cc.response == ""  # Empty when tool_calls present


@pytest.mark.unit
def test_gemini_adapter_complete_multiple_function_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GeminiAdapter should handle multiple function calls in response."""

    class _Models:
        def generate_content(self, **_kwargs):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"function_call": {"name": "get_weather", "args": {"city": "NYC"}}},
                                {"function_call": {"name": "get_time", "args": {"tz": "EST"}}},
                            ]
                        },
                        "finish_reason": "STOP",
                    }
                ],
                "usage_metadata": {},
            }

    class Client:
        def __init__(self, **_kwargs):
            self.models = _Models()

    dummy_genai = SimpleNamespace(Client=Client)
    monkeypatch.setattr(gemini_mod, "_require_google_genai", lambda: dummy_genai)

    adapter = GeminiAdapter(model="gemini-pro")
    cc = adapter.complete(LLMRequest(prompt="Weather and time?"))

    assert cc.tool_calls is not None
    assert len(cc.tool_calls) == 2
    assert cc.tool_calls[0]["name"] == "get_weather"
    assert cc.tool_calls[1]["name"] == "get_time"


@pytest.mark.unit
def test_gemini_adapter_complete_mixed_text_and_function_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GeminiAdapter should handle mixed text and function call parts."""

    class _Models:
        def generate_content(self, **_kwargs):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Let me check that for you."},
                                {"function_call": {"name": "search", "args": {"q": "python"}}},
                            ]
                        },
                        "finish_reason": "STOP",
                    }
                ],
                "usage_metadata": {},
            }

    class Client:
        def __init__(self, **_kwargs):
            self.models = _Models()

    dummy_genai = SimpleNamespace(Client=Client)
    monkeypatch.setattr(gemini_mod, "_require_google_genai", lambda: dummy_genai)

    adapter = GeminiAdapter(model="gemini-pro")
    cc = adapter.complete(LLMRequest(prompt="Search python"))

    assert cc.tool_calls is not None
    assert len(cc.tool_calls) == 1
    assert cc.tool_calls[0]["name"] == "search"
    # Text should still be extracted
    assert cc.response == "Let me check that for you."


@pytest.mark.unit
async def test_gemini_adapter_acomplete_with_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GeminiAdapter async path should handle tools (via sync-to-thread)."""
    from rlm.domain.agent_ports import ToolDefinition

    captured_kwargs: list[dict] = []

    class _Models:
        def generate_content(self, **kwargs):
            captured_kwargs.append(kwargs)
            return {"text": "Async response!"}

    class Client:
        def __init__(self, **_kwargs):
            self.models = _Models()

    dummy_genai = SimpleNamespace(Client=Client)
    monkeypatch.setattr(gemini_mod, "_require_google_genai", lambda: dummy_genai)

    tools: list[ToolDefinition] = [
        {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    adapter = GeminiAdapter(model="gemini-pro")
    cc = await adapter.acomplete(LLMRequest(prompt="Search", tools=tools))

    assert cc.response == "Async response!"
    assert len(captured_kwargs) == 1
    assert "tools" in captured_kwargs[0]
