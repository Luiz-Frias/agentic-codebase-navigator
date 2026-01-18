from __future__ import annotations

import pytest

from rlm.adapters.llm.provider_base import (
    UsageTracker,
    extract_finish_reason_anthropic,
    extract_finish_reason_gemini,
    extract_finish_reason_openai,
    extract_openai_style_token_usage,
    extract_text_from_chat_response,
    extract_tool_calls_anthropic,
    extract_tool_calls_gemini,
    extract_tool_calls_openai,
    prompt_to_messages,
    prompt_to_text,
    safe_provider_error_message,
    tool_choice_to_anthropic_format,
    tool_choice_to_gemini_function_calling_config,
    tool_choice_to_openai_format,
    tool_definition_to_anthropic_format,
    tool_definition_to_gemini_format,
    tool_definition_to_openai_format,
)


@pytest.mark.unit
def test_safe_provider_error_message_classifies_common_errors() -> None:
    assert safe_provider_error_message("x", TimeoutError()) == "x request timed out"
    assert safe_provider_error_message("x", ConnectionError()) == "x connection error"
    assert safe_provider_error_message("x", OSError()) == "x connection error"
    assert safe_provider_error_message("x", RuntimeError("boom")) == "x request failed"


@pytest.mark.unit
def test_prompt_to_messages_accepts_multiple_shapes() -> None:
    assert prompt_to_messages("hi") == [{"role": "user", "content": "hi"}]

    msgs = [{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]
    assert prompt_to_messages(msgs) == msgs

    assert prompt_to_messages([1, 2]) == [{"role": "user", "content": str([1, 2])}]

    assert prompt_to_messages({"messages": msgs}) == msgs
    assert prompt_to_messages({"prompt": "p"}) == [{"role": "user", "content": "p"}]
    assert prompt_to_messages({"content": "c"}) == [{"role": "user", "content": "c"}]
    assert prompt_to_messages({"x": 1}) == [{"role": "user", "content": str({"x": 1})}]

    assert prompt_to_messages(123) == [{"role": "user", "content": "123"}]  # type: ignore[arg-type]


@pytest.mark.unit
def test_prompt_to_text_accepts_multiple_shapes() -> None:
    assert prompt_to_text("hi") == "hi"

    msgs = [{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]
    assert prompt_to_text(msgs) == "system: a\nuser: b"

    assert prompt_to_text([1, 2]) == str([1, 2])

    assert prompt_to_text({"prompt": "p"}) == "p"
    assert prompt_to_text({"content": "c"}) == "c"
    assert prompt_to_text({"messages": msgs}) == "system: a\nuser: b"
    assert prompt_to_text({"x": 1}) == str({"x": 1})

    assert prompt_to_text(123) == "123"  # type: ignore[arg-type]


@pytest.mark.unit
def test_extract_text_from_chat_response_supports_object_and_dict_payloads() -> None:
    assert extract_text_from_chat_response("x") == "x"

    class Msg:
        def __init__(self, content: str) -> None:
            self.content = content

    class Choice:
        def __init__(self, message: Msg) -> None:
            self.message = message

    class Resp:
        def __init__(self) -> None:
            self.choices = [Choice(Msg("hello"))]

    assert extract_text_from_chat_response(Resp()) == "hello"

    payload = {"choices": [{"message": {"content": "hi"}}]}
    assert extract_text_from_chat_response(payload) == "hi"

    payload2 = {"choices": [{"text": "plain"}]}
    assert extract_text_from_chat_response(payload2) == "plain"


@pytest.mark.unit
def test_extract_text_from_chat_response_raises_on_missing_choices_or_content() -> None:
    with pytest.raises(ValueError, match="missing choices"):
        extract_text_from_chat_response({"choices": []})

    with pytest.raises(ValueError, match="missing message content"):
        extract_text_from_chat_response({"choices": [{"message": {}}]})


@pytest.mark.unit
def test_extract_openai_style_token_usage_handles_dict_object_and_missing_usage() -> None:
    assert extract_openai_style_token_usage({}) == (0, 0)

    assert extract_openai_style_token_usage(
        {"usage": {"prompt_tokens": 1, "completion_tokens": 2}},
    ) == (
        1,
        2,
    )
    assert extract_openai_style_token_usage(
        {"usage": {"input_tokens": "3", "output_tokens": "4"}},
    ) == (
        3,
        4,
    )

    class Usage:
        def __init__(self) -> None:
            self.prompt_tokens = 5
            self.completion_tokens = 6

    class Resp:
        def __init__(self) -> None:
            self.usage = Usage()

    assert extract_openai_style_token_usage(Resp()) == (5, 6)

    class BadUsage:
        def __init__(self) -> None:
            self.input_tokens = "nope"
            self.output_tokens = None

    class Resp2:
        def __init__(self) -> None:
            self.usage = BadUsage()

    assert extract_openai_style_token_usage(Resp2()) == (0, 0)


@pytest.mark.unit
def test_usage_tracker_tracks_totals_and_last_call() -> None:
    tracker = UsageTracker()
    last = tracker.record("m", input_tokens=2, output_tokens=3)
    assert last.total_calls == 1
    assert last.total_input_tokens == 2
    assert last.total_output_tokens == 3

    tracker.record("m", input_tokens=5, output_tokens=7)
    total = tracker.get_usage_summary().model_usage_summaries["m"]
    assert total.total_calls == 2
    assert total.total_input_tokens == 7
    assert total.total_output_tokens == 10

    last2 = tracker.get_last_usage().model_usage_summaries["m"]
    assert last2.total_calls == 1
    assert last2.total_input_tokens == 5
    assert last2.total_output_tokens == 7


# =============================================================================
# Phase 2: Tool Calling Format Converter Tests
# =============================================================================


@pytest.mark.unit
def test_tool_definition_to_openai_format() -> None:
    """ToolDefinition should convert to OpenAI's function calling format."""
    tool = {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }

    result = tool_definition_to_openai_format(tool)

    assert result == {
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


@pytest.mark.unit
def test_tool_definition_to_anthropic_format() -> None:
    """ToolDefinition should convert to Anthropic's tool format."""
    tool = {
        "name": "search",
        "description": "Search the web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }

    result = tool_definition_to_anthropic_format(tool)

    assert result == {
        "name": "search",
        "description": "Search the web",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    }


@pytest.mark.unit
def test_tool_definition_to_gemini_format() -> None:
    """ToolDefinition should convert to Gemini's FunctionDeclaration format."""
    tool = {
        "name": "calculate",
        "description": "Do math",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}},
    }

    result = tool_definition_to_gemini_format(tool)

    assert result == {
        "name": "calculate",
        "description": "Do math",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}},
    }


@pytest.mark.unit
def test_tool_choice_to_openai_format() -> None:
    """tool_choice should convert to OpenAI's format."""
    assert tool_choice_to_openai_format(None) is None
    assert tool_choice_to_openai_format("auto") == "auto"
    assert tool_choice_to_openai_format("required") == "required"
    assert tool_choice_to_openai_format("none") == "none"

    # Specific tool name
    result = tool_choice_to_openai_format("get_weather")
    assert result == {"type": "function", "function": {"name": "get_weather"}}


@pytest.mark.unit
def test_tool_choice_to_anthropic_format() -> None:
    assert tool_choice_to_anthropic_format(None) is None
    assert tool_choice_to_anthropic_format("auto") == {"type": "auto"}
    assert tool_choice_to_anthropic_format("required") == {"type": "any"}
    assert tool_choice_to_anthropic_format("none") == {"type": "none"}

    result = tool_choice_to_anthropic_format("get_weather")
    assert result == {"type": "tool", "name": "get_weather"}


@pytest.mark.unit
def test_tool_choice_to_gemini_function_calling_config() -> None:
    assert tool_choice_to_gemini_function_calling_config(None) is None
    assert tool_choice_to_gemini_function_calling_config("auto") == {"mode": "AUTO"}
    assert tool_choice_to_gemini_function_calling_config("required") == {"mode": "ANY"}
    assert tool_choice_to_gemini_function_calling_config("none") == {"mode": "NONE"}

    result = tool_choice_to_gemini_function_calling_config("get_weather")
    assert result == {"mode": "ANY", "allowed_function_names": ["get_weather"]}


@pytest.mark.unit
def test_extract_tool_calls_openai_from_dict() -> None:
    """extract_tool_calls_openai should parse dict-style responses."""
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "NYC"}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            },
        ],
    }

    result = extract_tool_calls_openai(response)

    assert result is not None
    assert len(result) == 1
    assert result[0]["id"] == "call_123"
    assert result[0]["name"] == "get_weather"
    assert result[0]["arguments"] == {"city": "NYC"}


@pytest.mark.unit
def test_extract_tool_calls_openai_returns_none_when_no_tools() -> None:
    """extract_tool_calls_openai should return None when no tool calls."""
    response = {"choices": [{"message": {"content": "Hello!"}}]}
    assert extract_tool_calls_openai(response) is None

    response2 = {"choices": [{"message": {"tool_calls": []}}]}
    assert extract_tool_calls_openai(response2) is None


@pytest.mark.unit
def test_extract_tool_calls_anthropic_from_dict() -> None:
    """extract_tool_calls_anthropic should parse Anthropic's tool_use blocks."""
    response = {
        "content": [
            {"type": "text", "text": "Let me check the weather."},
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "get_weather",
                "input": {"city": "Boston"},
            },
        ],
        "stop_reason": "tool_use",
    }

    result = extract_tool_calls_anthropic(response)

    assert result is not None
    assert len(result) == 1
    assert result[0]["id"] == "toolu_abc"
    assert result[0]["name"] == "get_weather"
    assert result[0]["arguments"] == {"city": "Boston"}


@pytest.mark.unit
def test_extract_tool_calls_gemini_from_dict() -> None:
    """extract_tool_calls_gemini should parse Gemini's functionCall parts."""
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "function_call": {
                                "name": "get_weather",
                                "args": {"location": "Seattle"},
                            },
                        },
                    ],
                },
            },
        ],
    }

    result = extract_tool_calls_gemini(response)

    assert result is not None
    assert len(result) == 1
    assert result[0]["id"].startswith("gemini_call_")  # Generated ID
    assert result[0]["name"] == "get_weather"
    assert result[0]["arguments"] == {"location": "Seattle"}


@pytest.mark.unit
def test_extract_finish_reason_openai() -> None:
    """extract_finish_reason_openai should extract finish_reason from response."""
    response = {"choices": [{"finish_reason": "stop"}]}
    assert extract_finish_reason_openai(response) == "stop"

    response2 = {"choices": [{"finish_reason": "tool_calls"}]}
    assert extract_finish_reason_openai(response2) == "tool_calls"

    assert extract_finish_reason_openai({}) is None


@pytest.mark.unit
def test_extract_finish_reason_anthropic_normalizes() -> None:
    """extract_finish_reason_anthropic should normalize Anthropic's stop reasons."""
    assert extract_finish_reason_anthropic({"stop_reason": "end_turn"}) == "stop"
    assert extract_finish_reason_anthropic({"stop_reason": "tool_use"}) == "tool_calls"
    assert extract_finish_reason_anthropic({"stop_reason": "max_tokens"}) == "length"
    assert extract_finish_reason_anthropic({}) is None


@pytest.mark.unit
def test_extract_finish_reason_gemini_normalizes() -> None:
    """extract_finish_reason_gemini should normalize Gemini's finish reasons."""
    response = {"candidates": [{"finish_reason": "STOP"}]}
    assert extract_finish_reason_gemini(response) == "stop"

    response2 = {"candidates": [{"finish_reason": "MAX_TOKENS"}]}
    assert extract_finish_reason_gemini(response2) == "length"

    assert extract_finish_reason_gemini({}) is None
