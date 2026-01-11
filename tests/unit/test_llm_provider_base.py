from __future__ import annotations

import pytest

from rlm.adapters.llm.provider_base import (
    UsageTracker,
    extract_openai_style_token_usage,
    extract_text_from_chat_response,
    prompt_to_messages,
    prompt_to_text,
    safe_provider_error_message,
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
        {"usage": {"prompt_tokens": 1, "completion_tokens": 2}}
    ) == (
        1,
        2,
    )
    assert extract_openai_style_token_usage(
        {"usage": {"input_tokens": "3", "output_tokens": "4"}}
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
