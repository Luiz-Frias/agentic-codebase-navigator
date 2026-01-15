from __future__ import annotations

import pytest

from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.domain.errors import LLMError
from rlm.domain.models import LLMRequest


@pytest.mark.unit
def test_mock_llm_adapter_echo_is_deterministic_and_tracks_usage() -> None:
    llm = MockLLMAdapter(model="mock-model")

    cc1 = llm.complete(LLMRequest(prompt="hello world"))
    assert cc1.root_model == "mock-model"
    assert cc1.response.startswith("Mock response to: ")
    assert "hello world" in cc1.response

    last1 = llm.get_last_usage().model_usage_summaries["mock-model"]
    assert last1.total_calls == 1

    total1 = llm.get_usage_summary().model_usage_summaries["mock-model"]
    assert total1.total_calls == 1

    cc2 = llm.complete(LLMRequest(prompt="hello world"))
    assert cc2.response == cc1.response

    last2 = llm.get_last_usage().model_usage_summaries["mock-model"]
    assert last2.total_calls == 1

    total2 = llm.get_usage_summary().model_usage_summaries["mock-model"]
    assert total2.total_calls == 2


@pytest.mark.unit
def test_mock_llm_adapter_usage_summary_is_snapshot_not_mutated_by_later_calls() -> None:
    llm = MockLLMAdapter(model="mock-model")

    llm.complete(LLMRequest(prompt="hello world"))
    snap1 = llm.get_usage_summary()
    assert snap1.model_usage_summaries["mock-model"].total_calls == 1

    llm.complete(LLMRequest(prompt="hello world"))
    # Earlier snapshot must not change after later calls.
    assert snap1.model_usage_summaries["mock-model"].total_calls == 1

    snap2 = llm.get_usage_summary()
    assert snap2.model_usage_summaries["mock-model"].total_calls == 2


@pytest.mark.unit
def test_mock_llm_adapter_usage_summary_does_not_allow_mutating_internal_state() -> None:
    llm = MockLLMAdapter(model="mock-model")

    llm.complete(LLMRequest(prompt="hello world"))
    snap = llm.get_usage_summary()
    snap.model_usage_summaries["mock-model"].total_calls = 999

    snap2 = llm.get_usage_summary()
    assert snap2.model_usage_summaries["mock-model"].total_calls == 1


@pytest.mark.unit
def test_mock_llm_adapter_scripted_responses_pop_in_order() -> None:
    llm = MockLLMAdapter(model="dummy", script=["A", "B"])

    assert llm.complete(LLMRequest(prompt="p")).response == "A"
    assert llm.complete(LLMRequest(prompt="p")).response == "B"

    with pytest.raises(LLMError, match="no scripted responses left"):
        llm.complete(LLMRequest(prompt="p"))


@pytest.mark.unit
def test_mock_llm_adapter_script_can_raise_exceptions() -> None:
    llm = MockLLMAdapter(model="dummy", script=[ValueError("boom")])
    with pytest.raises(ValueError, match="boom"):
        llm.complete(LLMRequest(prompt="p"))


# =============================================================================
# Phase 2: Tool Calling Tests
# =============================================================================


@pytest.mark.unit
def test_mock_llm_adapter_supports_tools_property() -> None:
    """MockLLMAdapter should report supports_tools=True."""
    llm = MockLLMAdapter(model="mock-model")
    assert llm.supports_tools is True


@pytest.mark.unit
def test_mock_llm_adapter_scripted_tool_calls() -> None:
    """MockLLMAdapter should return tool_calls from scripted dict responses."""
    from rlm.domain.agent_ports import ToolCallRequest

    tool_calls: list[ToolCallRequest] = [
        {"id": "call_123", "name": "get_weather", "arguments": {"city": "NYC"}},
    ]
    llm = MockLLMAdapter(
        model="mock-model",
        script=[
            {"tool_calls": tool_calls},
            "The weather is sunny",  # Final answer
        ],
    )

    # First call should return tool calls
    cc1 = llm.complete(LLMRequest(prompt="What's the weather?"))
    assert cc1.tool_calls is not None
    assert len(cc1.tool_calls) == 1
    assert cc1.tool_calls[0]["name"] == "get_weather"
    assert cc1.tool_calls[0]["arguments"] == {"city": "NYC"}
    assert cc1.finish_reason == "tool_calls"
    assert cc1.response == ""  # Empty when tool_calls present

    # Second call should return text
    cc2 = llm.complete(LLMRequest(prompt="[tool result: sunny]"))
    assert cc2.tool_calls is None
    assert cc2.response == "The weather is sunny"
    assert cc2.finish_reason == "stop"


@pytest.mark.unit
def test_mock_llm_adapter_scripted_tool_calls_with_text() -> None:
    """MockLLMAdapter should support both tool_calls and response text together."""
    from rlm.domain.agent_ports import ToolCallRequest

    tool_calls: list[ToolCallRequest] = [
        {"id": "call_abc", "name": "search", "arguments": {"query": "python"}},
    ]
    llm = MockLLMAdapter(
        model="mock-model",
        script=[
            {"tool_calls": tool_calls, "response": "Let me search for that."},
        ],
    )

    cc = llm.complete(LLMRequest(prompt="Find Python docs"))
    assert cc.tool_calls is not None
    assert len(cc.tool_calls) == 1
    assert cc.response == "Let me search for that."
    assert cc.finish_reason == "tool_calls"


@pytest.mark.unit
def test_mock_llm_adapter_scripted_custom_finish_reason() -> None:
    """MockLLMAdapter should support custom finish_reason in scripted responses."""
    llm = MockLLMAdapter(
        model="mock-model",
        script=[
            {"response": "Truncated response", "finish_reason": "length"},
        ],
    )

    cc = llm.complete(LLMRequest(prompt="Write an essay"))
    assert cc.tool_calls is None
    assert cc.response == "Truncated response"
    assert cc.finish_reason == "length"


@pytest.mark.unit
def test_mock_llm_adapter_multiple_tool_calls() -> None:
    """MockLLMAdapter should support multiple tool calls in a single response."""
    from rlm.domain.agent_ports import ToolCallRequest

    tool_calls: list[ToolCallRequest] = [
        {"id": "call_1", "name": "get_weather", "arguments": {"city": "NYC"}},
        {"id": "call_2", "name": "get_time", "arguments": {"timezone": "EST"}},
    ]
    llm = MockLLMAdapter(
        model="mock-model",
        script=[{"tool_calls": tool_calls}],
    )

    cc = llm.complete(LLMRequest(prompt="Weather and time?"))
    assert cc.tool_calls is not None
    assert len(cc.tool_calls) == 2
    assert cc.tool_calls[0]["name"] == "get_weather"
    assert cc.tool_calls[1]["name"] == "get_time"


@pytest.mark.unit
def test_mock_llm_adapter_echo_mode_includes_finish_reason() -> None:
    """MockLLMAdapter in echo mode should set finish_reason to 'stop'."""
    llm = MockLLMAdapter(model="mock-model")
    cc = llm.complete(LLMRequest(prompt="hello"))
    assert cc.finish_reason == "stop"
    assert cc.tool_calls is None
