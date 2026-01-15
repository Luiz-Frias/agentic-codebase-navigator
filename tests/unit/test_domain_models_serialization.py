from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from rlm.domain.models import (
    ChatCompletion,
    CodeBlock,
    Iteration,
    ModelUsageSummary,
    ReplResult,
    UsageSummary,
)
from rlm.domain.result import Err, Ok, Result


@pytest.mark.unit
def test_domain_models_roundtrip_via_to_from_dict_is_stable() -> None:
    usage = UsageSummary(model_usage_summaries={"m": ModelUsageSummary(1, 2, 3)})
    cc = ChatCompletion(
        root_model="m",
        prompt={"role": "user", "content": "hi"},
        response="ok",
        usage_summary=usage,
        execution_time=0.123,
    )
    repl = ReplResult(
        stdout="out",
        stderr="err",
        locals={
            "a": 1,
            "b": [1, 2, 3],
            "c": {"x": "y"},
            "callable": math.sin,
            "module_like": json,  # module should serialize deterministically
            "unknown": SimpleNamespace(x=1),
        },
        llm_calls=[cc],
        execution_time=0.456,
    )
    it = Iteration(
        prompt="p",
        response="r",
        code_blocks=[CodeBlock(code="print(1)", result=repl)],
        final_answer="fa",
        iteration_time=0.789,
    )

    # Contract: dict → from_dict(dict) → to_dict() is stable for JSON logging.
    d = it.to_dict()
    it2 = Iteration.from_dict(d)
    assert it2.to_dict() == d


@pytest.mark.unit
def test_domain_model_usage_summary_from_dict_treats_none_as_zero() -> None:
    s = ModelUsageSummary.from_dict(
        {"total_calls": None, "total_input_tokens": None, "total_output_tokens": None}
    )
    assert (s.total_calls, s.total_input_tokens, s.total_output_tokens) == (0, 0, 0)


@pytest.mark.unit
def test_domain_repl_result_from_dict_accepts_legacy_rlm_calls_key() -> None:
    legacy_like = {
        "stdout": "out",
        "stderr": "",
        "locals": {},
        "execution_time": 0.2,
        # Legacy REPLResult.to_dict() uses "rlm_calls" for schema compatibility.
        "rlm_calls": [
            {
                "root_model": "dummy",
                "prompt": "p",
                "response": "FINAL(pong)",
                "usage_summary": {"model_usage_summaries": {"dummy": {"total_calls": 1}}},
                "execution_time": 0.01,
            }
        ],
    }

    repl = ReplResult.from_dict(legacy_like)
    assert [c.response for c in repl.llm_calls] == ["FINAL(pong)"]


@pytest.mark.unit
def test_domain_result_type_pattern_matching() -> None:
    def _f(flag: bool) -> Result[int]:
        if flag:
            return Ok(123)
        return Err("nope")

    match _f(True):
        case Ok(value=v):
            assert v == 123
        case _:
            raise AssertionError("expected Ok")


# =============================================================================
# Phase 2: Tool Calling Model Tests
# =============================================================================


@pytest.mark.unit
def test_chat_completion_with_tool_calls_roundtrip() -> None:
    """ChatCompletion with tool_calls should serialize and deserialize correctly."""
    from rlm.domain.agent_ports import ToolCallRequest

    usage = UsageSummary(model_usage_summaries={"gpt-4": ModelUsageSummary(1, 100, 50)})
    tool_calls: list[ToolCallRequest] = [
        {"id": "call_abc123", "name": "get_weather", "arguments": {"city": "NYC"}},
        {"id": "call_def456", "name": "get_time", "arguments": {"timezone": "EST"}},
    ]
    cc = ChatCompletion(
        root_model="gpt-4",
        prompt="What's the weather in NYC?",
        response="",  # Empty when tool_calls present
        usage_summary=usage,
        execution_time=0.5,
        tool_calls=tool_calls,
        finish_reason="tool_calls",
    )

    # Serialize
    d = cc.to_dict()
    assert d["tool_calls"] == tool_calls
    assert d["finish_reason"] == "tool_calls"

    # Deserialize
    cc2 = ChatCompletion.from_dict(d)
    assert cc2.tool_calls == tool_calls
    assert cc2.finish_reason == "tool_calls"

    # Roundtrip stability
    assert cc2.to_dict() == d


@pytest.mark.unit
def test_chat_completion_without_tool_calls_backward_compatible() -> None:
    """ChatCompletion without tool_calls should remain backward compatible."""
    usage = UsageSummary(model_usage_summaries={"gpt-4": ModelUsageSummary(1, 50, 25)})
    cc = ChatCompletion(
        root_model="gpt-4",
        prompt="Hello",
        response="Hi there!",
        usage_summary=usage,
        execution_time=0.3,
        # tool_calls and finish_reason default to None
    )

    d = cc.to_dict()
    # Should NOT include tool_calls or finish_reason when None (backward compat)
    assert "tool_calls" not in d
    assert "finish_reason" not in d

    # Deserialize legacy dict without tool fields
    legacy_dict = {
        "root_model": "gpt-4",
        "prompt": "Hello",
        "response": "Hi!",
        "usage_summary": {"model_usage_summaries": {}},
        "execution_time": 0.2,
    }
    cc2 = ChatCompletion.from_dict(legacy_dict)
    assert cc2.tool_calls is None
    assert cc2.finish_reason is None


@pytest.mark.unit
def test_chat_completion_finish_reason_stop() -> None:
    """ChatCompletion with finish_reason='stop' and no tool_calls."""
    usage = UsageSummary(model_usage_summaries={"gpt-4": ModelUsageSummary(1, 10, 20)})
    cc = ChatCompletion(
        root_model="gpt-4",
        prompt="2+2?",
        response="4",
        usage_summary=usage,
        execution_time=0.1,
        tool_calls=None,
        finish_reason="stop",
    )

    d = cc.to_dict()
    assert "tool_calls" not in d  # None values not serialized
    assert d["finish_reason"] == "stop"

    cc2 = ChatCompletion.from_dict(d)
    assert cc2.tool_calls is None
    assert cc2.finish_reason == "stop"


@pytest.mark.unit
def test_llm_request_with_tools_and_tool_choice() -> None:
    """LLMRequest should accept tools and tool_choice parameters."""
    from rlm.domain.agent_ports import ToolDefinition
    from rlm.domain.models import LLMRequest

    tool_defs: list[ToolDefinition] = [
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

    # Test with tools and auto tool_choice
    req = LLMRequest(
        prompt="What's the weather?",
        model="gpt-4",
        tools=tool_defs,
        tool_choice="auto",
    )
    assert req.tools == tool_defs
    assert req.tool_choice == "auto"

    # Test with required tool_choice
    req2 = LLMRequest(
        prompt="Call the weather tool",
        tools=tool_defs,
        tool_choice="required",
    )
    assert req2.tool_choice == "required"

    # Test with specific tool name as tool_choice
    req3 = LLMRequest(
        prompt="Use get_weather",
        tools=tool_defs,
        tool_choice="get_weather",
    )
    assert req3.tool_choice == "get_weather"


@pytest.mark.unit
def test_llm_request_defaults_no_tools() -> None:
    """LLMRequest without tools should have None defaults (backward compat)."""
    from rlm.domain.models import LLMRequest

    req = LLMRequest(prompt="Hello")
    assert req.tools is None
    assert req.tool_choice is None
    assert req.model is None
