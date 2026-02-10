"""
Integration tests for RLM tool calling mode.

Tests the full stack: RLM facade → run_completion use case → orchestrator → MockLLM
"""

from __future__ import annotations

from typing import Any

import pytest

from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.api.rlm import RLM
from rlm.domain.agent_ports import ToolCallRequest
from tests.live_llm import LiveLLMSettings


def _make_tool_call(tool_id: str, name: str, arguments: dict[str, Any]) -> ToolCallRequest:
    """Helper to create a ToolCallRequest."""
    return ToolCallRequest(id=tool_id, name=name, arguments=arguments)


def get_weather(city: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a city.

    Args:
        city: The city name
        unit: Temperature unit (celsius or fahrenheit)

    """
    # Mock implementation
    temps = {"london": 15, "tokyo": 22, "new_york": 18}
    temp = temps.get(city.lower(), 20)
    if unit == "fahrenheit":
        temp = temp * 9 // 5 + 32
    return f"Weather in {city}: {temp}°{unit[0].upper()}"


def calculate(operation: str, a: float, b: float) -> float:
    """
    Perform a mathematical operation.

    Args:
        operation: The operation (add, subtract, multiply, divide)
        a: First operand
        b: Second operand

    """
    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float("inf"),
    }
    return ops.get(operation, lambda x, y: 0)(a, b)


def _build_live_rlm(settings: LiveLLMSettings, tools: list[object]) -> RLM:
    llm = settings.build_openai_adapter(
        request_kwargs={
            "temperature": 0,
            "max_tokens": 128,
        }
    )
    return RLM(
        llm=llm,
        tools=tools,
        agent_mode="tools",
    )


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


@pytest.mark.integration
def test_rlm_tool_calling_single_tool_happy_path(
    live_llm_settings: LiveLLMSettings | None,
) -> None:
    """Full stack test: LLM calls tool → executes → returns final answer."""
    if live_llm_settings is not None:
        rlm = _build_live_rlm(live_llm_settings, tools=[get_weather])
        result = rlm.completion("What's the weather in Tokyo?", tool_choice="required")
        assert result.response.strip()
        return

    # MockLLMAdapter with scripted tool_call response
    llm = MockLLMAdapter(
        model="mock-tool-test",
        script=[
            # First call: LLM requests tool call
            {
                "tool_calls": [
                    _make_tool_call("call_1", "get_weather", {"city": "Tokyo", "unit": "celsius"}),
                ],
                "response": "",
                "finish_reason": "tool_calls",
            },
            # Second call: LLM returns final answer after seeing tool result
            "The weather in Tokyo is 22°C.",
        ],
    )

    rlm = RLM(
        llm=llm,
        tools=[get_weather],
        agent_mode="tools",
    )

    result = rlm.completion("What's the weather in Tokyo?")

    assert result.response == "The weather in Tokyo is 22°C."
    assert result.finish_reason == "stop"


@pytest.mark.integration
def test_rlm_tool_calling_multi_turn_conversation(
    live_llm_settings: LiveLLMSettings | None,
) -> None:
    """Full stack test: LLM makes multiple tool calls across turns."""
    if live_llm_settings is not None:
        rlm = _build_live_rlm(live_llm_settings, tools=[get_weather])
        result = rlm.completion(
            "Compare the weather in London and Tokyo. Use tools if needed.",
            tool_choice="required",
        )
        assert result.response.strip()
        return

    llm = MockLLMAdapter(
        model="mock-tool-test",
        script=[
            # Turn 1: Get weather for first city
            {
                "tool_calls": [_make_tool_call("call_1", "get_weather", {"city": "London"})],
            },
            # Turn 2: Get weather for second city
            {
                "tool_calls": [_make_tool_call("call_2", "get_weather", {"city": "Tokyo"})],
            },
            # Turn 3: Final answer
            "London is 15°C and Tokyo is 22°C. Tokyo is warmer.",
        ],
    )

    rlm = RLM(
        llm=llm,
        tools=[get_weather],
        agent_mode="tools",
    )

    result = rlm.completion("Compare the weather in London and Tokyo")

    assert "London" in result.response
    assert "Tokyo" in result.response


@pytest.mark.integration
def test_rlm_tool_calling_multiple_tools(
    live_llm_settings: LiveLLMSettings | None,
) -> None:
    """Full stack test: LLM can use multiple different tools."""
    if live_llm_settings is not None:
        rlm = _build_live_rlm(live_llm_settings, tools=[get_weather, calculate])
        result = rlm.completion("What is 5 times 3?", tool_choice="required")
        assert result.response.strip()
        assert _has_digit(result.response)
        return

    llm = MockLLMAdapter(
        model="mock-tool-test",
        script=[
            # Call calculator
            {
                "tool_calls": [
                    _make_tool_call(
                        "call_1",
                        "calculate",
                        {"operation": "multiply", "a": 5, "b": 3},
                    ),
                ],
            },
            # Final answer
            "5 multiplied by 3 equals 15.",
        ],
    )

    rlm = RLM(
        llm=llm,
        tools=[get_weather, calculate],
        agent_mode="tools",
    )

    result = rlm.completion("What is 5 times 3?")

    assert "15" in result.response


@pytest.mark.integration
def test_rlm_tool_calling_immediate_answer(
    live_llm_settings: LiveLLMSettings | None,
) -> None:
    """Full stack test: LLM can respond without using tools."""
    if live_llm_settings is not None:
        rlm = _build_live_rlm(live_llm_settings, tools=[get_weather, calculate])
        result = rlm.completion("What is the capital of France?")
        assert result.response.strip()
        assert "paris" in result.response.lower()
        return

    llm = MockLLMAdapter(
        model="mock-tool-test",
        script=["I don't need any tools to answer this: Paris is the capital of France."],
    )

    rlm = RLM(
        llm=llm,
        tools=[get_weather, calculate],
        agent_mode="tools",
    )

    result = rlm.completion("What is the capital of France?")

    assert "Paris" in result.response
    assert result.finish_reason == "stop"


@pytest.mark.integration
def test_rlm_tool_calling_tool_error_captured(
    live_llm_settings: LiveLLMSettings | None,
) -> None:
    """Full stack test: Tool execution errors are passed to LLM, not raised."""

    def always_fails() -> None:
        """A tool that always fails."""
        raise RuntimeError("This tool is broken!")

    if live_llm_settings is not None:
        rlm = _build_live_rlm(live_llm_settings, tools=[always_fails])
        result = rlm.completion("Call the tool even if it fails.", tool_choice="required")
        assert result.response.strip()
        return

    llm = MockLLMAdapter(
        model="mock-tool-test",
        script=[
            # LLM calls the failing tool
            {
                "tool_calls": [_make_tool_call("call_1", "always_fails", {})],
            },
            # LLM responds to the error
            "I encountered an error while trying to help. Let me try differently.",
        ],
    )

    rlm = RLM(
        llm=llm,
        tools=[always_fails],
        agent_mode="tools",
    )

    # Should NOT raise - error is captured and passed to LLM
    result = rlm.completion("Do the thing")

    assert "error" in result.response.lower() or "try" in result.response.lower()


@pytest.mark.integration
async def test_rlm_tool_calling_async_path(
    live_llm_settings: LiveLLMSettings | None,
) -> None:
    """Full stack test: Async completion with tool calling."""
    if live_llm_settings is not None:
        rlm = _build_live_rlm(live_llm_settings, tools=[calculate])
        result = await rlm.acompletion("What is 10 plus 5?", tool_choice="required")
        assert result.response.strip()
        assert _has_digit(result.response)
        return

    llm = MockLLMAdapter(
        model="mock-tool-test",
        script=[
            {
                "tool_calls": [
                    _make_tool_call("call_1", "calculate", {"operation": "add", "a": 10, "b": 5}),
                ],
            },
            "10 + 5 = 15",
        ],
    )

    rlm = RLM(
        llm=llm,
        tools=[calculate],
        agent_mode="tools",
    )

    result = await rlm.acompletion("What is 10 plus 5?")

    assert result.response == "10 + 5 = 15"
    assert result.finish_reason == "stop"


@pytest.mark.integration
def test_rlm_tool_calling_code_mode_still_works() -> None:
    """Full stack test: Default code mode is unaffected by tool infra."""
    llm = MockLLMAdapter(
        model="mock-test",
        script=["FINAL(Code mode works!)"],
    )

    # No tools, default agent_mode="code"
    rlm = RLM(llm=llm)

    result = rlm.completion("Test prompt")

    assert result.response == "Code mode works!"
