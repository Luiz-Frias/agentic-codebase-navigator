"""E2E tests for RLM tool calling mode.

Tests the full stack using config-based instantiation:
create_rlm_from_config() → RLM facade → orchestrator → MockLLM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rlm.api import create_rlm_from_config
from rlm.application.config import EnvironmentConfig, LLMConfig, RLMConfig
from rlm.domain.agent_ports import ToolCallRequest


def _make_tool_call(tool_id: str, name: str, arguments: dict[str, Any]) -> ToolCallRequest:
    """Helper to create a ToolCallRequest."""
    return ToolCallRequest(id=tool_id, name=name, arguments=arguments)


# =============================================================================
# Sample Tools for Testing
# =============================================================================


def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b

    """
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b

    """
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        Result of a / b

    Raises:
        ValueError: If divisor is zero

    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def get_weather(city: str, unit: str = "celsius") -> dict[str, Any]:
    """Get weather for a city.

    Args:
        city: City name
        unit: Temperature unit (celsius or fahrenheit)

    Returns:
        Weather data dictionary

    """
    temps = {"london": 15, "tokyo": 22, "new_york": 18, "paris": 17}
    temp = temps.get(city.lower(), 20)
    if unit == "fahrenheit":
        temp = temp * 9 // 5 + 32
    return {"city": city, "temperature": temp, "unit": unit}


# =============================================================================
# E2E Tests: Tool Calling with Config
# =============================================================================


@pytest.mark.e2e
def test_e2e_tool_calling_calculator_single_tool() -> None:
    """E2E: Config-based RLM with single calculator tool call."""
    config = RLMConfig(
        llm=LLMConfig(
            backend="mock",
            model_name="mock-calc",
            backend_kwargs={
                "script": [
                    # LLM calls the add tool
                    {
                        "tool_calls": [_make_tool_call("call_1", "add", {"a": 5, "b": 3})],
                        "response": "",
                        "finish_reason": "tool_calls",
                    },
                    # LLM returns final answer
                    "The sum of 5 and 3 is 8.",
                ],
            },
        ),
        env=EnvironmentConfig(environment="local"),
        agent_mode="tools",
    )

    rlm = create_rlm_from_config(config, tools=[add, multiply, divide])
    result = rlm.completion("What is 5 + 3?")

    assert result.response == "The sum of 5 and 3 is 8."
    assert result.finish_reason == "stop"


@pytest.mark.e2e
def test_e2e_tool_calling_multi_step_calculation() -> None:
    """E2E: Config-based RLM with multi-step calculation (chain of tools)."""
    config = RLMConfig(
        llm=LLMConfig(
            backend="mock",
            model_name="mock-calc",
            backend_kwargs={
                "script": [
                    # First: multiply 4 * 5
                    {
                        "tool_calls": [_make_tool_call("call_1", "multiply", {"a": 4, "b": 5})],
                    },
                    # Second: add result (20) + 10
                    {
                        "tool_calls": [_make_tool_call("call_2", "add", {"a": 20, "b": 10})],
                    },
                    # Final answer
                    "4 times 5 is 20, plus 10 equals 30.",
                ],
            },
        ),
        env=EnvironmentConfig(environment="local"),
        agent_mode="tools",
    )

    rlm = create_rlm_from_config(config, tools=[add, multiply])
    result = rlm.completion("What is (4 * 5) + 10?")

    assert "30" in result.response


@pytest.mark.e2e
def test_e2e_tool_calling_weather_tool() -> None:
    """E2E: Config-based RLM with weather lookup tool."""
    config = RLMConfig(
        llm=LLMConfig(
            backend="mock",
            model_name="mock-weather",
            backend_kwargs={
                "script": [
                    {
                        "tool_calls": [
                            _make_tool_call(
                                "call_1",
                                "get_weather",
                                {"city": "Tokyo", "unit": "celsius"},
                            ),
                        ],
                    },
                    "The weather in Tokyo is 22°C.",
                ],
            },
        ),
        env=EnvironmentConfig(environment="local"),
        agent_mode="tools",
    )

    rlm = create_rlm_from_config(config, tools=[get_weather])
    result = rlm.completion("What's the weather in Tokyo?")

    assert "Tokyo" in result.response
    assert "22" in result.response


@pytest.mark.e2e
def test_e2e_tool_calling_immediate_answer_no_tools() -> None:
    """E2E: LLM can respond without using tools even in tool mode."""
    config = RLMConfig(
        llm=LLMConfig(
            backend="mock",
            model_name="mock-direct",
            backend_kwargs={
                "script": ["Paris is the capital of France. I didn't need any tools for this!"],
            },
        ),
        env=EnvironmentConfig(environment="local"),
        agent_mode="tools",
    )

    rlm = create_rlm_from_config(config, tools=[add, get_weather])
    result = rlm.completion("What is the capital of France?")

    assert "Paris" in result.response
    assert result.finish_reason == "stop"


@pytest.mark.e2e
def test_e2e_tool_calling_error_handling() -> None:
    """E2E: Tool execution errors are captured and passed to LLM."""
    config = RLMConfig(
        llm=LLMConfig(
            backend="mock",
            model_name="mock-error",
            backend_kwargs={
                "script": [
                    # LLM tries to divide by zero
                    {
                        "tool_calls": [_make_tool_call("call_1", "divide", {"a": 10, "b": 0})],
                    },
                    # LLM acknowledges the error
                    "I tried to divide 10 by 0, but that's not allowed. Division by zero is undefined.",
                ],
            },
        ),
        env=EnvironmentConfig(environment="local"),
        agent_mode="tools",
    )

    rlm = create_rlm_from_config(config, tools=[divide])
    result = rlm.completion("Divide 10 by 0")

    # Should NOT raise - error is passed to LLM
    assert "zero" in result.response.lower() or "divide" in result.response.lower()


@pytest.mark.e2e
async def test_e2e_tool_calling_async_path() -> None:
    """E2E: Async tool calling with config-based RLM."""
    config = RLMConfig(
        llm=LLMConfig(
            backend="mock",
            model_name="mock-async",
            backend_kwargs={
                "script": [
                    {
                        "tool_calls": [_make_tool_call("call_1", "multiply", {"a": 7, "b": 8})],
                    },
                    "7 multiplied by 8 equals 56.",
                ],
            },
        ),
        env=EnvironmentConfig(environment="local"),
        agent_mode="tools",
    )

    rlm = create_rlm_from_config(config, tools=[multiply])
    result = await rlm.acompletion("What is 7 times 8?")

    assert "56" in result.response
    assert result.finish_reason == "stop"


# =============================================================================
# E2E Tests: Backwards Compatibility
# =============================================================================


@pytest.mark.e2e
def test_e2e_code_mode_still_works_with_config() -> None:
    """E2E: Default code mode is unaffected by tool calling infrastructure."""
    config = RLMConfig(
        llm=LLMConfig(
            backend="mock",
            model_name="mock-code",
            backend_kwargs={"script": ["FINAL(Code execution mode works!)"]},
        ),
        env=EnvironmentConfig(environment="local"),
        # agent_mode defaults to "code"
    )

    rlm = create_rlm_from_config(config)
    result = rlm.completion("Test code mode")

    assert result.response == "Code execution mode works!"


@pytest.mark.e2e
def test_e2e_explicit_code_mode_in_config() -> None:
    """E2E: Explicit agent_mode='code' in config works correctly."""
    config = RLMConfig(
        llm=LLMConfig(
            backend="mock",
            model_name="mock-code",
            backend_kwargs={"script": ["FINAL(Explicit code mode)"]},
        ),
        env=EnvironmentConfig(environment="local"),
        agent_mode="code",  # Explicitly set
    )

    rlm = create_rlm_from_config(config)
    result = rlm.completion("Test explicit code mode")

    assert result.response == "Explicit code mode"


# =============================================================================
# E2E Tests: Structured Output with Tools
# =============================================================================


@dataclass
class CalculationResult:
    """Structured result from a calculation."""

    operation: str
    operands: list[float]
    result: float


@pytest.mark.e2e
def test_e2e_tool_calling_with_structured_response() -> None:
    """E2E: Tool calling followed by structured JSON response."""
    config = RLMConfig(
        llm=LLMConfig(
            backend="mock",
            model_name="mock-structured",
            backend_kwargs={
                "script": [
                    # LLM calls add tool
                    {
                        "tool_calls": [_make_tool_call("call_1", "add", {"a": 15, "b": 25})],
                    },
                    # LLM returns structured JSON response
                    '{"operation": "add", "operands": [15, 25], "result": 40}',
                ],
            },
        ),
        env=EnvironmentConfig(environment="local"),
        agent_mode="tools",
    )

    rlm = create_rlm_from_config(config, tools=[add])
    result = rlm.completion("Add 15 and 25, return as JSON")

    # Verify the JSON response can be parsed
    import json

    data = json.loads(result.response)
    assert data["operation"] == "add"
    assert data["result"] == 40


# =============================================================================
# E2E Tests: Mixed Mode (Sequential Sessions)
# =============================================================================


@pytest.mark.e2e
def test_e2e_can_create_separate_code_and_tool_instances() -> None:
    """E2E: Can create separate RLM instances for code and tool modes."""
    # Code mode instance
    code_config = RLMConfig(
        llm=LLMConfig(
            backend="mock",
            model_name="mock-code",
            backend_kwargs={"script": ["FINAL(Code result)"]},
        ),
        env=EnvironmentConfig(environment="local"),
        agent_mode="code",
    )

    # Tool mode instance
    tool_config = RLMConfig(
        llm=LLMConfig(
            backend="mock",
            model_name="mock-tool",
            backend_kwargs={
                "script": [
                    {"tool_calls": [_make_tool_call("call_1", "add", {"a": 1, "b": 2})]},
                    "Tool result: 3",
                ],
            },
        ),
        env=EnvironmentConfig(environment="local"),
        agent_mode="tools",
    )

    code_rlm = create_rlm_from_config(code_config)
    tool_rlm = create_rlm_from_config(tool_config, tools=[add])

    code_result = code_rlm.completion("Code task")
    tool_result = tool_rlm.completion("Tool task")

    assert code_result.response == "Code result"
    assert "3" in tool_result.response
