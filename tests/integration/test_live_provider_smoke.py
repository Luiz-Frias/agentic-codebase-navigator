from __future__ import annotations

import importlib.util
import os

import pytest

from rlm.domain.models import LLMRequest


@pytest.mark.integration
@pytest.mark.live_llm
def test_live_openai_adapter_smoke() -> None:
    """Real-network smoke test for OpenAIAdapter.

    Skipped by default; enable with:
      - RLM_RUN_LIVE_LLM_TESTS=1
      - OPENAI_API_KEY=...

    Optional:
      - OPENAI_BASE_URL=... (for OpenAI-compatible endpoints)
      - OPENAI_MODEL=... (defaults to gpt-5-nano)
    """
    if importlib.util.find_spec("openai") is None:
        pytest.skip("openai package not installed")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_MODEL") or "gpt-5-nano"
    base_url = os.environ.get("OPENAI_BASE_URL")

    from rlm.adapters.llm.openai import OpenAIAdapter

    llm = OpenAIAdapter(
        model=model,
        api_key=api_key,
        base_url=base_url,
        default_request_kwargs={
            # Keep spend and latency low; be deterministic-ish.
            "temperature": 0,
            "max_tokens": 16,
        },
    )
    cc = llm.complete(LLMRequest(prompt="Return exactly the word ok."))

    assert cc.root_model == model
    assert isinstance(cc.response, str) and cc.response.strip()
    assert "ok" in cc.response.strip().lower()

    mus = cc.usage_summary.model_usage_summaries.get(model)
    assert mus is not None
    assert mus.total_calls == 1
    assert mus.total_input_tokens >= 0
    assert mus.total_output_tokens >= 0


@pytest.mark.integration
@pytest.mark.live_llm
def test_live_openai_adapter_tool_calling() -> None:
    """Real-network test for OpenAI tool calling.

    Skipped by default; enable with:
      - RLM_RUN_LIVE_LLM_TESTS=1
      - OPENAI_API_KEY=...

    Optional:
      - OPENAI_BASE_URL=... (for OpenAI-compatible endpoints)
      - OPENAI_MODEL=... (defaults to gpt-4o-mini)
    """
    if importlib.util.find_spec("openai") is None:
        pytest.skip("openai package not installed")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
    base_url = os.environ.get("OPENAI_BASE_URL")

    from rlm.adapters.llm.openai import OpenAIAdapter
    from rlm.domain.agent_ports import ToolDefinition

    # Define a simple tool for the LLM to call
    tools: list[ToolDefinition] = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city, e.g. San Francisco",
                    },
                },
                "required": ["location"],
            },
        },
    ]

    llm = OpenAIAdapter(
        model=model,
        api_key=api_key,
        base_url=base_url,
        default_request_kwargs={
            "temperature": 0,
            "max_tokens": 100,
        },
    )

    # Request should trigger tool call
    cc = llm.complete(
        LLMRequest(
            prompt="What's the weather in Tokyo?",
            tools=tools,
            tool_choice="auto",
        ),
    )

    assert cc.root_model == model
    assert cc.finish_reason in ("stop", "tool_calls")

    # Check if LLM decided to call the tool (it should with this prompt)
    if cc.tool_calls:
        assert len(cc.tool_calls) >= 1
        assert cc.tool_calls[0]["name"] == "get_current_weather"
        assert "location" in cc.tool_calls[0]["arguments"]
        assert cc.finish_reason == "tool_calls"
    else:
        # Some models may respond directly without calling tools
        assert cc.response.strip() != ""
        assert cc.finish_reason == "stop"

    # Verify usage tracking works
    mus = cc.usage_summary.model_usage_summaries.get(model)
    assert mus is not None
    assert mus.total_calls == 1
    assert mus.total_input_tokens >= 0
    assert mus.total_output_tokens >= 0


@pytest.mark.integration
@pytest.mark.live_llm
def test_live_anthropic_adapter_smoke() -> None:
    """Real-network smoke test for AnthropicAdapter.

    Skipped by default; enable with:
      - RLM_RUN_LIVE_LLM_TESTS=1
      - ANTHROPIC_API_KEY=...

    Optional:
      - ANTHROPIC_MODEL=... (defaults to claude-3-5-haiku-20241022)
    """
    if importlib.util.find_spec("anthropic") is None:
        pytest.skip("anthropic package not installed (install the llm-anthropic extra)")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    model = os.environ.get("ANTHROPIC_MODEL") or "claude-3-5-haiku-20241022"

    from rlm.adapters.llm.anthropic import AnthropicAdapter

    llm = AnthropicAdapter(
        model=model,
        api_key=api_key,
        default_request_kwargs={
            # Keep spend and latency low.
            "max_tokens": 16,
        },
    )
    cc = llm.complete(LLMRequest(prompt="Return exactly the word ok."))

    assert cc.root_model == model
    assert isinstance(cc.response, str) and cc.response.strip()
    assert "ok" in cc.response.strip().lower()

    mus = cc.usage_summary.model_usage_summaries.get(model)
    assert mus is not None
    assert mus.total_calls == 1
    assert mus.total_input_tokens >= 0
    assert mus.total_output_tokens >= 0
