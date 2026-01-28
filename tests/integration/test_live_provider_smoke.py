from __future__ import annotations

import importlib.util
import os
import time
from urllib.parse import urlparse

import pytest

from rlm.adapters.llm.retry import RetryConfig, compute_retry_delay
from rlm.domain.models import LLMRequest
from rlm.domain.errors import LLMError
from rlm.domain.ports import LLMPort
from rlm.domain.models import ChatCompletion
from tests.live_llm import load_env_files

load_env_files()

RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay_seconds=5.0,
    max_delay_seconds=60.0,
    jitter_seconds=3.14,
)


@pytest.mark.integration
@pytest.mark.live_llm
def test_live_openai_adapter_smoke() -> None:
    """
    Real-network smoke test for OpenAIAdapter.

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
    api_version = os.environ.get("OPENAI_API_VERSION")

    if _is_azure_endpoint(base_url):
        if api_version is None:
            pytest.skip("OPENAI_API_VERSION not set for Azure OpenAI endpoint")
        from rlm.adapters.llm.azure_openai import AzureOpenAIAdapter

        llm = AzureOpenAIAdapter(
            deployment=model,
            api_key=api_key,
            endpoint=base_url,
            api_version=api_version,
            retry_config=RETRY_CONFIG,
            default_request_kwargs={
                "temperature": 1,
                "max_tokens": 16,
            },
        )
    else:
        from rlm.adapters.llm.openai import OpenAIAdapter

        llm = OpenAIAdapter(
            model=model,
            api_key=api_key,
            base_url=base_url,
            retry_config=RETRY_CONFIG,
            default_request_kwargs={
                # Keep spend and latency low; be deterministic-ish.
                "temperature": 1,
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
    """
    Real-network test for OpenAI tool calling.

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
    api_version = os.environ.get("OPENAI_API_VERSION")
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

    if _is_azure_endpoint(base_url):
        if api_version is None:
            pytest.skip("OPENAI_API_VERSION not set for Azure OpenAI endpoint")
        from rlm.adapters.llm.azure_openai import AzureOpenAIAdapter

        llm = AzureOpenAIAdapter(
            deployment=model,
            api_key=api_key,
            endpoint=base_url,
            api_version=api_version,
            retry_config=RETRY_CONFIG,
            default_request_kwargs={
                "temperature": 1,
                "max_tokens": 100,
            },
        )
    else:
        from rlm.adapters.llm.openai import OpenAIAdapter

        llm = OpenAIAdapter(
            model=model,
            api_key=api_key,
            base_url=base_url,
            retry_config=RETRY_CONFIG,
            default_request_kwargs={
                "temperature": 1,
                "max_tokens": 100,
            },
        )

    # Request should trigger tool call
    cc = llm.complete(
        LLMRequest(
            prompt="What's the weather in Tokyo?",
            tools=tools,
            tool_choice="auto",
        )
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


def _is_azure_endpoint(base_url: str | None) -> bool:
    if not base_url:
        return False
    parsed = urlparse(base_url)
    host = parsed.netloc.lower() if parsed.netloc else base_url.lower()
    return "azure.com" in host or "cognitiveservices.azure.com" in host


@pytest.mark.integration
@pytest.mark.live_llm
def test_live_anthropic_adapter_smoke() -> None:
    """
    Real-network smoke test for AnthropicAdapter.

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
    cc = _complete_with_retry(llm, LLMRequest(prompt="Return exactly the word ok."), RETRY_CONFIG)

    assert cc.root_model == model
    assert isinstance(cc.response, str) and cc.response.strip()
    assert "ok" in cc.response.strip().lower()

    mus = cc.usage_summary.model_usage_summaries.get(model)
    assert mus is not None
    assert mus.total_calls == 1
    assert mus.total_input_tokens >= 0
    assert mus.total_output_tokens >= 0


def _complete_with_retry(
    llm: LLMPort,
    request: LLMRequest,
    retry_config: RetryConfig,
) -> ChatCompletion:
    for attempt in range(1, retry_config.max_attempts + 1):
        try:
            return llm.complete(request)
        except LLMError as exc:
            message = str(exc).lower()
            if "rate limit" not in message or attempt >= retry_config.max_attempts:
                raise
            delay = compute_retry_delay(retry_config, attempt)
            time.sleep(delay)
    raise RuntimeError("Retry loop exhausted without raising")
