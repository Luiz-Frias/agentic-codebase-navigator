"""
Live LLM performance benchmarks.

These tests hit real provider APIs to measure actual latency and throughput.
Skipped by default to avoid API costs.

Enable with:
  RLM_RUN_LIVE_LLM_TESTS=1 pytest -m "performance and live_llm" tests/performance/

Required environment variables:
  - OPENAI_API_KEY: For OpenAI tests
  - ANTHROPIC_API_KEY: For Anthropic tests

Optional:
  - OPENAI_MODEL: Model to use (default: gpt-4o-mini)
  - ANTHROPIC_MODEL: Model to use (default: claude-3-5-haiku-20241022)
"""

from __future__ import annotations

import importlib.util
import os
import time

import pytest

from rlm.domain.models import LLMRequest
from rlm.domain.services.rlm_orchestrator import RLMOrchestrator

from .perf_utils import BenchmarkEnvironment, perf_timer


def _skip_if_no_openai() -> None:
    """Skip test if OpenAI is not available."""
    if importlib.util.find_spec("openai") is None:
        pytest.skip("openai package not installed")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


def _skip_if_no_anthropic() -> None:
    """Skip test if Anthropic is not available."""
    if importlib.util.find_spec("anthropic") is None:
        pytest.skip("anthropic package not installed (install the llm-anthropic extra)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


def _get_openai_adapter():
    """Create an OpenAI adapter with performance-oriented settings."""
    from rlm.adapters.llm.openai import OpenAIAdapter

    model = os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")

    return OpenAIAdapter(
        model=model,
        api_key=api_key,
        base_url=base_url,
        default_request_kwargs={
            "temperature": 0,
            "max_tokens": 100,
        },
    )


def _get_anthropic_adapter():
    """Create an Anthropic adapter with performance-oriented settings."""
    from rlm.adapters.llm.anthropic import AnthropicAdapter

    model = os.environ.get("ANTHROPIC_MODEL") or "claude-3-5-haiku-20241022"
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    return AnthropicAdapter(
        model=model,
        api_key=api_key,
        default_request_kwargs={
            "max_tokens": 100,
        },
    )


# =============================================================================
# OpenAI Performance Tests
# =============================================================================


@pytest.mark.performance
@pytest.mark.live_llm
def test_openai_single_completion_latency() -> None:
    """
    Measure single completion latency for OpenAI.

    This provides a baseline for expected LLM call overhead.
    """
    _skip_if_no_openai()

    llm = _get_openai_adapter()

    # Warm-up call
    llm.complete(LLMRequest(prompt="Say hello"))

    latencies = []
    for _ in range(3):
        start = time.perf_counter()
        cc = llm.complete(LLMRequest(prompt="Return the number 42"))
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        assert cc.response is not None
        assert cc.execution_time > 0

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    print(f"\nOpenAI latency: avg={avg_latency:.2f}s, min={min_latency:.2f}s, max={max_latency:.2f}s")

    # Sanity check: should complete within 30 seconds
    assert avg_latency < 30, f"OpenAI avg latency too high: {avg_latency:.2f}s"


@pytest.mark.performance
@pytest.mark.live_llm
def test_openai_usage_tracking_accuracy() -> None:
    """
    Verify usage tracking matches actual API usage.
    """
    _skip_if_no_openai()

    llm = _get_openai_adapter()

    # Make several calls
    for i in range(3):
        llm.complete(LLMRequest(prompt=f"Count to {i+1}"))

    usage = llm.get_usage_summary()
    model_usage = list(usage.model_usage_summaries.values())[0]

    assert model_usage.total_calls == 3
    assert model_usage.total_input_tokens > 0
    assert model_usage.total_output_tokens > 0

    print(f"\nOpenAI usage: calls={model_usage.total_calls}, "
          f"input={model_usage.total_input_tokens}, output={model_usage.total_output_tokens}")


@pytest.mark.performance
@pytest.mark.live_llm
def test_openai_orchestrator_iteration_timing() -> None:
    """
    Measure orchestrator iteration timing with real OpenAI calls.

    This tests the full stack including:
    - LLM call latency
    - Response parsing
    - Code execution
    - Usage tracking
    """
    _skip_if_no_openai()

    llm = _get_openai_adapter()
    env = BenchmarkEnvironment()

    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    with perf_timer() as timing:
        result = orchestrator.completion(
            prompt="Calculate 2+2 and return FINAL(4)",
            max_iterations=3,
            max_depth=1,
        )

    assert result.response is not None
    print(f"\nOpenAI orchestrator: elapsed={timing.elapsed_seconds:.2f}s, "
          f"response_len={len(result.response)}")


@pytest.mark.performance
@pytest.mark.live_llm
async def test_openai_async_completion_latency() -> None:
    """
    Measure async completion latency for OpenAI.
    """
    _skip_if_no_openai()

    llm = _get_openai_adapter()

    # Warm-up
    await llm.acomplete(LLMRequest(prompt="Say hello"))

    latencies = []
    for _ in range(3):
        start = time.perf_counter()
        cc = await llm.acomplete(LLMRequest(prompt="Return the number 42"))
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        assert cc.response is not None

    avg_latency = sum(latencies) / len(latencies)
    print(f"\nOpenAI async latency: avg={avg_latency:.2f}s")


# =============================================================================
# Anthropic Performance Tests
# =============================================================================


@pytest.mark.performance
@pytest.mark.live_llm
def test_anthropic_single_completion_latency() -> None:
    """
    Measure single completion latency for Anthropic.
    """
    _skip_if_no_anthropic()

    llm = _get_anthropic_adapter()

    # Warm-up call
    llm.complete(LLMRequest(prompt="Say hello"))

    latencies = []
    for _ in range(3):
        start = time.perf_counter()
        cc = llm.complete(LLMRequest(prompt="Return the number 42"))
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        assert cc.response is not None
        assert cc.execution_time > 0

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    print(f"\nAnthropic latency: avg={avg_latency:.2f}s, min={min_latency:.2f}s, max={max_latency:.2f}s")

    # Sanity check
    assert avg_latency < 30, f"Anthropic avg latency too high: {avg_latency:.2f}s"


@pytest.mark.performance
@pytest.mark.live_llm
def test_anthropic_usage_tracking_accuracy() -> None:
    """
    Verify usage tracking matches actual API usage.
    """
    _skip_if_no_anthropic()

    llm = _get_anthropic_adapter()

    # Make several calls
    for i in range(3):
        llm.complete(LLMRequest(prompt=f"Count to {i+1}"))

    usage = llm.get_usage_summary()
    model_usage = list(usage.model_usage_summaries.values())[0]

    assert model_usage.total_calls == 3
    assert model_usage.total_input_tokens > 0
    assert model_usage.total_output_tokens > 0

    print(f"\nAnthropic usage: calls={model_usage.total_calls}, "
          f"input={model_usage.total_input_tokens}, output={model_usage.total_output_tokens}")


@pytest.mark.performance
@pytest.mark.live_llm
def test_anthropic_orchestrator_iteration_timing() -> None:
    """
    Measure orchestrator iteration timing with real Anthropic calls.
    """
    _skip_if_no_anthropic()

    llm = _get_anthropic_adapter()
    env = BenchmarkEnvironment()

    orchestrator = RLMOrchestrator(llm=llm, environment=env)

    with perf_timer() as timing:
        result = orchestrator.completion(
            prompt="Calculate 2+2 and return FINAL(4)",
            max_iterations=3,
            max_depth=1,
        )

    assert result.response is not None
    print(f"\nAnthropic orchestrator: elapsed={timing.elapsed_seconds:.2f}s, "
          f"response_len={len(result.response)}")


@pytest.mark.performance
@pytest.mark.live_llm
async def test_anthropic_async_completion_latency() -> None:
    """
    Measure async completion latency for Anthropic.
    """
    _skip_if_no_anthropic()

    llm = _get_anthropic_adapter()

    # Warm-up
    await llm.acomplete(LLMRequest(prompt="Say hello"))

    latencies = []
    for _ in range(3):
        start = time.perf_counter()
        cc = await llm.acomplete(LLMRequest(prompt="Return the number 42"))
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        assert cc.response is not None

    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAnthropic async latency: avg={avg_latency:.2f}s")


# =============================================================================
# Comparative Tests
# =============================================================================


@pytest.mark.performance
@pytest.mark.live_llm
def test_provider_comparison_simple_task() -> None:
    """
    Compare performance between available providers on a simple task.

    Useful for selecting the best provider for latency-sensitive operations.
    """
    results = {}

    # Test OpenAI if available
    try:
        _skip_if_no_openai()
        llm = _get_openai_adapter()
        llm.complete(LLMRequest(prompt="warmup"))  # Warm up

        start = time.perf_counter()
        cc = llm.complete(LLMRequest(prompt="What is 2+2? Answer with just the number."))
        elapsed = time.perf_counter() - start
        results["openai"] = {
            "latency": elapsed,
            "response": cc.response[:50] if cc.response else "",
            "model": llm.model_name,
        }
    except pytest.skip.Exception:
        pass

    # Test Anthropic if available
    try:
        _skip_if_no_anthropic()
        llm = _get_anthropic_adapter()
        llm.complete(LLMRequest(prompt="warmup"))  # Warm up

        start = time.perf_counter()
        cc = llm.complete(LLMRequest(prompt="What is 2+2? Answer with just the number."))
        elapsed = time.perf_counter() - start
        results["anthropic"] = {
            "latency": elapsed,
            "response": cc.response[:50] if cc.response else "",
            "model": llm.model_name,
        }
    except pytest.skip.Exception:
        pass

    if not results:
        pytest.skip("No LLM providers available")

    print("\n=== Provider Comparison ===")
    for provider, data in sorted(results.items(), key=lambda x: x[1]["latency"]):
        print(f"{provider} ({data['model']}): {data['latency']:.2f}s - {data['response']!r}")

    # At least one provider should respond in reasonable time
    min_latency = min(r["latency"] for r in results.values())
    assert min_latency < 30, f"All providers too slow, fastest: {min_latency:.2f}s"
