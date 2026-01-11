"""
Performance testing utilities for RLM profiling.

Provides fixtures, helpers, and data generators for benchmarking
speed, memory, and reliability characteristics.

Live LLM Support:
    Set RLM_LIVE_LLM=1 to use real LLM providers instead of mocks.
    Requires appropriate API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY).

    Environment variables:
        RLM_LIVE_LLM=1          Enable live LLM calls
        RLM_LIVE_LLM_PROVIDER   Provider to use: "openai" (default) or "anthropic"
        OPENAI_MODEL            OpenAI model (default: gpt-4o-mini)
        ANTHROPIC_MODEL         Anthropic model (default: claude-3-5-haiku-20241022)
"""

from __future__ import annotations

import gc
import importlib.util
import os
import sys
import time
import tracemalloc
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from rlm.domain.models import (
    ChatCompletion,
    CodeBlock,
    Iteration,
    LLMRequest,
    ModelUsageSummary,
    ReplResult,
    UsageSummary,
)
from rlm.domain.ports import EnvironmentPort, LLMPort


@dataclass
class TimingResult:
    """Result of a timed code block."""

    elapsed_seconds: float
    iterations: int = 1

    @property
    def per_iteration_ms(self) -> float:
        return (self.elapsed_seconds / self.iterations) * 1000


@contextmanager
def perf_timer() -> Generator[TimingResult, None, None]:
    """Context manager for timing code blocks with high precision."""
    result = TimingResult(elapsed_seconds=0.0)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_seconds = time.perf_counter() - start


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    current_bytes: int
    peak_bytes: int
    traced_blocks: int


@dataclass
class MemoryResult:
    """Result of memory tracking."""

    before: MemorySnapshot
    after: MemorySnapshot
    allocations: list[tuple[int, str]] = field(default_factory=list)

    @property
    def delta_bytes(self) -> int:
        return self.after.current_bytes - self.before.current_bytes

    @property
    def peak_bytes(self) -> int:
        return self.after.peak_bytes


@contextmanager
def memory_tracker(top_n: int = 10) -> Generator[MemoryResult, None, None]:
    """
    Context manager for tracking memory allocations.

    Uses tracemalloc to capture allocation statistics.
    """
    gc.collect()
    tracemalloc.start()

    current, peak = tracemalloc.get_traced_memory()
    before = MemorySnapshot(
        current_bytes=current,
        peak_bytes=peak,
        traced_blocks=len(tracemalloc.take_snapshot().traces),
    )

    result = MemoryResult(
        before=before,
        after=MemorySnapshot(current_bytes=0, peak_bytes=0, traced_blocks=0),
    )

    try:
        yield result
    finally:
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()

        result.after = MemorySnapshot(
            current_bytes=current, peak_bytes=peak, traced_blocks=len(snapshot.traces)
        )

        # Get top allocations
        top_stats = snapshot.statistics("lineno")[:top_n]
        result.allocations = [(stat.size, str(stat.traceback)) for stat in top_stats]

        tracemalloc.stop()


def generate_large_context(
    *,
    num_keys: int = 100,
    value_size: int = 1000,
    nested_depth: int = 0,
) -> dict[str, Any]:
    """
    Generate a large context payload for testing.

    Args:
        num_keys: Number of top-level keys
        value_size: Size of string values
        nested_depth: Depth of nested dictionaries
    """
    value = "x" * value_size

    def build_nested(depth: int) -> dict[str, Any]:
        if depth <= 0:
            return {"value": value, "count": 42, "flag": True}
        return {"nested": build_nested(depth - 1), "level": depth}

    return {f"key_{i}": build_nested(nested_depth) for i in range(num_keys)}


def generate_deep_nested_data(depth: int = 100, width: int = 3) -> dict[str, Any]:
    """
    Generate deeply nested data structure for serialization stress tests.

    Warning: Very deep structures may hit Python's recursion limit.
    """
    if depth <= 0:
        return {"leaf": "value", "number": 42, "items": [1, 2, 3]}

    return {
        "level": depth,
        "child": generate_deep_nested_data(depth - 1, width),
        "siblings": [{"sibling": i} for i in range(width)],
    }


def generate_response_with_code_blocks(num_blocks: int = 10, code_size: int = 100) -> str:
    """Generate a response with multiple code blocks for parsing tests."""
    blocks = []
    for i in range(num_blocks):
        code = f"# Block {i}\n" + f"x = {i}\n" * (code_size // 10)
        blocks.append(f"```repl\n{code}\n```")
    return "\n\nSome text between blocks.\n\n".join(blocks)


def generate_long_text_with_final(length: int = 100000, final_answer: str = "42") -> str:
    """Generate long text with FINAL() marker at the end."""
    padding = "Lorem ipsum dolor sit amet. " * (length // 30)
    return f"{padding}\n\nFINAL({final_answer})"


class BenchmarkLLM(LLMPort):
    """
    LLM adapter for benchmarking that returns predictable responses.

    Configurable delay to simulate network latency.
    """

    def __init__(
        self,
        *,
        model_name: str = "benchmark-llm",
        response_template: str = "Response {n}",
        delay_seconds: float = 0.0,
        include_final: bool = False,
        final_after: int = 5,
    ) -> None:
        self._model_name = model_name
        self._response_template = response_template
        self._delay = delay_seconds
        self._include_final = include_final
        self._final_after = final_after
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    def _make_response(self) -> str:
        self._call_count += 1
        base = self._response_template.format(n=self._call_count)

        if self._include_final and self._call_count >= self._final_after:
            return f"{base}\n\nFINAL(done after {self._call_count} iterations)"

        # Add a code block to trigger iteration continuation
        return f"{base}\n\n```repl\nx = {self._call_count}\n```"

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        if self._delay > 0:
            time.sleep(self._delay)

        response = self._make_response()

        # Simulate token counting
        prompt_str = str(request.prompt)
        input_tokens = len(prompt_str) // 4
        output_tokens = len(response) // 4
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

        return ChatCompletion(
            root_model=self._model_name,
            prompt=request.prompt,
            response=response,
            usage_summary=UsageSummary(
                model_usage_summaries={
                    self._model_name: ModelUsageSummary(
                        total_calls=1,
                        total_input_tokens=input_tokens,
                        total_output_tokens=output_tokens,
                    )
                }
            ),
            execution_time=self._delay,
        )

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        return self.complete(request)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                self._model_name: ModelUsageSummary(
                    total_calls=self._call_count,
                    total_input_tokens=self._total_input_tokens,
                    total_output_tokens=self._total_output_tokens,
                )
            }
        )

    def get_last_usage(self) -> UsageSummary:
        return self.get_usage_summary()


class BenchmarkEnvironment(EnvironmentPort):
    """
    Environment adapter for benchmarking with configurable execution time.
    """

    def __init__(
        self,
        *,
        execute_delay_seconds: float = 0.0,
        result_size: int = 100,
    ) -> None:
        self._delay = execute_delay_seconds
        self._result_size = result_size
        self._context: Any = None
        self._execution_count = 0

    def load_context(self, context_payload: Any, /) -> None:
        self._context = context_payload

    def execute_code(self, code: str, /) -> ReplResult:
        if self._delay > 0:
            time.sleep(self._delay)

        self._execution_count += 1
        stdout = f"Execution {self._execution_count}: " + "x" * self._result_size

        return ReplResult(
            stdout=stdout,
            stderr="",
            locals={"x": self._execution_count},
            llm_calls=[],
            execution_time=self._delay,
        )

    def cleanup(self) -> None:
        self._context = None
        self._execution_count = 0


def make_iteration(
    *,
    num_code_blocks: int = 1,
    response_size: int = 500,
    stdout_size: int = 200,
) -> Iteration:
    """Create a synthetic iteration for testing format_iteration performance."""
    response = "Some response text. " * (response_size // 20)

    code_blocks = []
    for i in range(num_code_blocks):
        code_blocks.append(
            CodeBlock(
                code=f"x = {i}\nprint(x)",
                result=ReplResult(
                    stdout="x" * stdout_size,
                    stderr="",
                    locals={"x": i, "y": i * 2},
                    llm_calls=[],
                    execution_time=0.001,
                ),
            )
        )

    return Iteration(
        correlation_id="bench-corr-id",
        prompt=[{"role": "user", "content": "test prompt"}],
        response=response,
        code_blocks=code_blocks,
        final_answer=None,
        iteration_time=0.1,
        iteration_usage_summary=None,
        cumulative_usage_summary=None,
    )


def measure_object_size(obj: Any) -> int:
    """
    Recursively measure the approximate memory size of an object.

    Note: This is an approximation and may not capture all referenced memory.
    """
    seen = set()

    def _size(o: Any) -> int:
        obj_id = id(o)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        size = sys.getsizeof(o)

        if isinstance(o, dict):
            size += sum(_size(k) + _size(v) for k, v in o.items())
        elif isinstance(o, (list, tuple, set, frozenset)):
            size += sum(_size(item) for item in o)
        elif hasattr(o, "__dict__"):
            size += _size(o.__dict__)
        elif hasattr(o, "__slots__"):
            size += sum(_size(getattr(o, slot, None)) for slot in o.__slots__ if hasattr(o, slot))

        return size

    return _size(obj)


# =============================================================================
# Live LLM Support
# =============================================================================


def is_live_llm_enabled() -> bool:
    """Check if live LLM mode is enabled via RLM_LIVE_LLM=1."""
    raw = (os.environ.get("RLM_LIVE_LLM") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def get_live_llm_provider() -> str:
    """Get the configured live LLM provider (openai or anthropic)."""
    return (os.environ.get("RLM_LIVE_LLM_PROVIDER") or "openai").strip().lower()


def _create_openai_adapter() -> LLMPort:
    """Create an OpenAI adapter for live testing."""
    if importlib.util.find_spec("openai") is None:
        raise RuntimeError("openai package not installed")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    from rlm.adapters.llm.openai import OpenAIAdapter

    model = os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
    return OpenAIAdapter(
        model=model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        default_request_kwargs={"temperature": 0, "max_tokens": 256},
    )


def _create_anthropic_adapter() -> LLMPort:
    """Create an Anthropic adapter for live testing."""
    if importlib.util.find_spec("anthropic") is None:
        raise RuntimeError("anthropic package not installed")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    from rlm.adapters.llm.anthropic import AnthropicAdapter

    model = os.environ.get("ANTHROPIC_MODEL") or "claude-3-5-haiku-20241022"
    return AnthropicAdapter(
        model=model,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        default_request_kwargs={"max_tokens": 256},
    )


def get_llm_for_benchmark(
    *,
    include_final: bool = False,
    final_after: int = 5,
    delay_seconds: float = 0.0,
) -> LLMPort:
    """
    Get an LLM adapter for benchmarking.

    If RLM_LIVE_LLM=1, returns a real provider adapter.
    Otherwise, returns a BenchmarkLLM mock.

    Args:
        include_final: For mock LLM, whether to include FINAL() after N iterations
        final_after: For mock LLM, which iteration to include FINAL()
        delay_seconds: For mock LLM, simulated delay per call

    Returns:
        LLMPort: Either a live adapter or BenchmarkLLM mock
    """
    if is_live_llm_enabled():
        provider = get_live_llm_provider()
        if provider == "anthropic":
            return _create_anthropic_adapter()
        else:
            return _create_openai_adapter()

    return BenchmarkLLM(
        include_final=include_final,
        final_after=final_after,
        delay_seconds=delay_seconds,
    )


def get_llm_provider_info() -> dict[str, Any]:
    """
    Get information about the current LLM configuration.

    Returns dict with:
        - live: bool - whether using live LLM
        - provider: str - provider name or "mock"
        - model: str - model name
    """
    if is_live_llm_enabled():
        provider = get_live_llm_provider()
        if provider == "anthropic":
            model = os.environ.get("ANTHROPIC_MODEL") or "claude-3-5-haiku-20241022"
        else:
            model = os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        return {"live": True, "provider": provider, "model": model}

    return {"live": False, "provider": "mock", "model": "benchmark-llm"}
