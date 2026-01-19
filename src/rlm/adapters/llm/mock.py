from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

from rlm.adapters.base import BaseLLMAdapter
from rlm.domain.errors import LLMError
from rlm.domain.models import (
    ChatCompletion,
    LLMRequest,
    ModelUsageSummary,
    UsageSummary,
)

if TYPE_CHECKING:
    from rlm.domain.agent_ports import ToolCallRequest
    from rlm.domain.types import Prompt


def _prompt_preview(prompt: Prompt, /, *, max_chars: int = 50) -> str:
    """
    Deterministically stringify a prompt for mock responses.

    Avoids depending on provider-specific message formats while keeping behavior
    stable across test runs.
    """
    match prompt:
        case str():
            text = prompt
        case list():
            # Try to extract the last message content if it looks OpenAI-style.
            try:
                last = prompt[-1]
            except Exception:
                text = repr(prompt)
            else:
                if isinstance(last, dict) and "content" in last:
                    text = str(last.get("content"))
                else:
                    text = repr(prompt)
        case dict():
            # Common legacy shape: {"prompt": "..."} or {"messages": [...]}
            text = str(prompt.get("prompt")) if "prompt" in prompt else repr(prompt)
        case _:
            text = repr(prompt)

    text = text.replace("\n", "\\n")
    return text[:max_chars]


def _estimate_input_tokens(prompt: Prompt, /) -> int:
    """
    A tiny, deterministic token estimate for tests.

    We intentionally avoid any external tokenizer dependency.
    """
    preview = _prompt_preview(prompt, max_chars=10_000)
    # Count whitespace-separated chunks as an extremely rough proxy.
    return 0 if not preview.strip() else len(preview.split())


def _estimate_output_tokens(text: str, /) -> int:
    return 0 if not text.strip() else len(text.split())


@dataclass
class MockLLMAdapter(BaseLLMAdapter):
    """
    Deterministic, dependency-free `LLMPort` implementation for tests/examples.

    Behavior:
    - If `script` is provided, each call pops one item:
      - `str` => returned as completion response
      - `Exception` => raised
      - `dict` with "tool_calls" key => returned as tool call response
    - Otherwise, returns an "echo-like" response derived from the prompt.

    Tool Call Response Format:
    A dict with optional keys:
    - "tool_calls": list[ToolCallRequest] - tool calls to return
    - "response": str - optional text response alongside tools (default: "")
    - "finish_reason": str - optional finish reason (default: "tool_calls" if tool_calls present)

    Example script with tool calls:
        script = [
            {"tool_calls": [{"id": "call_1", "name": "get_weather", "arguments": {"city": "NYC"}}]},
            "The weather is sunny",  # Final answer after tool execution
        ]
    """

    model: str = "mock-model"
    script: list[str | Exception | dict[str, Any]] | None = None
    max_prompt_preview_chars: int = 50
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    # usage accounting
    _usage: UsageSummary = field(
        default_factory=lambda: UsageSummary(model_usage_summaries={}),
        init=False,
        repr=False,
    )
    _last_usage: UsageSummary = field(
        default_factory=lambda: UsageSummary(model_usage_summaries={}),
        init=False,
        repr=False,
    )

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def supports_tools(self) -> bool:
        """MockLLMAdapter supports tool calling via scripted responses."""
        return True

    def count_prompt_tokens(self, request: LLMRequest, /) -> int | None:
        return _estimate_input_tokens(request.prompt)

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        start = time.perf_counter()
        script_item = self._next_script_item(request.prompt)
        end = time.perf_counter()

        # Parse script item into response components
        response_text: str
        tool_calls: list[ToolCallRequest] | None = None
        finish_reason: str | None = None

        if isinstance(script_item, dict):
            # Tool call response
            tool_calls = script_item.get("tool_calls")
            response_text = str(script_item.get("response", ""))
            finish_reason = script_item.get("finish_reason")
            if finish_reason is None and tool_calls:
                finish_reason = "tool_calls"
            elif finish_reason is None:
                finish_reason = "stop"
        else:
            # Plain text response
            response_text = script_item
            finish_reason = "stop"

        in_tokens = _estimate_input_tokens(request.prompt)
        out_tokens = _estimate_output_tokens(response_text)

        with self._lock:
            mus = self._usage.model_usage_summaries.get(self.model_name)
            if mus is None:
                mus = ModelUsageSummary()
                self._usage.model_usage_summaries[self.model_name] = mus
            mus.total_calls += 1
            mus.total_input_tokens += in_tokens
            mus.total_output_tokens += out_tokens

            self._last_usage = UsageSummary(
                model_usage_summaries={
                    self.model_name: ModelUsageSummary(
                        total_calls=1,
                        total_input_tokens=in_tokens,
                        total_output_tokens=out_tokens,
                    ),
                },
            )

        return ChatCompletion(
            root_model=request.model or self.model_name,
            prompt=request.prompt,
            response=response_text,
            usage_summary=self._last_usage,
            execution_time=end - start,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        # Deterministic and dependency-free: reuse sync implementation.
        return self.complete(request)

    def get_usage_summary(self) -> UsageSummary:
        with self._lock:
            # Return a deep copy to protect internal mutability. `ModelUsageSummary`
            # instances are mutable, so copying only the dict is insufficient.
            return UsageSummary(
                model_usage_summaries={
                    model: ModelUsageSummary(
                        total_calls=mus.total_calls,
                        total_input_tokens=mus.total_input_tokens,
                        total_output_tokens=mus.total_output_tokens,
                    )
                    for model, mus in self._usage.model_usage_summaries.items()
                },
            )

    def get_last_usage(self) -> UsageSummary:
        with self._lock:
            return UsageSummary(
                model_usage_summaries={
                    model: ModelUsageSummary(
                        total_calls=mus.total_calls,
                        total_input_tokens=mus.total_input_tokens,
                        total_output_tokens=mus.total_output_tokens,
                    )
                    for model, mus in self._last_usage.model_usage_summaries.items()
                },
            )

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _next_script_item(self, prompt: Prompt, /) -> str | dict[str, Any]:
        """
        Get the next scripted response item.

        Returns:
            - str: plain text response
            - dict: tool call response with "tool_calls" key

        Raises:
            Exception: if the script item is an Exception
            LLMError: if the script is exhausted

        """
        with self._lock:
            script = self.script
            if script:
                item = script.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
            if script is not None and not script:
                # Scripted mode but exhausted: raise a domain error with a safe message.
                raise LLMError("MockLLMAdapter: no scripted responses left")

        preview = _prompt_preview(prompt, max_chars=self.max_prompt_preview_chars)
        return f"Mock response to: {preview}"
