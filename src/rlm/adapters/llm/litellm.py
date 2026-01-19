from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rlm.adapters.base import BaseLLMAdapter
from rlm.adapters.llm.provider_base import (
    UsageTracker,
    count_openai_prompt_tokens,
    extract_finish_reason_openai,
    extract_openai_style_token_usage,
    extract_text_from_chat_response,
    extract_tool_calls_openai,
    prompt_to_messages,
    safe_provider_error_message,
    tool_choice_to_openai_format,
    tool_definition_to_openai_format,
)
from rlm.domain.errors import LLMError
from rlm.domain.models import ChatCompletion, LLMRequest, UsageSummary


def _require_litellm() -> Any:
    """
    Lazily import LiteLLM.

    Installed via the optional extra: `agentic-codebase-navigator[llm-litellm]`.
    """
    try:
        import litellm  # type: ignore[import-not-found]
    except Exception as e:
        raise ImportError(
            "LiteLLM adapter selected but the 'litellm' package is not installed. "
            "Install the optional extra: `agentic-codebase-navigator[llm-litellm]`.",
        ) from e
    return litellm


@dataclass
class LiteLLMAdapter(BaseLLMAdapter):
    """Adapter skeleton: LiteLLM -> domain `LLMPort`."""

    model: str
    default_request_kwargs: dict[str, Any] = field(default_factory=dict)

    _usage_tracker: UsageTracker = field(default_factory=UsageTracker, init=False, repr=False)

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def supports_tools(self) -> bool:
        """LiteLLM supports tool passthrough (handles format conversion internally)."""
        return True

    def count_prompt_tokens(self, request: LLMRequest, /) -> int | None:
        model = request.model or self.model
        return count_openai_prompt_tokens(request.prompt, request.tools, model)

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        litellm = _require_litellm()

        model = request.model or self.model
        messages = prompt_to_messages(request.prompt)

        # Build API kwargs with tools if provided (LiteLLM handles format conversion)
        api_kwargs: dict[str, Any] = {**self.default_request_kwargs}
        if request.tools:
            api_kwargs["tools"] = [tool_definition_to_openai_format(t) for t in request.tools]
        if request.tool_choice is not None:
            api_kwargs["tool_choice"] = tool_choice_to_openai_format(request.tool_choice)

        start = time.perf_counter()
        try:
            resp = litellm.completion(model=model, messages=messages, **api_kwargs)
        except Exception as e:
            raise LLMError(safe_provider_error_message("LiteLLM", e)) from None
        end = time.perf_counter()

        # Extract tool calls (may be None if no tools called) - unwrap() raises LLMError on malformed
        tool_calls = extract_tool_calls_openai(resp).unwrap()
        finish_reason = extract_finish_reason_openai(resp)

        # Extract text response (may be empty if tool_calls present)
        try:
            text = extract_text_from_chat_response(resp)
        except Exception:
            if tool_calls:
                text = ""
            else:
                raise LLMError("LiteLLM response invalid") from None

        in_tokens, out_tokens = extract_openai_style_token_usage(resp)
        last = self._usage_tracker.record(model, input_tokens=in_tokens, output_tokens=out_tokens)
        # Use the per-call usage returned by `record()` (race-free under concurrency).
        last_usage = UsageSummary(model_usage_summaries={model: last})

        return ChatCompletion(
            root_model=model,
            prompt=request.prompt,
            response=text,
            usage_summary=last_usage,
            execution_time=end - start,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        litellm = _require_litellm()

        model = request.model or self.model
        messages = prompt_to_messages(request.prompt)

        # Build API kwargs with tools if provided (LiteLLM handles format conversion)
        api_kwargs: dict[str, Any] = {**self.default_request_kwargs}
        if request.tools:
            api_kwargs["tools"] = [tool_definition_to_openai_format(t) for t in request.tools]
        if request.tool_choice is not None:
            api_kwargs["tool_choice"] = tool_choice_to_openai_format(request.tool_choice)

        start = time.perf_counter()
        try:
            resp = await litellm.acompletion(model=model, messages=messages, **api_kwargs)
        except Exception as e:
            raise LLMError(safe_provider_error_message("LiteLLM", e)) from None
        end = time.perf_counter()

        # Extract tool calls (may be None if no tools called) - unwrap() raises LLMError on malformed
        tool_calls = extract_tool_calls_openai(resp).unwrap()
        finish_reason = extract_finish_reason_openai(resp)

        # Extract text response (may be empty if tool_calls present)
        try:
            text = extract_text_from_chat_response(resp)
        except Exception:
            if tool_calls:
                text = ""
            else:
                raise LLMError("LiteLLM response invalid") from None

        in_tokens, out_tokens = extract_openai_style_token_usage(resp)
        last = self._usage_tracker.record(model, input_tokens=in_tokens, output_tokens=out_tokens)
        # Use the per-call usage returned by `record()` (race-free under concurrency).
        last_usage = UsageSummary(model_usage_summaries={model: last})

        return ChatCompletion(
            root_model=model,
            prompt=request.prompt,
            response=text,
            usage_summary=last_usage,
            execution_time=end - start,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    def get_usage_summary(self) -> UsageSummary:
        return self._usage_tracker.get_usage_summary()

    def get_last_usage(self) -> UsageSummary:
        return self._usage_tracker.get_last_usage()


def build_litellm_adapter(*, model: str, **kwargs: Any) -> LiteLLMAdapter:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("LiteLLMAdapter requires a non-empty 'model'")
    return LiteLLMAdapter(model=model, default_request_kwargs=dict(kwargs))
