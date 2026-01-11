from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rlm.adapters.base import BaseLLMAdapter
from rlm.adapters.llm.provider_base import (
    UsageTracker,
    extract_openai_style_token_usage,
    extract_text_from_chat_response,
    prompt_to_messages,
    safe_provider_error_message,
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
    except Exception as e:  # noqa: BLE001 - dependency boundary
        raise ImportError(
            "LiteLLM adapter selected but the 'litellm' package is not installed. "
            "Install the optional extra: `agentic-codebase-navigator[llm-litellm]`."
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

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        litellm = _require_litellm()

        model = request.model or self.model
        messages = prompt_to_messages(request.prompt)

        start = time.perf_counter()
        try:
            resp = litellm.completion(model=model, messages=messages, **self.default_request_kwargs)
        except Exception as e:  # noqa: BLE001 - provider boundary
            raise LLMError(safe_provider_error_message("LiteLLM", e)) from None
        end = time.perf_counter()

        text = extract_text_from_chat_response(resp)
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
        )

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        litellm = _require_litellm()

        model = request.model or self.model
        messages = prompt_to_messages(request.prompt)

        start = time.perf_counter()
        try:
            resp = await litellm.acompletion(
                model=model, messages=messages, **self.default_request_kwargs
            )
        except Exception as e:  # noqa: BLE001 - provider boundary
            raise LLMError(safe_provider_error_message("LiteLLM", e)) from None
        end = time.perf_counter()

        text = extract_text_from_chat_response(resp)
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
        )

    def get_usage_summary(self) -> UsageSummary:
        return self._usage_tracker.get_usage_summary()

    def get_last_usage(self) -> UsageSummary:
        return self._usage_tracker.get_last_usage()


def build_litellm_adapter(*, model: str, **kwargs: Any) -> LiteLLMAdapter:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("LiteLLMAdapter requires a non-empty 'model'")
    return LiteLLMAdapter(model=model, default_request_kwargs=dict(kwargs))
