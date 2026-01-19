from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
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


def _require_portkey() -> Any:
    """
    Lazily import the Portkey SDK.

    Installed via the optional extra: `agentic-codebase-navigator[llm-portkey]`.
    """
    try:
        import portkey_ai  # type: ignore[import-not-found]
    except Exception as e:
        raise ImportError(
            "Portkey adapter selected but the 'portkey-ai' package is not installed. "
            "Install the optional extra: `agentic-codebase-navigator[llm-portkey]`.",
        ) from e
    return portkey_ai


@dataclass
class PortkeyAdapter(BaseLLMAdapter):
    """Adapter skeleton: Portkey SDK -> domain `LLMPort`."""

    model: str
    api_key: str | None = None
    base_url: str | None = "https://api.portkey.ai/v1"
    default_request_kwargs: dict[str, Any] = field(default_factory=dict)

    _client_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _client: Any | None = field(default=None, init=False, repr=False)
    _async_client: Any | None = field(default=None, init=False, repr=False)
    _usage_tracker: UsageTracker = field(default_factory=UsageTracker, init=False, repr=False)

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def supports_tools(self) -> bool:
        """Portkey supports tool passthrough (proxies to underlying providers)."""
        return True

    def count_prompt_tokens(self, request: LLMRequest, /) -> int | None:
        model = request.model or self.model
        return count_openai_prompt_tokens(request.prompt, request.tools, model)

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        portkey = _require_portkey()
        client = self._get_client(portkey)

        model = request.model or self.model
        messages = prompt_to_messages(request.prompt)

        # Build API kwargs with tools if provided (Portkey uses OpenAI format)
        api_kwargs: dict[str, Any] = {**self.default_request_kwargs}
        if request.tools:
            api_kwargs["tools"] = [tool_definition_to_openai_format(t) for t in request.tools]
        if request.tool_choice is not None:
            api_kwargs["tool_choice"] = tool_choice_to_openai_format(request.tool_choice)

        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(model=model, messages=messages, **api_kwargs)
        except Exception as e:
            raise LLMError(safe_provider_error_message("Portkey", e)) from None
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
                raise LLMError("Portkey response invalid") from None

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
        portkey = _require_portkey()
        client = self._get_async_client(portkey)

        model = request.model or self.model
        messages = prompt_to_messages(request.prompt)

        # Build API kwargs with tools if provided (Portkey uses OpenAI format)
        api_kwargs: dict[str, Any] = {**self.default_request_kwargs}
        if request.tools:
            api_kwargs["tools"] = [tool_definition_to_openai_format(t) for t in request.tools]
        if request.tool_choice is not None:
            api_kwargs["tool_choice"] = tool_choice_to_openai_format(request.tool_choice)

        start = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                **api_kwargs,
            )
        except Exception as e:
            raise LLMError(safe_provider_error_message("Portkey", e)) from None
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
                raise LLMError("Portkey response invalid") from None

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

    def _get_client(self, portkey: Any, /) -> Any:
        with self._client_lock:
            if self._client is not None:
                return self._client

            client_cls = getattr(portkey, "Portkey", None)
            if client_cls is None:
                raise ImportError(
                    "Portkey SDK API mismatch: expected `portkey_ai.Portkey` class. "
                    "Please upgrade `portkey-ai` (install `agentic-codebase-navigator[llm-portkey]`).",
                )

            kwargs: dict[str, Any] = {}
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key
            if self.base_url is not None:
                kwargs["base_url"] = self.base_url

            self._client = client_cls(**kwargs)
            return self._client

    def _get_async_client(self, portkey: Any, /) -> Any:
        with self._client_lock:
            if self._async_client is not None:
                return self._async_client

            client_cls = getattr(portkey, "AsyncPortkey", None)
            if client_cls is None:
                raise ImportError(
                    "Portkey SDK API mismatch: expected `portkey_ai.AsyncPortkey` class. "
                    "Please upgrade `portkey-ai` (install `agentic-codebase-navigator[llm-portkey]`).",
                )

            kwargs: dict[str, Any] = {}
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key
            if self.base_url is not None:
                kwargs["base_url"] = self.base_url

            self._async_client = client_cls(**kwargs)
            return self._async_client


def build_portkey_adapter(
    *,
    model: str,
    api_key: str | None = None,
    **kwargs: Any,
) -> PortkeyAdapter:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("PortkeyAdapter requires a non-empty 'model'")
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        raise ValueError("PortkeyAdapter.api_key must be a non-empty string when provided")
    base_url = kwargs.pop("base_url", "https://api.portkey.ai/v1")
    if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
        raise ValueError("PortkeyAdapter.base_url must be a non-empty string when provided")
    return PortkeyAdapter(
        model=model,
        api_key=api_key,
        base_url=base_url,
        default_request_kwargs=dict(kwargs),
    )
