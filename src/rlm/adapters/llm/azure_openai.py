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


def _require_openai() -> Any:
    """
    Lazily import the OpenAI SDK (used for Azure OpenAI as well).

    Installed via the optional extra: `agentic-codebase-navigator[llm-azure-openai]`.
    """
    try:
        import openai  # type: ignore[import-not-found]
    except Exception as e:
        raise ImportError(
            "Azure OpenAI adapter selected but the 'openai' package is not installed. "
            "Install the optional extra: `agentic-codebase-navigator[llm-azure-openai]`.",
        ) from e
    return openai


@dataclass
class AzureOpenAIAdapter(BaseLLMAdapter):
    """
    Adapter skeleton: Azure OpenAI (via OpenAI SDK) -> domain `LLMPort`.

    Phase 4 will implement real request/response mapping.
    """

    deployment: str
    api_key: str | None = None
    endpoint: str | None = None
    api_version: str | None = None
    default_request_kwargs: dict[str, Any] = field(default_factory=dict)

    _client_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _client: Any | None = field(default=None, init=False, repr=False)
    _async_client: Any | None = field(default=None, init=False, repr=False)
    _usage_tracker: UsageTracker = field(default_factory=UsageTracker, init=False, repr=False)

    @property
    def model_name(self) -> str:
        # For routing: treat the Azure deployment name as the "model".
        return self.deployment

    @property
    def supports_tools(self) -> bool:
        """Azure OpenAI adapter supports native function calling (same as OpenAI)."""
        return True

    def count_prompt_tokens(self, request: LLMRequest, /) -> int | None:
        deployment = request.model or self.deployment
        return count_openai_prompt_tokens(request.prompt, request.tools, deployment)

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        openai = _require_openai()
        client = self._get_client(openai)

        deployment = request.model or self.deployment
        messages = prompt_to_messages(request.prompt)

        # Build API kwargs with tools if provided
        api_kwargs: dict[str, Any] = {**self.default_request_kwargs}
        if request.tools:
            api_kwargs["tools"] = [tool_definition_to_openai_format(t) for t in request.tools]
        if request.tool_choice is not None:
            api_kwargs["tool_choice"] = tool_choice_to_openai_format(request.tool_choice)

        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(model=deployment, messages=messages, **api_kwargs)
        except Exception as e:
            raise LLMError(safe_provider_error_message("Azure OpenAI", e)) from None
        end = time.perf_counter()

        # Extract tool calls (may be None if no tools called) - unwrap() raises LLMError on malformed
        tool_calls = extract_tool_calls_openai(resp).unwrap()
        finish_reason = extract_finish_reason_openai(resp)

        # Extract text response (may be empty if tool_calls present)
        try:
            text = extract_text_from_chat_response(resp)
        except Exception:
            # Response may have no text content when tool_calls are present
            if tool_calls:
                text = ""
            else:
                raise LLMError("Azure OpenAI response invalid") from None

        in_tokens, out_tokens = extract_openai_style_token_usage(resp)
        last = self._usage_tracker.record(
            deployment,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )
        # Use the per-call usage returned by `record()` (race-free under concurrency).
        last_usage = UsageSummary(model_usage_summaries={deployment: last})

        return ChatCompletion(
            root_model=deployment,
            prompt=request.prompt,
            response=text,
            usage_summary=last_usage,
            execution_time=end - start,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        openai = _require_openai()
        client = self._get_async_client(openai)

        deployment = request.model or self.deployment
        messages = prompt_to_messages(request.prompt)

        # Build API kwargs with tools if provided
        api_kwargs: dict[str, Any] = {**self.default_request_kwargs}
        if request.tools:
            api_kwargs["tools"] = [tool_definition_to_openai_format(t) for t in request.tools]
        if request.tool_choice is not None:
            api_kwargs["tool_choice"] = tool_choice_to_openai_format(request.tool_choice)

        start = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=deployment,
                messages=messages,
                **api_kwargs,
            )
        except Exception as e:
            raise LLMError(safe_provider_error_message("Azure OpenAI", e)) from None
        end = time.perf_counter()

        # Extract tool calls (may be None if no tools called) - unwrap() raises LLMError on malformed
        tool_calls = extract_tool_calls_openai(resp).unwrap()
        finish_reason = extract_finish_reason_openai(resp)

        # Extract text response (may be empty if tool_calls present)
        try:
            text = extract_text_from_chat_response(resp)
        except Exception:
            # Response may have no text content when tool_calls are present
            if tool_calls:
                text = ""
            else:
                raise LLMError("Azure OpenAI response invalid") from None

        in_tokens, out_tokens = extract_openai_style_token_usage(resp)
        last = self._usage_tracker.record(
            deployment,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )
        # Use the per-call usage returned by `record()` (race-free under concurrency).
        last_usage = UsageSummary(model_usage_summaries={deployment: last})

        return ChatCompletion(
            root_model=deployment,
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

    def _get_client(self, openai: Any, /) -> Any:
        with self._client_lock:
            if self._client is not None:
                return self._client

            client_cls = getattr(openai, "AzureOpenAI", None)
            if client_cls is None:
                raise ImportError(
                    "OpenAI SDK API mismatch: expected `openai.AzureOpenAI` class. "
                    "Please upgrade `openai` (install `agentic-codebase-navigator[llm-azure-openai]`).",
                )

            kwargs: dict[str, Any] = {}
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key
            if self.endpoint is not None:
                kwargs["azure_endpoint"] = self.endpoint
            if self.api_version is not None:
                kwargs["api_version"] = self.api_version

            self._client = client_cls(**kwargs)
            return self._client

    def _get_async_client(self, openai: Any, /) -> Any:
        with self._client_lock:
            if self._async_client is not None:
                return self._async_client

            client_cls = getattr(openai, "AsyncAzureOpenAI", None)
            if client_cls is None:
                raise ImportError(
                    "OpenAI SDK API mismatch: expected `openai.AsyncAzureOpenAI` class. "
                    "Please upgrade `openai` (install `agentic-codebase-navigator[llm-azure-openai]`).",
                )

            kwargs: dict[str, Any] = {}
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key
            if self.endpoint is not None:
                kwargs["azure_endpoint"] = self.endpoint
            if self.api_version is not None:
                kwargs["api_version"] = self.api_version

            self._async_client = client_cls(**kwargs)
            return self._async_client


def build_azure_openai_adapter(
    *,
    deployment: str,
    api_key: str | None = None,
    endpoint: str | None = None,
    api_version: str | None = None,
    **kwargs: Any,
) -> AzureOpenAIAdapter:
    if not isinstance(deployment, str) or not deployment.strip():
        raise ValueError("AzureOpenAIAdapter requires a non-empty 'deployment'")
    return AzureOpenAIAdapter(
        deployment=deployment,
        api_key=api_key,
        endpoint=endpoint,
        api_version=api_version,
        default_request_kwargs=dict(kwargs),
    )
