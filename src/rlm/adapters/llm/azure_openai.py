from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
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


def _require_openai() -> Any:
    """
    Lazily import the OpenAI SDK (used for Azure OpenAI as well).

    Installed via the optional extra: `agentic-codebase-navigator[llm-azure-openai]`.
    """

    try:
        import openai  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001 - dependency boundary
        raise ImportError(
            "Azure OpenAI adapter selected but the 'openai' package is not installed. "
            "Install the optional extra: `agentic-codebase-navigator[llm-azure-openai]`."
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

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        openai = _require_openai()
        client = self._get_client(openai)

        deployment = request.model or self.deployment
        messages = prompt_to_messages(request.prompt)

        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=deployment, messages=messages, **self.default_request_kwargs
            )
        except Exception as e:  # noqa: BLE001 - provider boundary
            raise LLMError(safe_provider_error_message("Azure OpenAI", e)) from None
        end = time.perf_counter()

        text = extract_text_from_chat_response(resp)
        in_tokens, out_tokens = extract_openai_style_token_usage(resp)
        last = self._usage_tracker.record(
            deployment, input_tokens=in_tokens, output_tokens=out_tokens
        )
        # Use the per-call usage returned by `record()` (race-free under concurrency).
        last_usage = UsageSummary(model_usage_summaries={deployment: last})

        return ChatCompletion(
            root_model=deployment,
            prompt=request.prompt,
            response=text,
            usage_summary=last_usage,
            execution_time=end - start,
        )

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        openai = _require_openai()
        client = self._get_async_client(openai)

        deployment = request.model or self.deployment
        messages = prompt_to_messages(request.prompt)

        start = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=deployment, messages=messages, **self.default_request_kwargs
            )
        except Exception as e:  # noqa: BLE001 - provider boundary
            raise LLMError(safe_provider_error_message("Azure OpenAI", e)) from None
        end = time.perf_counter()

        text = extract_text_from_chat_response(resp)
        in_tokens, out_tokens = extract_openai_style_token_usage(resp)
        last = self._usage_tracker.record(
            deployment, input_tokens=in_tokens, output_tokens=out_tokens
        )
        # Use the per-call usage returned by `record()` (race-free under concurrency).
        last_usage = UsageSummary(model_usage_summaries={deployment: last})

        return ChatCompletion(
            root_model=deployment,
            prompt=request.prompt,
            response=text,
            usage_summary=last_usage,
            execution_time=end - start,
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
                    "Please upgrade `openai` (install `agentic-codebase-navigator[llm-azure-openai]`)."
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
                    "Please upgrade `openai` (install `agentic-codebase-navigator[llm-azure-openai]`)."
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
