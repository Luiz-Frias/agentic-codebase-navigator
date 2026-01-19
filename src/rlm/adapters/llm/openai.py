from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

from rlm.adapters.base import BaseLLMAdapter
from rlm.adapters.llm.provider_base import (
    UsageTracker,
    count_openai_prompt_tokens,
    extract_finish_reason_openai,
    extract_openai_style_token_usage,
    extract_text_from_chat_response,
    extract_tool_calls_openai,
    prompt_to_messages,
    tool_choice_to_openai_format,
    tool_definition_to_openai_format,
)
from rlm.domain.errors import LLMError
from rlm.domain.models import ChatCompletion, LLMRequest, UsageSummary

if TYPE_CHECKING:
    from types import ModuleType


def _require_openai() -> ModuleType:  # type: ignore[reportAny]  # OpenAI module has no stubs
    """
    Lazily import the OpenAI SDK.

    This keeps the default install lightweight and avoids importing optional
    dependencies unless the adapter is actually used.
    """
    try:
        import openai  # type: ignore[import-not-found]
    except Exception as e:
        raise ImportError(
            "OpenAI adapter selected but the 'openai' package is not installed. "
            "Install it with: `pip install rlm[llm-openai]`.",
        ) from e
    return openai  # type: ignore[reportAny]  # No type stubs available for openai module


def _safe_openai_error_message(exc: BaseException, /) -> str:
    """
    Convert OpenAI SDK exceptions to safe, user-facing messages.

    We intentionally avoid leaking stack traces or provider response bodies.
    """
    if isinstance(exc, TimeoutError):
        return "OpenAI request timed out"
    if isinstance(exc, (ConnectionError, OSError)):
        return "OpenAI connection error"

    name = type(exc).__name__
    match name:
        case "AuthenticationError" | "PermissionDeniedError":
            return "OpenAI authentication failed (check OPENAI_API_KEY)"
        case "RateLimitError":
            return "OpenAI rate limit exceeded"
        case "BadRequestError" | "UnprocessableEntityError":
            return "OpenAI request rejected"
        case _:
            return "OpenAI request failed"


@dataclass
class OpenAIAdapter(BaseLLMAdapter):
    """
    Adapter skeleton: OpenAI SDK -> domain `LLMPort`.

    Phase 4 will implement real request/response mapping and usage extraction.
    This class exists as a lazy-import boundary and a typed configuration surface.
    """

    model: str
    api_key: str | None = None
    base_url: str | None = None
    default_request_kwargs: dict[str, Any] = field(default_factory=dict)
    _client_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _client: Any | None = field(default=None, init=False, repr=False)  # type: ignore[reportAny]  # OpenAI client has no stubs
    _async_client: Any | None = field(default=None, init=False, repr=False)  # type: ignore[reportAny]  # OpenAI client has no stubs

    _usage_tracker: UsageTracker = field(default_factory=UsageTracker, init=False, repr=False)

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def supports_tools(self) -> bool:
        """OpenAI adapter supports native function calling."""
        return True

    def count_prompt_tokens(self, request: LLMRequest, /) -> int | None:
        model = request.model or self.model
        return count_openai_prompt_tokens(request.prompt, request.tools, model)

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        openai = _require_openai()
        client = self._get_client(openai)  # type: ignore[reportAny]  # OpenAI client has no stubs

        model = request.model or self.model
        messages = prompt_to_messages(request.prompt)  # type: ignore[reportAny]  # Returns OpenAI message dict

        # Build API kwargs with tools if provided
        api_kwargs: dict[str, Any] = {**self.default_request_kwargs}
        if request.tools:
            api_kwargs["tools"] = [tool_definition_to_openai_format(t) for t in request.tools]  # type: ignore[reportAny]  # OpenAI tool format
        if request.tool_choice is not None:
            api_kwargs["tool_choice"] = tool_choice_to_openai_format(request.tool_choice)  # type: ignore[reportAny]  # OpenAI tool choice format

        start = time.perf_counter()
        try:
            resp = client.chat.completions.create(model=model, messages=messages, **api_kwargs)  # type: ignore[reportAny]  # OpenAI response has no stubs
        except Exception as e:
            raise LLMError(_safe_openai_error_message(e)) from None
        end = time.perf_counter()

        # Extract tool calls (may be None if no tools called) - unwrap() raises LLMError on malformed
        tool_calls = extract_tool_calls_openai(resp).unwrap()  # type: ignore[reportAny]  # Extracts from OpenAI response
        finish_reason = extract_finish_reason_openai(resp)  # type: ignore[reportAny]  # Extracts from OpenAI response

        # Extract text response (may be empty if tool_calls present)
        try:
            text = extract_text_from_chat_response(resp)  # type: ignore[reportAny]  # Extracts from OpenAI response
        except Exception as e:
            # Response may have no text content when tool_calls are present
            if tool_calls:
                text = ""
            else:
                raise LLMError(f"OpenAI response invalid: {e}") from None

        in_tokens, out_tokens = extract_openai_style_token_usage(resp)  # type: ignore[reportAny]  # Extracts from OpenAI response
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
        openai = _require_openai()
        client = self._get_async_client(openai)  # type: ignore[reportAny]  # OpenAI client has no stubs

        model = request.model or self.model
        messages = prompt_to_messages(request.prompt)  # type: ignore[reportAny]  # Returns OpenAI message dict

        # Build API kwargs with tools if provided
        api_kwargs: dict[str, Any] = {**self.default_request_kwargs}
        if request.tools:
            api_kwargs["tools"] = [tool_definition_to_openai_format(t) for t in request.tools]  # type: ignore[reportAny]  # OpenAI tool format
        if request.tool_choice is not None:
            api_kwargs["tool_choice"] = tool_choice_to_openai_format(request.tool_choice)  # type: ignore[reportAny]  # OpenAI tool choice format

        start = time.perf_counter()
        try:
            resp = await client.chat.completions.create(  # type: ignore[reportAny]  # OpenAI response has no stubs
                model=model,
                messages=messages,
                **api_kwargs,
            )
        except Exception as e:
            raise LLMError(_safe_openai_error_message(e)) from None
        end = time.perf_counter()

        # Extract tool calls (may be None if no tools called) - unwrap() raises LLMError on malformed
        tool_calls = extract_tool_calls_openai(resp).unwrap()  # type: ignore[reportAny]  # Extracts from OpenAI response
        finish_reason = extract_finish_reason_openai(resp)  # type: ignore[reportAny]  # Extracts from OpenAI response

        # Extract text response (may be empty if tool_calls present)
        try:
            text = extract_text_from_chat_response(resp)  # type: ignore[reportAny]  # Extracts from OpenAI response
        except Exception as e:
            # Response may have no text content when tool_calls are present
            if tool_calls:
                text = ""
            else:
                raise LLMError(f"OpenAI response invalid: {e}") from None

        in_tokens, out_tokens = extract_openai_style_token_usage(resp)  # type: ignore[reportAny]  # Extracts from OpenAI response
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

    def _get_client(self, openai: ModuleType, /) -> Any:  # type: ignore[reportAny]  # OpenAI client has no stubs
        with self._client_lock:
            if self._client is not None:
                return self._client  # type: ignore[reportAny]  # OpenAI client has no stubs

            client_cls = getattr(openai, "OpenAI", None)  # type: ignore[reportAny]  # Dynamic import
            if client_cls is None:
                raise ImportError(
                    "OpenAI SDK API mismatch: expected `openai.OpenAI` class. "
                    "Please upgrade `openai` (install `agentic-codebase-navigator[llm-openai]`).",
                )

            kwargs: dict[str, Any] = {}
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key
            if self.base_url is not None:
                kwargs["base_url"] = self.base_url

            self._client = client_cls(**kwargs)  # type: ignore[reportAny]  # OpenAI client has no stubs
            return self._client  # type: ignore[reportAny]  # OpenAI client has no stubs

    def _get_async_client(self, openai: ModuleType, /) -> Any:  # type: ignore[reportAny]  # OpenAI client has no stubs
        with self._client_lock:
            if self._async_client is not None:
                return self._async_client  # type: ignore[reportAny]  # OpenAI client has no stubs

            client_cls = getattr(openai, "AsyncOpenAI", None)  # type: ignore[reportAny]  # Dynamic import
            if client_cls is None:
                raise ImportError(
                    "OpenAI SDK API mismatch: expected `openai.AsyncOpenAI` class. "
                    "Please upgrade `openai` (install `agentic-codebase-navigator[llm-openai]`).",
                )

            kwargs: dict[str, Any] = {}
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key
            if self.base_url is not None:
                kwargs["base_url"] = self.base_url

            self._async_client = client_cls(**kwargs)  # type: ignore[reportAny]  # OpenAI client has no stubs
            return self._async_client  # type: ignore[reportAny]  # OpenAI client has no stubs


def build_openai_adapter(*, model: str, api_key: str | None = None, **kwargs: Any) -> OpenAIAdapter:
    """
    Small builder helper for registries/composition roots.

    Keeps adapter construction logic in one place and makes future defaults
    explicit (OCP-friendly).
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("OpenAIAdapter requires a non-empty 'model'")
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        raise ValueError("OpenAIAdapter.api_key must be a non-empty string when provided")
    return OpenAIAdapter(model=model, api_key=api_key, default_request_kwargs=dict(kwargs))
