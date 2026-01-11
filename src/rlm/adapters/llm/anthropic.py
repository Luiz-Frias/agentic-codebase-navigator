from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from rlm.adapters.base import BaseLLMAdapter
from rlm.adapters.llm.provider_base import (
    UsageTracker,
    prompt_to_messages,
    safe_provider_error_message,
)
from rlm.domain.errors import LLMError
from rlm.domain.models import ChatCompletion, LLMRequest, UsageSummary
from rlm.domain.types import Prompt


def _require_anthropic() -> Any:
    """
    Lazily import the Anthropic SDK.

    Installed via the optional extra: `agentic-codebase-navigator[llm-anthropic]`.
    """

    try:
        import anthropic  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001 - dependency boundary
        raise ImportError(
            "Anthropic adapter selected but the 'anthropic' package is not installed. "
            "Install the optional extra: `agentic-codebase-navigator[llm-anthropic]`."
        ) from e
    return anthropic


def _messages_and_system(prompt: Prompt, /) -> tuple[list[dict[str, Any]], str | None]:
    """
    Convert a Prompt into Anthropic `messages` and optional `system`.

    Anthropic uses a dedicated `system` parameter; OpenAI-style "system" messages
    are stripped and mapped into that field.
    """

    messages = prompt_to_messages(prompt)
    system: str | None = None

    if messages and messages[0].get("role") == "system":
        system = str(messages[0].get("content", "") or "")
        messages = messages[1:]

    clean: list[dict[str, Any]] = []
    for m in messages:
        role = str(m.get("role", "user"))
        if role == "system":
            continue
        clean.append({"role": role, "content": str(m.get("content", "") or "")})

    return (clean, system)


def _extract_text(response: Any, /) -> str:
    try:
        content = response.content
    except Exception:
        content = response.get("content") if isinstance(response, dict) else None

    if isinstance(content, list) and content:
        first = content[0]
        text = getattr(first, "text", None) if first is not None else None
        if text is None and isinstance(first, dict):
            text = first.get("text")
        if text is not None:
            return str(text)

    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    raise ValueError("Anthropic response missing content")


def _extract_usage_tokens(response: Any, /) -> tuple[int, int]:
    usage: Any | None
    try:
        usage = response.usage
    except Exception:
        usage = response.get("usage") if isinstance(response, dict) else None

    if usage is None:
        return (0, 0)

    def _int(value: Any | None) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except Exception:
            return 0

    if isinstance(usage, dict):
        return (_int(usage.get("input_tokens")), _int(usage.get("output_tokens")))
    return (
        _int(getattr(usage, "input_tokens", None)),
        _int(getattr(usage, "output_tokens", None)),
    )


@dataclass
class AnthropicAdapter(BaseLLMAdapter):
    """Adapter skeleton: Anthropic SDK -> domain `LLMPort`."""

    model: str
    api_key: str | None = None
    default_request_kwargs: dict[str, Any] = field(default_factory=dict)

    _client_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _client: Any | None = field(default=None, init=False, repr=False)
    _async_client: Any | None = field(default=None, init=False, repr=False)
    _usage_tracker: UsageTracker = field(default_factory=UsageTracker, init=False, repr=False)

    @property
    def model_name(self) -> str:
        return self.model

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        anthropic = _require_anthropic()
        client = self._get_client(anthropic)

        model = request.model or self.model
        messages, system = _messages_and_system(request.prompt)

        kwargs = dict(self.default_request_kwargs)
        max_tokens = int(kwargs.pop("max_tokens", 1024))

        start = time.perf_counter()
        try:
            if system is not None:
                resp = client.messages.create(
                    model=model,
                    messages=messages,
                    system=system,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            else:
                resp = client.messages.create(
                    model=model, messages=messages, max_tokens=max_tokens, **kwargs
                )
        except Exception as e:  # noqa: BLE001 - provider boundary
            raise LLMError(safe_provider_error_message("Anthropic", e)) from None
        end = time.perf_counter()

        text = _extract_text(resp)
        in_tokens, out_tokens = _extract_usage_tokens(resp)
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
        anthropic = _require_anthropic()

        # Prefer the provider's async client if available; fall back to a thread to
        # avoid blocking the event loop if the SDK surface differs.
        try:
            client = self._get_async_client(anthropic)
        except Exception:  # noqa: BLE001
            return await asyncio.to_thread(self.complete, request)

        model = request.model or self.model
        messages, system = _messages_and_system(request.prompt)

        kwargs = dict(self.default_request_kwargs)
        max_tokens = int(kwargs.pop("max_tokens", 1024))

        start = time.perf_counter()
        try:
            if system is not None:
                resp = await client.messages.create(
                    model=model,
                    messages=messages,
                    system=system,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            else:
                resp = await client.messages.create(
                    model=model, messages=messages, max_tokens=max_tokens, **kwargs
                )
        except Exception as e:  # noqa: BLE001 - provider boundary
            raise LLMError(safe_provider_error_message("Anthropic", e)) from None
        end = time.perf_counter()

        text = _extract_text(resp)
        in_tokens, out_tokens = _extract_usage_tokens(resp)
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

    def _get_client(self, anthropic: Any, /) -> Any:
        with self._client_lock:
            if self._client is not None:
                return self._client

            client_cls = getattr(anthropic, "Anthropic", None)
            if client_cls is None:
                raise ImportError(
                    "Anthropic SDK API mismatch: expected `anthropic.Anthropic` class. "
                    "Please upgrade `anthropic` (install `agentic-codebase-navigator[llm-anthropic]`)."
                )

            kwargs: dict[str, Any] = {}
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key

            self._client = client_cls(**kwargs)
            return self._client

    def _get_async_client(self, anthropic: Any, /) -> Any:
        with self._client_lock:
            if self._async_client is not None:
                return self._async_client

            client_cls = getattr(anthropic, "AsyncAnthropic", None)
            if client_cls is None:
                raise ImportError(
                    "Anthropic SDK API mismatch: expected `anthropic.AsyncAnthropic` class. "
                    "Please upgrade `anthropic` (install `agentic-codebase-navigator[llm-anthropic]`)."
                )

            kwargs: dict[str, Any] = {}
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key

            self._async_client = client_cls(**kwargs)
            return self._async_client


def build_anthropic_adapter(
    *, model: str, api_key: str | None = None, **kwargs: Any
) -> AnthropicAdapter:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("AnthropicAdapter requires a non-empty 'model'")
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        raise ValueError("AnthropicAdapter.api_key must be a non-empty string when provided")
    return AnthropicAdapter(model=model, api_key=api_key, default_request_kwargs=dict(kwargs))
