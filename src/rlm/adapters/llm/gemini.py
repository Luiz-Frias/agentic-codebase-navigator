from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from rlm.adapters.base import BaseLLMAdapter
from rlm.adapters.llm.provider_base import UsageTracker, prompt_to_text, safe_provider_error_message
from rlm.domain.errors import LLMError
from rlm.domain.models import ChatCompletion, LLMRequest, UsageSummary


def _require_google_genai() -> Any:
    """
    Lazily import the Google GenAI (Gemini) SDK.

    Installed via the optional extra: `agentic-codebase-navigator[llm-gemini]`.
    """

    try:
        # `google-genai` exposes `google.genai`
        from google import genai  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001 - dependency boundary
        raise ImportError(
            "Gemini adapter selected but the 'google-genai' package is not installed. "
            "Install the optional extra: `agentic-codebase-navigator[llm-gemini]`."
        ) from e
    return genai


def _extract_text(response: Any, /) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    if isinstance(response, dict) and response.get("text"):
        return str(response.get("text"))

    candidates = getattr(response, "candidates", None)
    if candidates is None and isinstance(response, dict):
        candidates = response.get("candidates")
    if candidates:
        first = candidates[0]
        content = getattr(first, "content", None)
        if content is None and isinstance(first, dict):
            content = first.get("content")
        parts = getattr(content, "parts", None) if content is not None else None
        if parts is None and isinstance(content, dict):
            parts = content.get("parts")
        if parts:
            p0 = parts[0]
            t = getattr(p0, "text", None)
            if t is None and isinstance(p0, dict):
                t = p0.get("text")
            if t is not None:
                return str(t)

    raise ValueError("Gemini response missing text")


def _extract_usage_tokens(response: Any, /) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage_metadata")

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
        in_tokens = _int(usage.get("prompt_token_count") or usage.get("input_token_count"))
        out_tokens = _int(usage.get("candidates_token_count") or usage.get("output_token_count"))
        return (in_tokens, out_tokens)

    in_tokens = _int(
        getattr(usage, "prompt_token_count", None) or getattr(usage, "input_token_count", None)
    )
    out_tokens = _int(
        getattr(usage, "candidates_token_count", None) or getattr(usage, "output_token_count", None)
    )
    return (in_tokens, out_tokens)


@dataclass
class GeminiAdapter(BaseLLMAdapter):
    """Adapter skeleton: Google GenAI SDK -> domain `LLMPort`."""

    model: str
    api_key: str | None = None
    default_request_kwargs: dict[str, Any] = field(default_factory=dict)

    _client_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _client: Any | None = field(default=None, init=False, repr=False)
    _usage_tracker: UsageTracker = field(default_factory=UsageTracker, init=False, repr=False)

    @property
    def model_name(self) -> str:
        return self.model

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        genai = _require_google_genai()
        client = self._get_client(genai)

        model = request.model or self.model
        contents = prompt_to_text(request.prompt)

        start = time.perf_counter()
        try:
            resp = client.models.generate_content(
                model=model, contents=contents, **self.default_request_kwargs
            )
        except Exception as e:  # noqa: BLE001 - provider boundary
            raise LLMError(safe_provider_error_message("Gemini", e)) from None
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
        # The SDK surface for async varies; to keep default installs stable and
        # avoid blocking, run the sync path in a thread.
        return await asyncio.to_thread(self.complete, request)

    def get_usage_summary(self) -> UsageSummary:
        return self._usage_tracker.get_usage_summary()

    def get_last_usage(self) -> UsageSummary:
        return self._usage_tracker.get_last_usage()

    def _get_client(self, genai: Any, /) -> Any:
        with self._client_lock:
            if self._client is not None:
                return self._client

            client_cls = getattr(genai, "Client", None)
            if client_cls is None:
                raise ImportError(
                    "Gemini SDK API mismatch: expected `google.genai.Client` class. "
                    "Please upgrade `google-genai` (install `agentic-codebase-navigator[llm-gemini]`)."
                )

            kwargs: dict[str, Any] = {}
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key

            self._client = client_cls(**kwargs)
            return self._client


def build_gemini_adapter(*, model: str, api_key: str | None = None, **kwargs: Any) -> GeminiAdapter:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("GeminiAdapter requires a non-empty 'model'")
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        raise ValueError("GeminiAdapter.api_key must be a non-empty string when provided")
    return GeminiAdapter(model=model, api_key=api_key, default_request_kwargs=dict(kwargs))
