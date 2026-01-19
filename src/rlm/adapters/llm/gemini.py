from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

from rlm.adapters.base import BaseLLMAdapter
from rlm.adapters.llm.provider_base import (
    UsageTracker,
    extract_finish_reason_gemini,
    extract_tool_calls_gemini,
    prompt_to_messages,
    safe_provider_error_message,
    tool_choice_to_gemini_function_calling_config,
    tool_definition_to_gemini_format,
)
from rlm.domain.errors import LLMError
from rlm.domain.models import ChatCompletion, LLMRequest, UsageSummary

if TYPE_CHECKING:
    from rlm.domain.types import Prompt


def _require_google_genai() -> Any:
    """
    Lazily import the Google GenAI (Gemini) SDK.

    Installed via the optional extra: `agentic-codebase-navigator[llm-gemini]`.
    """
    try:
        # `google-genai` exposes `google.genai`
        from google import genai  # type: ignore[import-not-found]
    except Exception as e:
        raise ImportError(
            "Gemini adapter selected but the 'google-genai' package is not installed. "
            "Install the optional extra: `agentic-codebase-navigator[llm-gemini]`.",
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
        getattr(usage, "prompt_token_count", None) or getattr(usage, "input_token_count", None),
    )
    out_tokens = _int(
        getattr(usage, "candidates_token_count", None)
        or getattr(usage, "output_token_count", None),
    )
    return (in_tokens, out_tokens)


def _normalize_openai_tool_call(tc: dict[str, Any], /) -> tuple[str, dict[str, Any], str]:
    tc_id = str(tc.get("id", "") or "")
    if "function" in tc:
        function = tc.get("function", {}) or {}
        name = str(function.get("name", "") or "")
        raw_args = function.get("arguments", {})
    else:
        name = str(tc.get("name", "") or "")
        raw_args = tc.get("arguments", {})
    args = raw_args
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {"value": args}
    return name, args, tc_id


def _parse_tool_response_content(content: Any, /) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except Exception:
            return {"content": content}
    return content


def _prompt_to_gemini_contents(prompt: Prompt, /) -> list[dict[str, Any]] | str:
    if isinstance(prompt, str):
        return prompt

    messages = prompt_to_messages(prompt)
    contents: list[dict[str, Any]] = []
    call_id_to_name: dict[str, str] = {}

    for message in messages:
        role = str(message.get("role", "user"))

        if role == "tool":
            tool_call_id = str(message.get("tool_call_id", "") or "")
            tool_name = call_id_to_name.get(tool_call_id, tool_call_id)
            response_payload = _parse_tool_response_content(message.get("content"))
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {"function_response": {"name": tool_name, "response": response_payload}},
                    ],
                },
            )
            continue

        tool_calls = message.get("tool_calls")
        if role == "assistant" and tool_calls:
            parts: list[dict[str, Any]] = []
            content = message.get("content")
            if content:
                parts.append({"text": str(content)})
            for tc in tool_calls:
                name, args, tc_id = _normalize_openai_tool_call(tc)
                if tc_id:
                    call_id_to_name[tc_id] = name
                parts.append({"function_call": {"name": name, "args": args}})
            contents.append({"role": "model", "parts": parts})
            continue

        content = message.get("content", "")
        if role == "system":
            content = f"System: {content}"

        mapped_role = "model" if role == "assistant" else "user"
        contents.append({"role": mapped_role, "parts": [{"text": str(content)}]})

    return contents


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

    @property
    def tool_prompt_format(self) -> str:
        return "gemini"

    @property
    def supports_tools(self) -> bool:
        """Gemini adapter supports native function calling."""
        return True

    def count_prompt_tokens(self, request: LLMRequest, /) -> int | None:
        genai = _require_google_genai()
        client = self._get_client(genai)

        model = request.model or self.model
        contents = _prompt_to_gemini_contents(request.prompt)

        api_kwargs: dict[str, Any] = dict(self.default_request_kwargs)
        if request.tools:
            function_declarations = [tool_definition_to_gemini_format(t) for t in request.tools]
            api_kwargs["tools"] = [{"function_declarations": function_declarations}]

        try:
            resp = client.models.count_tokens(model=model, contents=contents, **api_kwargs)
        except Exception:
            try:
                resp = client.models.count_tokens(model=model, contents=contents)
            except Exception:
                return None

        total_tokens = None
        try:
            total_tokens = resp.total_tokens
        except Exception:
            if isinstance(resp, dict):
                total_tokens = resp.get("total_tokens") or resp.get("totalTokens")
        if total_tokens is None:
            return None
        try:
            return int(total_tokens)
        except Exception:
            return None

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        genai = _require_google_genai()
        client = self._get_client(genai)

        model = request.model or self.model
        contents = _prompt_to_gemini_contents(request.prompt)

        # Build kwargs with tools if provided
        api_kwargs: dict[str, Any] = dict(self.default_request_kwargs)
        if request.tools:
            # Gemini expects tools wrapped in a Tool object with function_declarations
            function_declarations = [tool_definition_to_gemini_format(t) for t in request.tools]
            api_kwargs["tools"] = [{"function_declarations": function_declarations}]
        if request.tool_choice is not None:
            function_calling_config = tool_choice_to_gemini_function_calling_config(
                request.tool_choice,
            )
            if function_calling_config is not None:
                tool_config = api_kwargs.get("tool_config")
                if not isinstance(tool_config, dict):
                    tool_config = {}
                tool_config["function_calling_config"] = function_calling_config
                api_kwargs["tool_config"] = tool_config

        start = time.perf_counter()
        try:
            resp = client.models.generate_content(model=model, contents=contents, **api_kwargs)
        except Exception as e:
            raise LLMError(safe_provider_error_message("Gemini", e)) from None
        end = time.perf_counter()

        # Extract tool calls (may be None if no tools called) - unwrap() raises LLMError on malformed
        tool_calls = extract_tool_calls_gemini(resp).unwrap()
        finish_reason = extract_finish_reason_gemini(resp)

        # Extract text response (may be empty if tool_calls present)
        try:
            text = _extract_text(resp)
        except ValueError:
            # Response may have no text content when tool_calls are present
            if tool_calls:
                text = ""
            else:
                raise

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
            tool_calls=tool_calls,
            finish_reason=finish_reason,
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
                    "Please upgrade `google-genai` (install `agentic-codebase-navigator[llm-gemini]`).",
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
