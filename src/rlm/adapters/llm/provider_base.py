from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from rlm.domain.models import ModelUsageSummary, UsageSummary
from rlm.domain.types import Prompt


def safe_provider_error_message(provider: str, exc: BaseException, /) -> str:
    """
    Convert provider exceptions into safe, user-facing messages.

    This intentionally avoids leaking stack traces or provider response bodies.
    """

    if isinstance(exc, TimeoutError):
        return f"{provider} request timed out"
    if isinstance(exc, (ConnectionError, OSError)):
        return f"{provider} connection error"
    return f"{provider} request failed"


def prompt_to_messages(prompt: Prompt, /) -> list[dict[str, Any]]:
    """
    Convert a domain Prompt payload to an OpenAI-style chat messages list.

    Many provider SDKs accept this common `messages=[{role, content}, ...]` shape.
    """

    match prompt:
        case str():
            return [{"role": "user", "content": prompt}]
        case list() as messages:
            if all(isinstance(m, dict) for m in messages):
                return list(messages)  # type: ignore[return-value]
            return [{"role": "user", "content": str(prompt)}]
        case dict() as payload:
            if "messages" in payload and isinstance(payload.get("messages"), list):
                msgs = payload.get("messages")
                if isinstance(msgs, list) and all(isinstance(m, dict) for m in msgs):
                    return list(msgs)  # type: ignore[return-value]
            if "prompt" in payload:
                return [{"role": "user", "content": str(payload.get("prompt"))}]
            if "content" in payload:
                return [{"role": "user", "content": str(payload.get("content"))}]
            return [{"role": "user", "content": str(payload)}]
        case _:
            return [{"role": "user", "content": str(prompt)}]


def prompt_to_text(prompt: Prompt, /) -> str:
    """Best-effort prompt stringification for providers that accept plain text."""

    match prompt:
        case str():
            return prompt
        case list() as messages:
            if all(isinstance(m, dict) for m in messages):
                parts: list[str] = []
                for m in messages:
                    role = m.get("role", "")
                    content = m.get("content", "")
                    parts.append(f"{role}: {content}")
                return "\n".join(parts)
            return str(prompt)
        case dict() as payload:
            if "prompt" in payload:
                return str(payload.get("prompt"))
            if "content" in payload:
                return str(payload.get("content"))
            if "messages" in payload and isinstance(payload.get("messages"), list):
                return prompt_to_text(payload.get("messages"))  # type: ignore[arg-type]
            return str(payload)
        case _:
            return str(prompt)


def extract_text_from_chat_response(response: Any, /) -> str:
    """
    Extract a response string from an OpenAI-style chat completion payload.

    Supports both object-style (SDK models) and dict-style payloads.
    """

    if isinstance(response, str):
        return response

    try:
        choices = response.choices  # SDK model
    except Exception:
        choices = None
    if choices is None:
        choices = response.get("choices") if isinstance(response, dict) else None
    if not choices:
        raise ValueError("Provider response missing choices")

    first = choices[0]
    message = None
    try:
        message = first.message
    except Exception:
        if isinstance(first, dict):
            message = first.get("message")
    if message is not None:
        try:
            content = message.content
        except Exception:
            content = message.get("content") if isinstance(message, dict) else None
        if content is not None:
            return str(content)

    try:
        text = first.text  # type: ignore[union-attr]
    except Exception:
        text = first.get("text") if isinstance(first, dict) else None
    if text is not None:
        return str(text)

    raise ValueError("Provider response missing message content")


def extract_openai_style_token_usage(response: Any, /) -> tuple[int, int]:
    """
    Best-effort token extraction from `response.usage`.

    Supports both the classic (prompt_tokens/completion_tokens) and newer
    (input_tokens/output_tokens) key names.
    """

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
        in_tokens = _int(usage.get("prompt_tokens") or usage.get("input_tokens"))
        out_tokens = _int(usage.get("completion_tokens") or usage.get("output_tokens"))
        return (in_tokens, out_tokens)

    in_tokens = _int(getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None))
    out_tokens = _int(
        getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
    )
    return (in_tokens, out_tokens)


@dataclass
class UsageTracker:
    """
    Shared usage accounting helper for provider adapters.

    - Tracks totals per model
    - Tracks last-call usage as a single-entry summary (legacy-compatible)
    """

    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _total: dict[str, ModelUsageSummary] = field(default_factory=dict, init=False, repr=False)
    _last: dict[str, ModelUsageSummary] = field(default_factory=dict, init=False, repr=False)

    def record(
        self,
        model: str,
        /,
        *,
        calls: int = 1,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> ModelUsageSummary:
        last = ModelUsageSummary(
            total_calls=calls,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
        )
        with self._lock:
            total = self._total.get(model)
            if total is None:
                total = ModelUsageSummary()
                self._total[model] = total
            total.total_calls += calls
            total.total_input_tokens += input_tokens
            total.total_output_tokens += output_tokens
            self._last = {model: last}
        return last

    def get_usage_summary(self) -> UsageSummary:
        with self._lock:
            # Snapshot values (copy the *numbers*) while holding the lock.
            items = [
                (
                    model,
                    mus.total_calls,
                    mus.total_input_tokens,
                    mus.total_output_tokens,
                )
                for model, mus in self._total.items()
            ]
        # Return deep-copied ModelUsageSummary objects so callers can't observe
        # future `record()` mutations (or mutate our internal state via aliasing).
        return UsageSummary(
            model_usage_summaries={
                model: ModelUsageSummary(
                    total_calls=calls,
                    total_input_tokens=input_tokens,
                    total_output_tokens=output_tokens,
                )
                for model, calls, input_tokens, output_tokens in items
            }
        )

    def get_last_usage(self) -> UsageSummary:
        with self._lock:
            items = [
                (
                    model,
                    mus.total_calls,
                    mus.total_input_tokens,
                    mus.total_output_tokens,
                )
                for model, mus in self._last.items()
            ]
        return UsageSummary(
            model_usage_summaries={
                model: ModelUsageSummary(
                    total_calls=calls,
                    total_input_tokens=input_tokens,
                    total_output_tokens=output_tokens,
                )
                for model, calls, input_tokens, output_tokens in items
            }
        )
