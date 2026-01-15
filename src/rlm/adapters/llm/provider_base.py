from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

from rlm.domain.models import ModelUsageSummary, UsageSummary
from rlm.domain.types import Prompt

if TYPE_CHECKING:
    from rlm.domain.agent_ports import ToolCallRequest, ToolDefinition
    from rlm.domain.models.llm_request import ToolChoice


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


# =============================================================================
# Tool Calling Format Converters (Phase 2)
# =============================================================================


def tool_definition_to_openai_format(tool: ToolDefinition, /) -> dict[str, Any]:
    """
    Convert a ToolDefinition to OpenAI's function calling format.

    OpenAI expects tools in the format:
    {
        "type": "function",
        "function": {
            "name": "...",
            "description": "...",
            "parameters": {...}  # JSON Schema
        }
    }
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        },
    }


def tool_definition_to_anthropic_format(tool: ToolDefinition, /) -> dict[str, Any]:
    """
    Convert a ToolDefinition to Anthropic's tool format.

    Anthropic expects tools in the format:
    {
        "name": "...",
        "description": "...",
        "input_schema": {...}  # JSON Schema
    }
    """
    return {
        "name": tool["name"],
        "description": tool["description"],
        "input_schema": tool["parameters"],
    }


def tool_definition_to_gemini_format(tool: ToolDefinition, /) -> dict[str, Any]:
    """
    Convert a ToolDefinition to Google Gemini's FunctionDeclaration format.

    Gemini expects tools wrapped in a Tool object with function_declarations:
    {
        "name": "...",
        "description": "...",
        "parameters": {...}  # OpenAPI-style schema
    }
    """
    return {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["parameters"],
    }


def tool_choice_to_openai_format(tool_choice: ToolChoice, /) -> dict[str, Any] | str | None:
    """
    Convert a ToolChoice to OpenAI's tool_choice format.

    - "auto" → "auto"
    - "required" → "required"
    - "none" → "none"
    - specific tool name → {"type": "function", "function": {"name": "..."}}
    """
    if tool_choice is None:
        return None
    if tool_choice in ("auto", "required", "none"):
        return tool_choice
    # Specific tool name
    return {"type": "function", "function": {"name": tool_choice}}


def extract_tool_calls_openai(response: Any, /) -> list[ToolCallRequest] | None:
    """
    Extract tool calls from an OpenAI-style chat completion response.

    OpenAI returns tool calls in:
    response.choices[0].message.tool_calls[].{id, function.name, function.arguments}

    Returns None if no tool calls are present.
    """
    try:
        choices = response.choices
    except Exception:
        choices = response.get("choices") if isinstance(response, dict) else None

    if not choices:
        return None

    first = choices[0]

    # Get message from choice
    message = None
    try:
        message = first.message
    except Exception:
        if isinstance(first, dict):
            message = first.get("message")

    if message is None:
        return None

    # Get tool_calls from message
    tool_calls_raw = None
    try:
        tool_calls_raw = message.tool_calls
    except Exception:
        if isinstance(message, dict):
            tool_calls_raw = message.get("tool_calls")

    if not tool_calls_raw:
        return None

    result: list[ToolCallRequest] = []
    for tc in tool_calls_raw:
        try:
            # SDK model style
            tc_id = tc.id
            func = tc.function
            name = func.name
            args_str = func.arguments
        except Exception:
            # Dict style
            if isinstance(tc, dict):
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                name = func.get("name", "") if isinstance(func, dict) else ""
                args_str = func.get("arguments", "{}") if isinstance(func, dict) else "{}"
            else:
                continue

        # Parse arguments JSON
        try:
            arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            arguments = {}

        result.append({"id": tc_id, "name": name, "arguments": arguments})

    return result if result else None


def extract_tool_calls_anthropic(response: Any, /) -> list[ToolCallRequest] | None:
    """
    Extract tool calls from an Anthropic response.

    Anthropic returns tool use in content blocks:
    response.content[].{type: "tool_use", id, name, input}

    Returns None if no tool calls are present.
    """
    content = None
    try:
        content = response.content
    except Exception:
        if isinstance(response, dict):
            content = response.get("content")

    if not content:
        return None

    result: list[ToolCallRequest] = []
    for block in content:
        block_type = None
        try:
            block_type = block.type
        except Exception:
            if isinstance(block, dict):
                block_type = block.get("type")

        if block_type != "tool_use":
            continue

        try:
            # SDK model style (duck-typed attribute access)
            tc_id = block.id  # type: ignore[union-attr]
            name = block.name  # type: ignore[union-attr]
            arguments = block.input  # type: ignore[union-attr]
        except Exception:
            # Dict style
            if isinstance(block, dict):
                tc_id = block.get("id", "")
                name = block.get("name", "")
                arguments = block.get("input", {})
            else:
                continue

        result.append({"id": tc_id, "name": name, "arguments": arguments})

    return result if result else None


def extract_tool_calls_gemini(response: Any, /) -> list[ToolCallRequest] | None:
    """
    Extract tool calls from a Google Gemini response.

    Gemini returns function calls in:
    response.candidates[0].content.parts[].function_call.{name, args}

    Returns None if no tool calls are present.
    """
    candidates = None
    try:
        candidates = response.candidates
    except Exception:
        if isinstance(response, dict):
            candidates = response.get("candidates")

    if not candidates:
        return None

    first = candidates[0]

    # Get content from candidate
    content = None
    try:
        content = first.content
    except Exception:
        if isinstance(first, dict):
            content = first.get("content")

    if content is None:
        return None

    # Get parts from content
    parts = None
    try:
        parts = content.parts
    except Exception:
        if isinstance(content, dict):
            parts = content.get("parts")

    if not parts:
        return None

    result: list[ToolCallRequest] = []
    call_counter = 0
    for part in parts:
        function_call = None
        try:
            function_call = part.function_call
        except Exception:
            if isinstance(part, dict):
                function_call = part.get("function_call") or part.get("functionCall")

        if function_call is None:
            continue

        try:
            # SDK model style
            name = function_call.name
            args = function_call.args
        except Exception:
            # Dict style
            if isinstance(function_call, dict):
                name = function_call.get("name", "")
                args = function_call.get("args", {})
            else:
                continue

        # Gemini doesn't provide IDs, so we generate them
        call_counter += 1
        tc_id = f"gemini_call_{call_counter}"

        result.append({"id": tc_id, "name": name, "arguments": args})

    return result if result else None


def extract_finish_reason_openai(response: Any, /) -> str | None:
    """
    Extract finish_reason from an OpenAI-style response.

    Returns: "stop", "tool_calls", "length", etc. or None if not available.
    """
    try:
        choices = response.choices
    except Exception:
        choices = response.get("choices") if isinstance(response, dict) else None

    if not choices:
        return None

    first = choices[0]

    try:
        return first.finish_reason
    except Exception:
        if isinstance(first, dict):
            return first.get("finish_reason")

    return None


def extract_finish_reason_anthropic(response: Any, /) -> str | None:
    """
    Extract stop_reason from an Anthropic response and normalize to OpenAI-style.

    Anthropic uses "end_turn", "tool_use", "max_tokens" etc.
    We normalize to "stop", "tool_calls", "length" for consistency.
    """
    stop_reason = None
    try:
        stop_reason = response.stop_reason
    except Exception:
        if isinstance(response, dict):
            stop_reason = response.get("stop_reason")

    if stop_reason is None:
        return None

    # Normalize Anthropic's stop reasons to OpenAI-style
    mapping = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
    }
    return mapping.get(stop_reason, stop_reason)


def extract_finish_reason_gemini(response: Any, /) -> str | None:
    """
    Extract finish_reason from a Gemini response and normalize to OpenAI-style.

    Gemini uses STOP, MAX_TOKENS, SAFETY, etc.
    We normalize to "stop", "length", etc. for consistency.
    """
    candidates = None
    try:
        candidates = response.candidates
    except Exception:
        if isinstance(response, dict):
            candidates = response.get("candidates")

    if not candidates:
        return None

    first = candidates[0]

    finish_reason = None
    try:
        finish_reason = first.finish_reason
    except Exception:
        if isinstance(first, dict):
            finish_reason = first.get("finish_reason") or first.get("finishReason")

    if finish_reason is None:
        return None

    # Handle enum values (Gemini SDK returns enums)
    if hasattr(finish_reason, "name"):
        finish_reason = finish_reason.name

    # Normalize Gemini's finish reasons to OpenAI-style
    mapping = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "OTHER": "stop",
    }
    return mapping.get(str(finish_reason), str(finish_reason).lower())


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
