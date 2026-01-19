from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from itertools import count
from threading import Lock
from typing import TYPE_CHECKING, Any

from rlm.domain.errors import LLMError
from rlm.domain.models import ModelUsageSummary, UsageSummary
from rlm.domain.models.result import Err, Ok, Result
from rlm.domain.models.safe_accessor import SafeAccessor

if TYPE_CHECKING:
    from rlm.domain.agent_ports import ToolCallRequest, ToolDefinition
    from rlm.domain.models.llm_request import ToolChoice
    from rlm.domain.types import Prompt


# =============================================================================
# Pydantic TypeAdapter Integration (ADR-001 - Adapter Layer)
# =============================================================================
# Uses Pydantic BaseModel + TypeAdapter for SDK response validation when available.
# Falls back to SafeAccessor when Pydantic not installed. This is an adapter-layer
# concern - domain remains Pydantic-free per hexagonal discipline.
#
# Lazy class definition preserves optional dependency (pip install rlm[pydantic]).

_PYDANTIC_MODELS_LOADED: bool = False
_OpenAIFunctionModel: type | None = None
_OpenAIToolCallModel: type | None = None
_OpenAIToolCallsAdapter: object | None = None


def _load_pydantic_models() -> bool:
    """
    Lazily load Pydantic BaseModel classes for SDK response validation.

    Returns True if Pydantic is available and models were loaded.
    Models are defined inside this function to preserve optional dependency.
    """
    global _PYDANTIC_MODELS_LOADED, _OpenAIFunctionModel, _OpenAIToolCallModel  # noqa: PLW0603
    global _OpenAIToolCallsAdapter  # noqa: PLW0603

    if _PYDANTIC_MODELS_LOADED:
        return _OpenAIToolCallModel is not None

    _PYDANTIC_MODELS_LOADED = True

    try:
        from pydantic import BaseModel, TypeAdapter

        class OpenAIFunction(BaseModel):
            """OpenAI function call shape within tool_calls."""

            name: str
            arguments: str  # JSON string to be parsed

        class OpenAIToolCall(BaseModel):
            """OpenAI tool_call response shape."""

            id: str
            type: str = "function"
            function: OpenAIFunction

        _OpenAIFunctionModel = OpenAIFunction
        _OpenAIToolCallModel = OpenAIToolCall
        _OpenAIToolCallsAdapter = TypeAdapter(list[OpenAIToolCall])

    except ImportError:
        return False

    else:
        return True


def _validate_openai_tool_calls(
    raw_tool_calls: list[object],
) -> Result[list[ToolCallRequest], LLMError] | None:
    """
    Validate OpenAI tool calls using Pydantic TypeAdapter if available.

    Returns:
        Ok(list) - Pydantic validation succeeded, returns domain ToolCallRequest list
        Err - Pydantic validation failed with clear error
        None - Pydantic not available, caller should use SafeAccessor fallback

    """
    if not _load_pydantic_models() or _OpenAIToolCallsAdapter is None:
        return None  # Signal: use SafeAccessor fallback

    try:
        # TypeAdapter.validate_python() - runtime validation with coercion
        # pyright: ignore - TypeAdapter is dynamically imported
        validated = _OpenAIToolCallsAdapter.validate_python(raw_tool_calls)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue, reportAny]

        result: list[ToolCallRequest] = []
        for tc in validated:  # pyright: ignore[reportAny]
            # Parse arguments JSON
            args_str: str = tc.function.arguments  # pyright: ignore[reportAny]
            arguments: dict[str, object] = {}
            if args_str:
                try:
                    parsed = json.loads(args_str)
                except json.JSONDecodeError as e:
                    return Err(LLMError(f"Invalid JSON in tool call arguments: {e}"))
                if not isinstance(parsed, dict):
                    return Err(
                        LLMError(f"Tool call arguments must be dict, got {type(parsed).__name__}")
                    )
                arguments = parsed

            result.append(
                {
                    "id": tc.id,  # pyright: ignore[reportAny]
                    "name": tc.function.name,  # pyright: ignore[reportAny]
                    "arguments": arguments,
                }
            )

        return Ok(result)

    except Exception as e:
        return Err(LLMError(f"Tool call validation failed: {e}"))


def _extract_tool_calls_manual(
    tool_calls_raw: list[object],
) -> Result[list[ToolCallRequest] | None, LLMError]:
    """
    Extract OpenAI tool calls using SafeAccessor (fallback when Pydantic unavailable).

    This is the manual parsing path that works without Pydantic installed.

    """
    result: list[ToolCallRequest] = []
    for tc in tool_calls_raw:
        tc_acc = SafeAccessor(tc)
        tc_id = tc_acc.get_str_or("id", "")

        func = tc_acc.get("function")
        if func is None:
            continue

        func_acc = SafeAccessor(func)
        name = func_acc.get_str_or("name", "")
        args_str = func_acc.get_str_or("arguments", "")

        # Parse arguments JSON with Result pattern
        arguments: dict[str, object] = {}
        if args_str:
            try:
                parsed = json.loads(args_str)
            except json.JSONDecodeError as e:
                return Err(LLMError(f"Invalid JSON in tool call arguments: {e}"))
            if not isinstance(parsed, dict):
                return Err(
                    LLMError(f"Tool call arguments must be dict, got {type(parsed).__name__}")
                )
            arguments = parsed

        result.append({"id": tc_id, "name": name, "arguments": arguments})

    return Ok(result if result else None)


_GEMINI_CALL_COUNTER = count(1)
_GEMINI_CALL_LOCK = Lock()


def _next_gemini_call_id() -> str:
    with _GEMINI_CALL_LOCK:
        counter_value = next(_GEMINI_CALL_COUNTER)
    return f"gemini_call_{os.getpid()}_{counter_value}"


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


# =============================================================================
# Prompt Conversion Helpers (extracted for PLR0911 compliance)
# =============================================================================


def _list_to_messages(messages: list[object]) -> list[dict[str, Any]]:
    """Convert a list prompt to OpenAI-style messages."""
    if all(isinstance(m, dict) for m in messages):
        # Validated: all elements are dicts - create new dict copies
        return [dict(m.items()) for m in messages if isinstance(m, dict)]
    return [{"role": "user", "content": str(messages)}]


def _dict_to_messages(payload: dict[str, object]) -> list[dict[str, Any]]:
    """Convert a dict prompt to OpenAI-style messages using SafeAccessor."""
    accessor = SafeAccessor(payload)

    # Try to extract messages list
    msgs = accessor.get("messages")
    if isinstance(msgs, list) and all(isinstance(m, dict) for m in msgs):
        return [dict(m.items()) for m in msgs if isinstance(m, dict)]

    # Try prompt or content keys
    prompt_val = accessor.get("prompt")
    if isinstance(prompt_val, str) and prompt_val:
        return [{"role": "user", "content": prompt_val}]

    content_val = accessor.get("content")
    if isinstance(content_val, str) and content_val:
        return [{"role": "user", "content": content_val}]

    return [{"role": "user", "content": str(payload)}]


def prompt_to_messages(prompt: Prompt, /) -> list[dict[str, Any]]:
    """
    Convert a domain Prompt payload to an OpenAI-style chat messages list.

    Many provider SDKs accept this common `messages=[{role, content}, ...]` shape.
    """
    match prompt:
        case str():
            return [{"role": "user", "content": prompt}]
        case list():
            return _list_to_messages(list(prompt))
        case dict():
            return _dict_to_messages(prompt)
        case _:
            return [{"role": "user", "content": str(prompt)}]


def _list_to_text(messages: list[object]) -> str:
    """Convert a list prompt to plain text."""
    if all(isinstance(m, dict) for m in messages):
        parts: list[str] = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "")
                content = m.get("content", "")
                parts.append(f"{role}: {content}")
        return "\n".join(parts)
    return str(messages)


def _dict_to_text(payload: dict[str, object]) -> str:
    """Convert a dict prompt to plain text using SafeAccessor."""
    accessor = SafeAccessor(payload)

    # Try prompt or content keys first
    prompt_val = accessor.get("prompt")
    if isinstance(prompt_val, str) and prompt_val:
        return prompt_val

    content_val = accessor.get("content")
    if isinstance(content_val, str) and content_val:
        return content_val

    # Try messages key (recursive)
    msgs = accessor.get("messages")
    if isinstance(msgs, list):
        return _list_to_text(list(msgs))

    return str(payload)


def prompt_to_text(prompt: Prompt, /) -> str:
    """Best-effort prompt stringification for providers that accept plain text."""
    match prompt:
        case str():
            return prompt
        case list():
            return _list_to_text(list(prompt))
        case dict():
            return _dict_to_text(prompt)
        case _:
            return str(prompt)


def count_openai_prompt_tokens(
    prompt: Prompt,
    tools: list[ToolDefinition] | None,
    model: str,
    /,
) -> int | None:
    """
    Count tokens for OpenAI-style chat prompts using tiktoken if available.

    Returns None if tiktoken is not installed or counting fails.
    """
    try:
        import tiktoken  # Optional dependency (config: pyproject.toml [[tool.mypy.overrides]])
    except Exception:
        return None

    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("o200k_base")

    messages = prompt_to_messages(prompt)
    tokens_per_message = 3
    tokens_per_name = 1
    total = 0

    for message in messages:
        total += tokens_per_message
        for key, value in message.items():
            if value is None:
                continue
            if key == "tool_calls":
                encoded = json.dumps(value, ensure_ascii=True, default=str)
            else:
                encoded = str(value)
            total += len(encoding.encode(encoded))
            if key == "name":
                total += tokens_per_name

    total += 3

    if tools:
        openai_tools = [tool_definition_to_openai_format(t) for t in tools]
        total += len(encoding.encode(json.dumps(openai_tools, ensure_ascii=True, default=str)))

    return total


def extract_text_from_chat_response(response: Any, /) -> str:
    """
    Extract a response string from an OpenAI-style chat completion payload.

    Supports both object-style (SDK models) and dict-style payloads.
    """
    if isinstance(response, str):
        return response

    accessor = SafeAccessor(response)
    choices = accessor.get("choices")
    if not choices or not isinstance(choices, list):
        raise ValueError("Provider response missing choices")

    first_acc = SafeAccessor(choices[0])

    # Try message.content path (standard chat completion format)
    message = first_acc.get("message")
    if message is not None:
        msg_acc = SafeAccessor(message)
        content = msg_acc.get("content")
        if content is not None:
            return str(content)

    # Try text path (some providers use this)
    text = first_acc.get("text")
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


def tool_choice_to_anthropic_format(tool_choice: ToolChoice, /) -> dict[str, Any] | None:
    """
    Convert a ToolChoice to Anthropic's tool_choice format.

    - "auto" → {"type": "auto"}
    - "required" → {"type": "any"}
    - "none" → {"type": "none"}
    - specific tool name → {"type": "tool", "name": "..."}
    """
    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return {"type": "auto"}
    if tool_choice == "required":
        return {"type": "any"}
    if tool_choice == "none":
        return {"type": "none"}
    return {"type": "tool", "name": tool_choice}


def tool_choice_to_gemini_function_calling_config(
    tool_choice: ToolChoice,
    /,
) -> dict[str, Any] | None:
    """
    Convert a ToolChoice to Gemini's function_calling_config shape.

    - "auto" → {"mode": "AUTO"}
    - "required" → {"mode": "ANY"}
    - "none" → {"mode": "NONE"}
    - specific tool name → {"mode": "ANY", "allowed_function_names": ["..."]}
    """
    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return {"mode": "AUTO"}
    if tool_choice == "required":
        return {"mode": "ANY"}
    if tool_choice == "none":
        return {"mode": "NONE"}
    return {"mode": "ANY", "allowed_function_names": [tool_choice]}


def extract_tool_calls_openai(response: Any, /) -> Result[list[ToolCallRequest] | None, LLMError]:
    """
    Extract tool calls from an OpenAI-style chat completion response.

    OpenAI returns tool calls in:
    response.choices[0].message.tool_calls[].{id, function.name, function.arguments}

    Returns:
        Ok(list) - tool calls found
        Ok(None) - no tool calls present (valid response, model didn't call tools)
        Err(LLMError) - malformed response (missing required structural fields)

    Strategy (ADR-001):
        1. Use SafeAccessor to navigate to tool_calls (works with SDK objects or dicts)
        2. Try Pydantic TypeAdapter validation when available (better errors, type coercion)
        3. Fall back to SafeAccessor manual parsing when Pydantic unavailable

    Note:
        Missing `choices` or `message` indicates a malformed response and returns Err.
        Missing `tool_calls` is valid (model responded with text only) and returns Ok(None).

    """
    accessor = SafeAccessor(response)

    choices = accessor.get("choices")
    if not choices or not isinstance(choices, list):
        return Err(LLMError("Provider response missing choices"))

    first_acc = SafeAccessor(choices[0])
    message = first_acc.get("message")
    if message is None:
        return Err(LLMError("Provider response missing message"))

    msg_acc = SafeAccessor(message)
    tool_calls_raw = msg_acc.get("tool_calls")
    if not tool_calls_raw or not isinstance(tool_calls_raw, list):
        return Ok(None)

    # Try Pydantic TypeAdapter validation (ADR-001: better errors when available)
    pydantic_result = _validate_openai_tool_calls(list(tool_calls_raw))
    if pydantic_result is not None:
        # Pydantic handled it - convert empty list to None per API contract
        if isinstance(pydantic_result, Err):
            return pydantic_result
        tool_calls = pydantic_result.unwrap()
        return Ok(tool_calls if tool_calls else None)

    # Fallback: SafeAccessor manual parsing (no Pydantic)
    return _extract_tool_calls_manual(list(tool_calls_raw))


def extract_tool_calls_anthropic(response: Any, /) -> list[ToolCallRequest] | None:
    """
    Extract tool calls from an Anthropic response.

    Anthropic returns tool use in content blocks:
    response.content[].{type: "tool_use", id, name, input}

    Returns None if no tool calls are present.
    """
    accessor = SafeAccessor(response)
    content = accessor.get("content")
    if not content or not isinstance(content, list):
        return None

    result: list[ToolCallRequest] = []
    for block in content:
        block_acc = SafeAccessor(block)
        block_type = block_acc.get_str_or("type", "")

        if block_type != "tool_use":
            continue

        tc_id = block_acc.get_str_or("id", "")
        name = block_acc.get_str_or("name", "")
        arguments = block_acc.get("input")
        if not isinstance(arguments, dict):
            arguments = {}

        result.append({"id": tc_id, "name": name, "arguments": arguments})

    return result if result else None


def extract_tool_calls_gemini(response: Any, /) -> Result[list[ToolCallRequest] | None, LLMError]:
    """
    Extract tool calls from a Google Gemini response.

    Gemini returns function calls in:
    response.candidates[0].content.parts[].function_call.{name, args}

    Returns:
        Ok(list) - tool calls found
        Ok(None) - no tool calls present (valid response, model didn't call tools)
        Err(LLMError) - malformed response (missing required structural fields)

    Note:
        Uses SafeAccessor for unified SDK/dict access pattern.
        Missing `candidates` or `content` indicates a malformed response and returns Err.
        Missing `parts` or parts without `function_call` is valid and returns Ok(None).

    """
    accessor = SafeAccessor(response)

    candidates = accessor.get("candidates")
    if not candidates or not isinstance(candidates, list):
        return Err(LLMError("Provider response missing candidates"))

    first_acc = SafeAccessor(candidates[0])
    content = first_acc.get("content")
    if content is None:
        return Err(LLMError("Provider response missing content"))

    content_acc = SafeAccessor(content)
    parts = content_acc.get("parts")
    if not parts or not isinstance(parts, list):
        return Ok(None)

    result: list[ToolCallRequest] = []
    for part in parts:
        part_acc = SafeAccessor(part)

        # Try both naming conventions (function_call and functionCall)
        function_call = part_acc.get("function_call") or part_acc.get("functionCall")
        if function_call is None:
            continue

        fc_acc = SafeAccessor(function_call)
        name = fc_acc.get_str_or("name", "")

        # Gemini args: None/missing = no args, non-dict = error
        args_raw = fc_acc.get("args")
        if args_raw is None:
            arguments: dict[str, object] = {}
        elif isinstance(args_raw, dict):
            arguments = args_raw
        else:
            return Err(
                LLMError(f"Gemini function call args must be dict, got {type(args_raw).__name__}")
            )

        # Gemini doesn't provide IDs, so we generate process-unique ones
        tc_id = _next_gemini_call_id()

        result.append({"id": tc_id, "name": name, "arguments": arguments})

    return Ok(result if result else None)


def extract_finish_reason_openai(response: Any, /) -> str | None:
    """
    Extract finish_reason from an OpenAI-style response.

    Returns: "stop", "tool_calls", "length", etc. or None if not available.
    """
    accessor = SafeAccessor(response)
    choices = accessor.get("choices")
    if not choices or not isinstance(choices, list):
        return None

    first_acc = SafeAccessor(choices[0])
    finish_reason = first_acc.get("finish_reason")
    return str(finish_reason) if finish_reason is not None else None


def extract_finish_reason_anthropic(response: Any, /) -> str | None:
    """
    Extract stop_reason from an Anthropic response and normalize to OpenAI-style.

    Anthropic uses "end_turn", "tool_use", "max_tokens" etc.
    We normalize to "stop", "tool_calls", "length" for consistency.
    """
    accessor = SafeAccessor(response)
    stop_reason = accessor.get("stop_reason")
    if stop_reason is None:
        return None

    stop_reason_str = str(stop_reason)

    # Normalize Anthropic's stop reasons to OpenAI-style
    mapping = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
    }
    return mapping.get(stop_reason_str, stop_reason_str)


def extract_finish_reason_gemini(response: Any, /) -> str | None:
    """
    Extract finish_reason from a Gemini response and normalize to OpenAI-style.

    Gemini uses STOP, MAX_TOKENS, SAFETY, etc.
    We normalize to "stop", "length", etc. for consistency.
    """
    accessor = SafeAccessor(response)
    candidates = accessor.get("candidates")
    if not candidates or not isinstance(candidates, list):
        return None

    first_acc = SafeAccessor(candidates[0])

    # Try both naming conventions (finish_reason and finishReason)
    finish_reason = first_acc.get("finish_reason") or first_acc.get("finishReason")
    if finish_reason is None:
        return None

    # Handle enum values (Gemini SDK returns enums)
    finish_reason_name = getattr(finish_reason, "name", None)
    if finish_reason_name is not None:
        finish_reason = finish_reason_name

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
    accessor = SafeAccessor(response)
    usage = accessor.get("usage")
    if usage is None:
        return (0, 0)

    def _coerce_int(value: object) -> int:
        """Coerce value to int, returning 0 on failure."""
        if value is None:
            return 0
        try:
            return int(value)  # type: ignore[arg-type,call-overload]
        except (ValueError, TypeError):
            return 0

    usage_acc = SafeAccessor(usage)

    # Try both naming conventions (classic and newer)
    in_tokens = _coerce_int(usage_acc.get("prompt_tokens")) or _coerce_int(
        usage_acc.get("input_tokens")
    )
    out_tokens = _coerce_int(usage_acc.get("completion_tokens")) or _coerce_int(
        usage_acc.get("output_tokens")
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
            },
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
            },
        )
