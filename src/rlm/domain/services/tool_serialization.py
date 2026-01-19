"""
Tool result serialization utilities.

This module provides JSON serialization for tool execution results.
It handles common Python types that aren't natively JSON-serializable.

Uses the type-driven boundary pattern:
- Input: `object` (accepts any tool result)
- Output: `SerializedValue` (strict JSON-compatible union)

Shared by:
- rlm_orchestrator.py (original tool calling loop)
- tools_mode_event_source.py (StateMachine-based tool calling)
"""

from __future__ import annotations

import base64
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, cast

from rlm.domain.models.serialization import SerializedValue
from rlm.domain.models.type_mapping import TypeMapper

if TYPE_CHECKING:
    from collections.abc import Callable

# ============================================================================
# TypeMapper handlers - each returns SerializedValue
# ============================================================================


def _serialize_enum(v: object) -> SerializedValue:
    """Enums serialize to their value."""
    # Enum.value is typed as Any in stdlib (PEP 435 limitation) - we narrow immediately
    enum_value: object = cast("Enum", v).value  # pyright: ignore[reportAny] - stdlib Enum.value is Any
    # Explicit narrowing to JSON primitives
    if enum_value is None or isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    return str(enum_value)


def _serialize_set(v: object) -> SerializedValue:
    """Sets serialize to lists."""
    return [_coerce_to_serialized(item) for item in cast("set[object]", v)]


def _serialize_datetime(v: object) -> SerializedValue:
    """Datetimes serialize to ISO format strings."""
    return cast("datetime", v).isoformat()


def _serialize_date(v: object) -> SerializedValue:
    """Dates serialize to ISO format strings."""
    return cast("date", v).isoformat()


def _serialize_decimal(v: object) -> SerializedValue:
    """Decimals serialize to strings (preserves precision)."""
    return str(cast("Decimal", v))


def _serialize_bytes(v: object) -> SerializedValue:
    """Bytes serialize to base64 dict."""
    return {"__bytes__": base64.b64encode(cast("bytes", v)).decode("ascii")}


def _serialize_bytearray(v: object) -> SerializedValue:
    """Bytearrays serialize to base64 dict."""
    return {"__bytes__": base64.b64encode(bytes(cast("bytearray", v))).decode("ascii")}


def _serialize_memoryview(v: object) -> SerializedValue:
    """Memoryviews serialize to base64 dict."""
    return {"__bytes__": base64.b64encode(bytes(cast("memoryview", v))).decode("ascii")}


def _coerce_to_serialized(value: object) -> SerializedValue:
    """Coerce a value to SerializedValue (for recursive calls)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_to_serialized(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce_to_serialized(v) for k, v in value.items()}
    return str(value)


# ============================================================================
# TypeMapper: Declarative type dispatch
# ============================================================================

tool_json_mapper: TypeMapper[object, SerializedValue] = (
    TypeMapper[object, SerializedValue]()
    .register(Enum, _serialize_enum)
    .register(set, _serialize_set)
    .register(datetime, _serialize_datetime)
    .register(date, _serialize_date)
    .register(Decimal, _serialize_decimal)
    .register(bytes, _serialize_bytes)
    .register(bytearray, _serialize_bytearray)
    .register(memoryview, _serialize_memoryview)
)


# ============================================================================
# Public API
# ============================================================================


def tool_json_default(value: object, /) -> SerializedValue:
    """
    Coerce tool results into JSON-friendly structures or raise TypeError.

    Type-driven boundary pattern:
    - Input: `object` (accepts any tool result at boundary)
    - Output: `SerializedValue` (strict JSON-compatible union)

    Dispatch strategy:
    1. Dataclasses → asdict() (must check first as they're not registered types)
    2. Known types → TypeMapper dispatch (Enum, set, datetime, Decimal, bytes, etc.)
    3. Duck-typed serialization → model_dump/to_dict/dict methods (Pydantic, attrs, custom)
    4. Fallback → raise TypeError

    Usage:
        json.dumps(tool_result, default=tool_json_default)
    """
    # Dataclass handling (must check first as dataclass instances can match other types)
    if is_dataclass(value) and not isinstance(value, type):
        result = asdict(value)  # type: ignore[arg-type]
        return _coerce_to_serialized(result)

    # TypeMapper handles known JSON-incompatible types declaratively
    if tool_json_mapper.can_handle(value):
        return tool_json_mapper.map(value)

    # Duck-typed serialization methods (Pydantic v2, Pydantic v1, attrs, custom)
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            duck_result: object = cast("Callable[[], object]", method)()
            return _coerce_to_serialized(duck_result)

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
