"""
Generic value serialization for JSON compatibility.

This module converts Python values into JSON-serializable representations
using the type-driven boundary pattern:

1. Input: `object` (accepts anything at the boundary)
2. Dispatch: TypeMapper validates against known types
3. Output: `SerializedValue` (strict JSON-compatible union)

Unknown types get a deterministic string fallback (repr or type name).
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, cast

from rlm.domain.models.type_mapping import TypeMapper

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

# ============================================================================
# Domain Type: Strict JSON-serializable output
# ============================================================================

type JsonPrimitive = str | int | float | bool | None
type SerializedValue = JsonPrimitive | list[SerializedValue] | dict[str, SerializedValue]


# ============================================================================
# TypeMapper handlers - each returns SerializedValue
# ============================================================================


def _serialize_none(_value: object) -> SerializedValue:
    """None passes through."""
    return None


def _serialize_primitive(value: object) -> SerializedValue:
    """Primitives (str, int, float, bool) pass through."""
    return cast("JsonPrimitive", value)


def _serialize_module(value: object) -> SerializedValue:
    """Modules serialize to descriptive string."""
    mod = cast("ModuleType", value)
    return f"<module '{mod.__name__}'>"


def _serialize_sequence(value: object) -> SerializedValue:
    """Lists/tuples serialize recursively."""
    seq = cast("Sequence[object]", value)
    return [serialize_value(item) for item in seq]


def _serialize_mapping(value: object) -> SerializedValue:
    """Dicts serialize with string keys, recursive values."""
    mapping = cast("Mapping[object, object]", value)
    return {str(k): serialize_value(v) for k, v in mapping.items()}


def _serialize_callable(value: object) -> SerializedValue:
    """Callables serialize to descriptive string."""
    func = cast("Callable[..., object]", value)
    name = getattr(func, "__name__", repr(func))
    return f"<{type(func).__name__} '{name}'>"


def _serialize_fallback(value: object) -> SerializedValue:
    """
    Fallback: try repr(), else type name.

    The bare except catches repr() failures on exotic objects (e.g., objects
    that raise in __repr__). This is intentional - we never want serialization
    to fail, just degrade gracefully to type name.
    """
    try:
        return repr(value)
    except Exception:  # noqa: BLE001 - intentional fallback for repr() failures
        return f"<{type(value).__name__}>"


# ============================================================================
# TypeMapper: Declarative type dispatch
# ============================================================================

_serializer: TypeMapper[object, SerializedValue] = (
    TypeMapper[object, SerializedValue]()
    .register(type(None), _serialize_none)
    .register(bool, _serialize_primitive)  # Must come before int (bool is subclass of int)
    .register(int, _serialize_primitive)
    .register(float, _serialize_primitive)
    .register(str, _serialize_primitive)
    .register(ModuleType, _serialize_module)
    .register(list, _serialize_sequence)
    .register(tuple, _serialize_sequence)
    .register(dict, _serialize_mapping)
    .default(_serialize_fallback)
)


# ============================================================================
# Public API
# ============================================================================


def serialize_value(value: object) -> SerializedValue:
    """
    Convert a Python value into a JSON-serializable representation.

    Type-driven boundary pattern:
    - Input: `object` (accepts anything)
    - Output: `SerializedValue` (strict JSON-compatible union)

    Known types dispatch through TypeMapper. Unknown types get string fallback.
    """
    # Special case: callables that aren't in the mapper (functions, lambdas, etc.)
    # Use has_registered_handler() to check for explicit type support, not can_handle()
    # which always returns True when a default handler exists.
    # This ensures callables get stable "<function 'name'>" format instead of repr().
    if callable(value) and not _serializer.has_registered_handler(value):
        return _serialize_callable(value)

    return _serializer.map(value)
