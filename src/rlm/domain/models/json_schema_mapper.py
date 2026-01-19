"""
JSON Schema mapper for Python types.

Converts Python types to JSON Schema dictionaries. This replaces the duplicate
_python_type_to_json_schema functions in native.py and pydantic_output.py.

Design notes:
- Uses identity-based dispatch for basic types (dict lookup, O(1))
- Uses get_origin/get_args for parameterized generics
- Recursively handles nested types
- Falls back to {"type": "string"} for unknown types
- When Pydantic is available (optional dep), uses TypeAdapter for better edge case handling

Pydantic Integration (ADR-001):
- Pydantic is an optional dependency (`pip install rlm[pydantic]`)
- When available, TypeAdapter provides battle-tested schema generation
- When unavailable, manual implementation provides equivalent functionality
- Import is lazy-loaded and cached for performance

Example:
    mapper = JsonSchemaMapper()
    mapper.map(str)           # {"type": "string"}
    mapper.map(list[int])     # {"type": "array", "items": {"type": "integer"}}
    mapper.map(Optional[str]) # {"type": "string"}

"""

from __future__ import annotations

import dataclasses
import types
import typing
from typing import Any, get_args, get_origin

from rlm.domain.models.result import try_call

# =============================================================================
# Pydantic Optional Integration (ADR-001)
# =============================================================================

# Lazy import cache for Pydantic TypeAdapter
_PYDANTIC_CHECKED: bool = False
_TYPE_ADAPTER: type | None = None


def _get_pydantic_type_adapter() -> type | None:
    """
    Get Pydantic's TypeAdapter class if available.

    Uses lazy loading with caching - import check happens once per process.
    Returns None if pydantic is not installed.
    """
    global _PYDANTIC_CHECKED, _TYPE_ADAPTER  # noqa: PLW0603

    if not _PYDANTIC_CHECKED:
        try:
            from pydantic import TypeAdapter  # noqa: PLC0415 - Lazy import for optional dep

            _TYPE_ADAPTER = TypeAdapter
        except ImportError:
            _TYPE_ADAPTER = None
        _PYDANTIC_CHECKED = True

    return _TYPE_ADAPTER


# Basic type to JSON schema mapping (identity-based lookup)
_BASIC_TYPE_SCHEMAS: dict[type, dict[str, str]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
}

# Minimum number of type args for dict[K, V]
_DICT_TYPE_ARGS_COUNT = 2


class JsonSchemaMapper:
    """
    Maps Python types to JSON Schema dictionaries.

    Handles:
    - Basic types (str, int, float, bool, None)
    - Container types (list, dict)
    - Parameterized generics (list[X], dict[K, V])
    - Union and Optional types
    - Dataclasses (generates object schema from fields)
    - Pydantic models (delegates to model_json_schema)

    Pydantic Integration (ADR-001):
    - By default, uses manual implementation for backward compatibility
    - Set `prefer_pydantic=True` to use Pydantic TypeAdapter when available
    - Pydantic schemas are more explicit (e.g., Optional[int] → anyOf vs unwrapped int)
    - When Pydantic unavailable, falls back to manual implementation regardless of setting

    Thread safety:
    - Thread-safe (stateless after construction)

    Args:
        prefer_pydantic: If True (default), try Pydantic TypeAdapter first when available.
                        If False, always use manual implementation (useful for testing,
                        backward compatibility, or deterministic schema output).

    """

    def __init__(self, *, prefer_pydantic: bool = False) -> None:
        """
        Initialize the mapper.

        Args:
            prefer_pydantic: Whether to prefer Pydantic TypeAdapter when available.
                            Defaults to False for backward compatibility.
                            Set to True to get Pydantic's more explicit schemas
                            (e.g., Optional[int] → anyOf instead of unwrapped int).

        """
        self._prefer_pydantic = prefer_pydantic

    def map(self, python_type: type) -> dict[str, Any]:
        """
        Convert a Python type to its JSON Schema representation.

        Args:
            python_type: A Python type (e.g., str, list[int], Optional[str])

        Returns:
            JSON Schema dictionary

        Strategy (ADR-001):
            1. Try Pydantic TypeAdapter first (if pydantic installed)
            2. Fall back to manual implementation if Pydantic unavailable or fails

        """
        # Try Pydantic TypeAdapter first if preferred (optional dependency, ADR-001)
        if self._prefer_pydantic:
            type_adapter_cls = _get_pydantic_type_adapter()
            if type_adapter_cls is not None:
                pydantic_result = self._try_pydantic_schema(type_adapter_cls, python_type)
                if pydantic_result is not None:
                    return pydantic_result

        # Manual implementation (fallback when Pydantic unavailable, fails, or not preferred)
        return self._map_manual(python_type)

    def _try_pydantic_schema(
        self, type_adapter_cls: type, python_type: type
    ) -> dict[str, Any] | None:
        """
        Try to generate schema using Pydantic TypeAdapter.

        Returns None if schema generation fails for any reason.
        This allows graceful fallback to manual implementation.
        """
        try:
            # TypeAdapter is dynamically imported - pyright can't know its type
            adapter = type_adapter_cls(python_type)  # pyright: ignore[reportAny]
            schema = adapter.json_schema()  # pyright: ignore[reportAny]
            return dict(schema)  # pyright: ignore[reportAny]
        except Exception:  # noqa: BLE001 - Pydantic can raise various exceptions
            # Fall back to manual implementation
            return None

    def _map_manual(self, python_type: type) -> dict[str, Any]:
        """
        Manual JSON schema generation (no Pydantic dependency).

        This is the fallback implementation when Pydantic is not installed
        or when Pydantic fails to generate a schema for a given type.
        """
        # Handle None/NoneType
        if python_type is type(None):
            return {"type": "null"}

        # Basic types (O(1) lookup)
        if python_type in _BASIC_TYPE_SCHEMAS:
            # Return a copy to prevent mutation
            return _BASIC_TYPE_SCHEMAS[python_type].copy()

        # Handle parameterized generics and unions
        origin: type | None = get_origin(python_type)
        args: tuple[type, ...] = get_args(python_type)

        if origin is not None:
            return self._map_generic(origin, args)

        # Handle dataclasses
        if dataclasses.is_dataclass(python_type):
            return self._map_dataclass(python_type)

        # Handle Pydantic models (duck typing - works even without pydantic import)
        if hasattr(python_type, "model_json_schema"):
            return dict(python_type.model_json_schema())

        # Fallback to string for unknown types
        return {"type": "string"}

    def _map_generic(self, origin: type, args: tuple[type, ...]) -> dict[str, Any]:
        """Map a parameterized generic type to JSON schema."""
        # Handle list[X]
        if origin is list:
            if args:
                return {
                    "type": "array",
                    "items": self.map(args[0]),
                }
            return {"type": "array"}

        # Handle dict[K, V]
        if origin is dict:
            if len(args) >= _DICT_TYPE_ARGS_COUNT:
                return {
                    "type": "object",
                    "additionalProperties": self.map(args[1]),
                }
            return {"type": "object"}

        # Handle Union types (including Optional)
        if origin in (types.UnionType, typing.Union):
            return self._map_union(args)

        # Unknown generic - fallback
        return {"type": "string"}

    def _map_union(self, args: tuple[type, ...]) -> dict[str, Any]:
        """Map a Union type to JSON schema."""
        # Filter out None for Optional handling
        non_none_args = [a for a in args if a is not type(None)]

        # Optional[X] (Union[X, None]) - unwrap to X
        if len(non_none_args) == 1:
            return self.map(non_none_args[0])

        # Multi-type union - use anyOf
        return {
            "anyOf": [self.map(arg) for arg in args],
        }

    def _map_dataclass(self, dc_type: type) -> dict[str, Any]:
        """Map a dataclass to JSON schema."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        # Get type hints safely using Result pattern
        hints: dict[str, type] = try_call(lambda: typing.get_type_hints(dc_type)).unwrap_or({})

        for field in dataclasses.fields(dc_type):
            field_type: type = hints.get(field.name, str)
            properties[field.name] = self.map(field_type)

            # Field is required if it has no default and no default_factory
            if (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            ):
                required.append(field.name)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        return schema


# =============================================================================
# Testing Utilities
# =============================================================================


def has_pydantic() -> bool:
    """
    Check if Pydantic is available.

    Useful for conditional test logic and documentation.
    """
    return _get_pydantic_type_adapter() is not None


def _reset_pydantic_cache() -> None:
    """
    Reset the Pydantic import cache (for testing only).

    This allows tests to simulate Pydantic being unavailable.
    """
    global _PYDANTIC_CHECKED, _TYPE_ADAPTER  # noqa: PLW0603
    _PYDANTIC_CHECKED = False
    _TYPE_ADAPTER = None
