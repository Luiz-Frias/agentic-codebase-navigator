"""
JSON Schema mapper for Python types.

Converts Python types to JSON Schema dictionaries. This replaces the duplicate
_python_type_to_json_schema functions in native.py and pydantic_output.py.

Design notes:
- Uses identity-based dispatch for basic types (dict lookup, O(1))
- Uses get_origin/get_args for parameterized generics
- Recursively handles nested types
- Falls back to {"type": "string"} for unknown types

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

    Thread safety:
    - Thread-safe (stateless, all methods are pure functions)

    """

    def map(self, python_type: type) -> dict[str, Any]:
        """
        Convert a Python type to its JSON Schema representation.

        Args:
            python_type: A Python type (e.g., str, list[int], Optional[str])

        Returns:
            JSON Schema dictionary

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

        # Handle Pydantic models (duck typing)
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
