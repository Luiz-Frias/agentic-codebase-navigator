"""
Pydantic-based structured output adapter.

Validates and parses LLM responses into typed Python objects using Pydantic.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, cast, get_type_hints

from rlm.adapters.base import BaseStructuredOutputAdapter
from rlm.domain.errors import ValidationError


def _extract_json_from_response(response: str) -> str:
    """
    Extract JSON from an LLM response.

    Handles responses that contain JSON in code blocks or mixed with text.
    """
    # Try to find JSON in code blocks first
    code_block_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", response)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Try to find raw JSON object or array
    json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", response)
    if json_match:
        return json_match.group(1).strip()

    # Return as-is if no JSON found
    return response.strip()


def _type_to_json_schema(python_type: type) -> dict[str, Any]:
    """Convert a Python type to JSON Schema for structured output guidance."""
    # Handle None
    if python_type is type(None):
        return {"type": "null"}

    # Basic type mappings
    type_map: dict[type, dict[str, Any]] = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }

    if python_type in type_map:
        return type_map[python_type]

    # Handle generic types
    origin = getattr(python_type, "__origin__", None)
    args = getattr(python_type, "__args__", ())

    if origin is list and args:
        return {"type": "array", "items": _type_to_json_schema(args[0])}

    if origin is dict and len(args) >= 2:  # noqa: PLR2004
        value_type = args[1]  # type: ignore[misc]
        return {
            "type": "object",
            "additionalProperties": _type_to_json_schema(value_type),
        }

    # Handle dataclasses
    if dataclasses.is_dataclass(python_type):
        return _dataclass_to_schema(python_type)

    # Check for Pydantic model
    if hasattr(python_type, "model_json_schema"):
        return python_type.model_json_schema()

    # Fallback
    return {"type": "object"}


def _dataclass_to_schema(dc_type: type) -> dict[str, Any]:
    """Convert a dataclass to JSON Schema."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    try:
        hints = get_type_hints(dc_type)
    except Exception:
        hints = {}

    for dc_field in dataclasses.fields(dc_type):
        field_type = hints.get(dc_field.name, str)
        properties[dc_field.name] = _type_to_json_schema(field_type)

        # Field is required if it has no default and no default_factory
        if (
            dc_field.default is dataclasses.MISSING
            and dc_field.default_factory is dataclasses.MISSING
        ):
            required.append(dc_field.name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


class PydanticOutputAdapter[T](BaseStructuredOutputAdapter[T]):
    """
    Validates LLM responses against Pydantic models or dataclasses.

    Supports:
    - Pydantic BaseModel subclasses
    - Python dataclasses
    - TypedDict (basic support)
    - Simple types (str, int, float, bool, list, dict)

    Example:
        from pydantic import BaseModel

        class WeatherResponse(BaseModel):
            city: str
            temperature: float
            unit: str

        adapter = PydanticOutputAdapter()
        result = adapter.validate(
            '{"city": "NYC", "temperature": 72.5, "unit": "F"}',
            WeatherResponse
        )
        # result is a WeatherResponse instance
    """

    def validate(self, response: str, output_type: type[T], /) -> T:
        """
        Validate and parse an LLM response into the target type.

        Args:
            response: Raw LLM response (typically contains JSON)
            output_type: Target type to parse into

        Returns:
            Parsed and validated instance of output_type

        Raises:
            ValidationError: If parsing or validation fails
        """
        # Extract JSON from response
        json_str = _extract_json_from_response(response)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Failed to parse JSON from response: {e}") from e

        # Handle Pydantic models (duck typing for model_validate)
        if hasattr(output_type, "model_validate"):
            try:
                model_validate = output_type.model_validate  # type: ignore[attr-defined]
                return cast(T, model_validate(data))  # type: ignore[redundant-cast]
            except Exception as e:
                raise ValidationError(f"Pydantic validation failed: {e}") from e

        # Handle dataclasses
        if dataclasses.is_dataclass(output_type):
            if not isinstance(output_type, type):
                raise ValidationError("Expected a dataclass type, not an instance")
            try:
                return cast(T, output_type(**data))  # type: ignore[redundant-cast]
            except Exception as e:
                raise ValidationError(f"Dataclass instantiation failed: {e}") from e

        # Handle simple types
        if output_type in (str, int, float, bool):
            try:
                return cast(T, output_type(data))  # type: ignore[call-arg,redundant-cast]
            except Exception as e:
                raise ValidationError(f"Type conversion failed: {e}") from e

        # Handle list/dict - return as-is if types match
        if isinstance(data, output_type):  # type: ignore[arg-type]
            return cast(T, data)  # type: ignore[redundant-cast]

        raise ValidationError(f"Cannot validate response to type {output_type.__name__}")

    def get_schema(self, output_type: type[T], /) -> dict[str, Any]:
        """
        Get the JSON schema for an output type.

        This schema can be included in the system prompt to guide LLM output.
        """
        return _type_to_json_schema(output_type)
