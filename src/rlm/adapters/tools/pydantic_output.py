"""
Pydantic-based structured output adapter.

Validates and parses LLM responses into typed Python objects using Pydantic.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, cast

from rlm.adapters.base import BaseStructuredOutputAdapter
from rlm.domain.errors import ValidationError
from rlm.domain.models.json_schema_mapper import JsonSchemaMapper

# Module-level mapper instance (stateless, thread-safe)
_schema_mapper = JsonSchemaMapper()


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
        model_validate = getattr(output_type, "model_validate", None)
        if model_validate is not None:
            try:
                validated_result = model_validate(data)
                return cast("T", validated_result)
            except Exception as e:
                raise ValidationError(f"Pydantic validation failed: {e}") from e

        # Handle dataclasses
        if dataclasses.is_dataclass(output_type):
            if not isinstance(output_type, type):
                raise ValidationError("Expected a dataclass type, not an instance")
            try:
                return output_type(**data)  # type: ignore[return-value]
            except Exception as e:
                raise ValidationError(f"Dataclass instantiation failed: {e}") from e

        # Handle simple types
        if output_type in (str, int, float, bool):
            try:
                return output_type(data)  # type: ignore[call-arg,return-value]
            except Exception as e:
                raise ValidationError(f"Type conversion failed: {e}") from e

        # Handle list/dict - return as-is if types match
        if isinstance(data, (list, dict)):
            return cast("T", data)

        raise ValidationError(f"Cannot validate response to type {output_type.__name__}")

    def get_schema(self, output_type: type[T], /) -> dict[str, Any]:
        """
        Get the JSON schema for an output type.

        This schema can be included in the system prompt to guide LLM output.
        """
        return _schema_mapper.map(output_type)
