"""
TDD tests for JsonSchemaMapper.

This mapper converts Python types to JSON Schema dictionaries, replacing the
duplicate _python_type_to_json_schema functions in native.py and pydantic_output.py.

Test categories:
1. Basic types (str, int, float, bool, None)
2. Container types (list, dict)
3. Parameterized generics (list[str], dict[str, int])
4. Union and Optional types
5. Nested types (list[list[int]], dict[str, list[int]])
6. Unknown types (fallback behavior)
"""

from __future__ import annotations

import dataclasses
import types
import typing
from typing import Optional, Union

import pytest

from rlm.domain.models.json_schema_mapper import JsonSchemaMapper


# =============================================================================
# Basic Types
# =============================================================================


@pytest.mark.unit
class TestBasicTypes:
    """Test JSON schema generation for primitive types."""

    def test_str_maps_to_string_schema(self) -> None:
        mapper = JsonSchemaMapper()
        assert mapper.map(str) == {"type": "string"}

    def test_int_maps_to_integer_schema(self) -> None:
        mapper = JsonSchemaMapper()
        assert mapper.map(int) == {"type": "integer"}

    def test_float_maps_to_number_schema(self) -> None:
        mapper = JsonSchemaMapper()
        assert mapper.map(float) == {"type": "number"}

    def test_bool_maps_to_boolean_schema(self) -> None:
        mapper = JsonSchemaMapper()
        assert mapper.map(bool) == {"type": "boolean"}

    def test_none_type_maps_to_null_schema(self) -> None:
        mapper = JsonSchemaMapper()
        assert mapper.map(type(None)) == {"type": "null"}


# =============================================================================
# Container Types (non-parameterized)
# =============================================================================


@pytest.mark.unit
class TestContainerTypes:
    """Test JSON schema generation for bare container types."""

    def test_bare_list_maps_to_array_schema(self) -> None:
        mapper = JsonSchemaMapper()
        assert mapper.map(list) == {"type": "array"}

    def test_bare_dict_maps_to_object_schema(self) -> None:
        mapper = JsonSchemaMapper()
        assert mapper.map(dict) == {"type": "object"}


# =============================================================================
# Parameterized Generics
# =============================================================================


@pytest.mark.unit
class TestParameterizedGenerics:
    """Test JSON schema generation for parameterized generic types."""

    def test_list_of_str_maps_to_array_with_items(self) -> None:
        mapper = JsonSchemaMapper()
        assert mapper.map(list[str]) == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_list_of_int_maps_to_array_with_items(self) -> None:
        mapper = JsonSchemaMapper()
        assert mapper.map(list[int]) == {
            "type": "array",
            "items": {"type": "integer"},
        }

    def test_dict_str_int_maps_to_object_with_additional_properties(self) -> None:
        mapper = JsonSchemaMapper()
        assert mapper.map(dict[str, int]) == {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }

    def test_dict_str_str_maps_to_object_with_additional_properties(self) -> None:
        mapper = JsonSchemaMapper()
        assert mapper.map(dict[str, str]) == {
            "type": "object",
            "additionalProperties": {"type": "string"},
        }


# =============================================================================
# Union and Optional Types
# =============================================================================


@pytest.mark.unit
class TestUnionTypes:
    """Test JSON schema generation for Union and Optional types."""

    def test_optional_int_unwraps_to_integer(self) -> None:
        """Optional[X] with single non-None type unwraps to X's schema."""
        mapper = JsonSchemaMapper()
        # Optional[int] is Union[int, None]
        assert mapper.map(Optional[int]) == {"type": "integer"}

    def test_int_or_none_unwraps_to_integer(self) -> None:
        """X | None syntax also unwraps to X's schema."""
        mapper = JsonSchemaMapper()
        assert mapper.map(int | None) == {"type": "integer"}

    def test_union_of_two_types_maps_to_anyof(self) -> None:
        """Union[X, Y] with multiple non-None types maps to anyOf."""
        mapper = JsonSchemaMapper()
        result = mapper.map(Union[int, str])
        assert result == {
            "anyOf": [{"type": "integer"}, {"type": "string"}],
        }

    def test_union_with_pipe_syntax_maps_to_anyof(self) -> None:
        """X | Y syntax maps to anyOf."""
        mapper = JsonSchemaMapper()
        result = mapper.map(int | str)
        assert result == {
            "anyOf": [{"type": "integer"}, {"type": "string"}],
        }

    def test_union_of_three_types_maps_to_anyof(self) -> None:
        """Union with 3+ types maps to anyOf with all schemas."""
        mapper = JsonSchemaMapper()
        result = mapper.map(Union[int, str, float])
        assert result == {
            "anyOf": [
                {"type": "integer"},
                {"type": "string"},
                {"type": "number"},
            ],
        }


# =============================================================================
# Nested Types
# =============================================================================


@pytest.mark.unit
class TestNestedTypes:
    """Test JSON schema generation for nested/complex types."""

    def test_list_of_list_of_int(self) -> None:
        mapper = JsonSchemaMapper()
        assert mapper.map(list[list[int]]) == {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
            },
        }

    def test_dict_str_list_int(self) -> None:
        mapper = JsonSchemaMapper()
        assert mapper.map(dict[str, list[int]]) == {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "integer"},
            },
        }

    def test_list_of_optional_str(self) -> None:
        mapper = JsonSchemaMapper()
        # list[Optional[str]] - the Optional unwraps to str
        assert mapper.map(list[Optional[str]]) == {
            "type": "array",
            "items": {"type": "string"},
        }


# =============================================================================
# Fallback Behavior
# =============================================================================


@pytest.mark.unit
class TestFallbackBehavior:
    """Test fallback for unknown/unsupported types."""

    def test_unknown_class_falls_back_to_string(self) -> None:
        mapper = JsonSchemaMapper()

        class CustomClass:
            pass

        assert mapper.map(CustomClass) == {"type": "string"}

    def test_bytes_falls_back_to_string(self) -> None:
        """bytes is not a standard JSON type, falls back to string."""
        mapper = JsonSchemaMapper()
        assert mapper.map(bytes) == {"type": "string"}


# =============================================================================
# Dataclass Support (for pydantic_output.py compatibility)
# =============================================================================


@pytest.mark.unit
class TestDataclassSupport:
    """Test JSON schema generation for dataclasses."""

    def test_simple_dataclass_maps_to_object_schema(self) -> None:
        @dataclasses.dataclass
        class Person:
            name: str
            age: int

        mapper = JsonSchemaMapper()
        result = mapper.map(Person)

        assert result["type"] == "object"
        assert result["properties"]["name"] == {"type": "string"}
        assert result["properties"]["age"] == {"type": "integer"}
        assert set(result["required"]) == {"name", "age"}

    def test_dataclass_with_optional_field(self) -> None:
        @dataclasses.dataclass
        class Config:
            name: str
            timeout: int = 30  # Has default, not required

        mapper = JsonSchemaMapper()
        result = mapper.map(Config)

        assert result["type"] == "object"
        assert "name" in result["required"]
        assert "timeout" not in result["required"]

    def test_dataclass_with_nested_types(self) -> None:
        @dataclasses.dataclass
        class Container:
            items: list[str]
            metadata: dict[str, int]

        mapper = JsonSchemaMapper()
        result = mapper.map(Container)

        assert result["properties"]["items"] == {
            "type": "array",
            "items": {"type": "string"},
        }
        assert result["properties"]["metadata"] == {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }


# =============================================================================
# Pydantic Model Support
# =============================================================================


@pytest.mark.unit
class TestPydanticSupport:
    """Test that Pydantic models delegate to model_json_schema."""

    def test_pydantic_model_uses_model_json_schema(self) -> None:
        """If type has model_json_schema method, use it."""

        class FakePydanticModel:
            @staticmethod
            def model_json_schema() -> dict[str, object]:
                return {"type": "object", "title": "FakeModel"}

        mapper = JsonSchemaMapper()
        result = mapper.map(FakePydanticModel)

        assert result == {"type": "object", "title": "FakeModel"}


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_typing_list_vs_builtin_list(self) -> None:
        """Both typing.List and list should work the same."""
        mapper = JsonSchemaMapper()
        assert mapper.map(typing.List[int]) == mapper.map(list[int])  # noqa: UP006

    def test_typing_dict_vs_builtin_dict(self) -> None:
        """Both typing.Dict and dict should work the same."""
        mapper = JsonSchemaMapper()
        assert mapper.map(typing.Dict[str, int]) == mapper.map(dict[str, int])  # noqa: UP006

    def test_returns_copy_not_reference(self) -> None:
        """Ensure each call returns a fresh dict, not a shared reference."""
        mapper = JsonSchemaMapper()
        result1 = mapper.map(str)
        result2 = mapper.map(str)

        # Modify one
        result1["extra"] = "modified"

        # Other should be unaffected
        assert "extra" not in result2
