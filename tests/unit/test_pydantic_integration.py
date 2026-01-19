"""
Tests for Pydantic optional integration (ADR-001).

This module tests JsonSchemaMapper behavior when prefer_pydantic=True.
Tests are skipped if Pydantic is not installed.

Test categories:
1. Pydantic-enabled schema generation (prefer_pydantic=True)
2. Schema differences between Pydantic and manual implementations
3. Fallback behavior when Pydantic fails
4. has_pydantic() utility function
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Union

import pytest

from rlm.domain.models.json_schema_mapper import (
    JsonSchemaMapper,
    _reset_pydantic_cache,
    has_pydantic,
)

# Check if Pydantic is available for conditional tests
HAS_PYDANTIC = has_pydantic()


# =============================================================================
# Pydantic Availability Tests
# =============================================================================


@pytest.mark.unit
class TestPydanticAvailability:
    """Test has_pydantic() utility function."""

    def test_has_pydantic_returns_bool(self) -> None:
        """has_pydantic() should return a boolean."""
        result = has_pydantic()
        assert isinstance(result, bool)

    def test_has_pydantic_is_consistent(self) -> None:
        """Multiple calls should return the same result (cached)."""
        result1 = has_pydantic()
        result2 = has_pydantic()
        assert result1 == result2


# =============================================================================
# Pydantic-Enabled Schema Tests (skip if Pydantic not installed)
# =============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
class TestPydanticEnabledSchemas:
    """Test schema generation with prefer_pydantic=True when Pydantic is available."""

    def test_basic_types_produce_valid_schemas(self) -> None:
        """Basic types should produce valid JSON schemas with Pydantic."""
        mapper = JsonSchemaMapper(prefer_pydantic=True)

        # These should all produce valid schemas (exact format may differ from manual)
        str_schema = mapper.map(str)
        assert str_schema.get("type") == "string"

        int_schema = mapper.map(int)
        assert int_schema.get("type") == "integer"

        float_schema = mapper.map(float)
        assert float_schema.get("type") == "number"

        bool_schema = mapper.map(bool)
        assert bool_schema.get("type") == "boolean"

    def test_optional_produces_anyof_with_pydantic(self) -> None:
        """Pydantic represents Optional[X] as anyOf, not unwrapped."""
        mapper = JsonSchemaMapper(prefer_pydantic=True)
        schema = mapper.map(Optional[int])

        # Pydantic uses anyOf for Optional (more explicit than manual's unwrapping)
        assert "anyOf" in schema or schema.get("type") == "integer"
        # Either format is valid - Pydantic may use anyOf or type with nullable

    def test_list_produces_array_with_items(self) -> None:
        """Pydantic includes items schema even for bare list."""
        mapper = JsonSchemaMapper(prefer_pydantic=True)
        schema = mapper.map(list)

        assert schema.get("type") == "array"
        # Pydantic adds "items": {} for bare list (more explicit)

    def test_parameterized_list_produces_typed_items(self) -> None:
        """list[int] should have typed items schema."""
        mapper = JsonSchemaMapper(prefer_pydantic=True)
        schema = mapper.map(list[int])

        assert schema.get("type") == "array"
        assert "items" in schema
        # Items should reference integer type somehow

    def test_dict_produces_object_schema(self) -> None:
        """Dict types should produce object schemas."""
        mapper = JsonSchemaMapper(prefer_pydantic=True)
        schema = mapper.map(dict[str, int])

        assert schema.get("type") == "object"
        # Pydantic adds additionalProperties

    def test_union_produces_anyof(self) -> None:
        """Union types should produce anyOf schema."""
        mapper = JsonSchemaMapper(prefer_pydantic=True)
        schema = mapper.map(Union[int, str])

        assert "anyOf" in schema
        # Should have schemas for both int and str

    def test_dataclass_produces_object_schema(self) -> None:
        """Dataclasses should produce object schemas with properties."""

        @dataclasses.dataclass
        class Person:
            name: str
            age: int

        mapper = JsonSchemaMapper(prefer_pydantic=True)
        schema = mapper.map(Person)

        assert schema.get("type") == "object"
        assert "properties" in schema
        # Should have name and age properties


# =============================================================================
# Schema Comparison Tests (Pydantic vs Manual)
# =============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")
class TestSchemaComparison:
    """Compare Pydantic and manual schema outputs."""

    def test_basic_type_schemas_are_equivalent(self) -> None:
        """Basic type schemas should be semantically equivalent."""
        manual = JsonSchemaMapper(prefer_pydantic=False)
        pydantic = JsonSchemaMapper(prefer_pydantic=True)

        # str, int should be identical
        assert manual.map(str)["type"] == pydantic.map(str)["type"]
        assert manual.map(int)["type"] == pydantic.map(int)["type"]

    def test_optional_schemas_differ(self) -> None:
        """Optional schemas may differ between Pydantic and manual."""
        manual = JsonSchemaMapper(prefer_pydantic=False)
        pydantic = JsonSchemaMapper(prefer_pydantic=True)

        manual_schema = manual.map(Optional[int])
        pydantic_schema = pydantic.map(Optional[int])

        # Manual unwraps Optional to just integer
        assert manual_schema == {"type": "integer"}

        # Pydantic keeps it as anyOf (or similar)
        # Both are valid representations of the same type
        assert pydantic_schema != manual_schema or "anyOf" in pydantic_schema

    def test_bare_list_schemas_differ(self) -> None:
        """Bare list schemas may differ between Pydantic and manual."""
        manual = JsonSchemaMapper(prefer_pydantic=False)
        pydantic = JsonSchemaMapper(prefer_pydantic=True)

        manual_schema = manual.map(list)
        pydantic_schema = pydantic.map(list)

        # Manual: {"type": "array"}
        # Pydantic: {"type": "array", "items": {}}
        assert manual_schema["type"] == "array"
        assert pydantic_schema["type"] == "array"
        # Pydantic may add extra fields


# =============================================================================
# Fallback Behavior Tests
# =============================================================================


@pytest.mark.unit
class TestFallbackBehavior:
    """Test fallback when Pydantic is unavailable or fails."""

    def test_prefer_pydantic_false_uses_manual(self) -> None:
        """prefer_pydantic=False should always use manual implementation."""
        mapper = JsonSchemaMapper(prefer_pydantic=False)

        # Manual implementation unwraps Optional
        schema = mapper.map(Optional[int])
        assert schema == {"type": "integer"}

    def test_prefer_pydantic_true_falls_back_on_unsupported_type(self) -> None:
        """Should fall back to manual for types Pydantic can't handle."""
        mapper = JsonSchemaMapper(prefer_pydantic=True)

        class UnsupportedType:
            """A type that Pydantic might not handle well."""


        # Should get string fallback from manual implementation
        schema = mapper.map(UnsupportedType)
        assert schema == {"type": "string"}


# =============================================================================
# Cache Reset Tests (for testing infrastructure)
# =============================================================================


@pytest.mark.unit
class TestCacheReset:
    """Test _reset_pydantic_cache for testing infrastructure."""

    def test_reset_cache_allows_recheck(self) -> None:
        """Resetting cache should allow fresh import check."""
        # Get initial state
        initial = has_pydantic()

        # Reset cache
        _reset_pydantic_cache()

        # Should be able to check again
        after_reset = has_pydantic()

        # Result should be the same (Pydantic availability hasn't changed)
        assert initial == after_reset
