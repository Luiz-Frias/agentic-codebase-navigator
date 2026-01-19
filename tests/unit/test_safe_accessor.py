"""
TDD tests for SafeAccessor abstraction.

These tests define the contract BEFORE implementation.
SafeAccessor provides a unified interface for accessing attributes/keys
from either SDK objects or dict representations - the "duck-typing with fallback"
pattern that appears throughout SDK boundary code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

pytestmark = pytest.mark.unit


# ============================================================================
# Test Fixtures - Mock SDK objects and dicts
# ============================================================================


@dataclass
class MockSDKResponse:
    """Simulates an SDK response object with attributes."""

    choices: list[Any]
    model: str = "gpt-4"
    usage: dict[str, int] | None = None


@dataclass
class MockSDKChoice:
    """Simulates an SDK choice object."""

    message: MockSDKMessage
    finish_reason: str = "stop"


@dataclass
class MockSDKMessage:
    """Simulates an SDK message object."""

    content: str
    role: str = "assistant"
    tool_calls: list[Any] | None = None


# ============================================================================
# Phase 1: Basic Attribute/Key Access
# ============================================================================


class TestSafeAccessorBasicAccess:
    """Core access behavior - unified attribute/key retrieval."""

    def test_get_attribute_from_object(self) -> None:
        """get() retrieves attributes from objects."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        obj = MockSDKResponse(choices=[{"text": "hello"}])
        accessor = SafeAccessor(obj)

        result = accessor.get("choices")

        assert result == [{"text": "hello"}]

    def test_get_key_from_dict(self) -> None:
        """get() retrieves keys from dicts."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"choices": [{"text": "hello"}], "model": "gpt-4"}
        accessor = SafeAccessor(data)

        result = accessor.get("choices")

        assert result == [{"text": "hello"}]

    def test_get_missing_returns_none_by_default(self) -> None:
        """get() returns None for missing attributes/keys."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        obj = MockSDKResponse(choices=[])
        accessor = SafeAccessor(obj)

        result = accessor.get("nonexistent")

        assert result is None

    def test_get_missing_returns_custom_default(self) -> None:
        """get() returns custom default for missing attributes/keys."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        obj = MockSDKResponse(choices=[])
        accessor = SafeAccessor(obj)

        result = accessor.get("nonexistent", default="fallback")

        assert result == "fallback"

    def test_prefers_attribute_over_dict_key(self) -> None:
        """When object has both attr and __getitem__, prefer attribute."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        # A dict-like object that also has attributes
        class HybridObject(dict):  # type: ignore[type-arg]
            def __init__(self) -> None:
                super().__init__()
                self["choices"] = "from_dict"
                self.choices = "from_attr"

        hybrid = HybridObject()
        accessor = SafeAccessor(hybrid)

        result = accessor.get("choices")

        # Should prefer attribute access
        assert result == "from_attr"


# ============================================================================
# Phase 2: Nested Access
# ============================================================================


class TestSafeAccessorNestedAccess:
    """Navigate nested structures (SDK objects, dicts, mixed)."""

    def test_get_nested_from_objects(self) -> None:
        """get_nested() navigates through nested objects."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        message = MockSDKMessage(content="hello", role="assistant")
        choice = MockSDKChoice(message=message)
        response = MockSDKResponse(choices=[choice])

        accessor = SafeAccessor(response)

        # Navigate: response.choices[0].message.content
        result = accessor.get_nested("choices", 0, "message", "content")

        assert result == "hello"

    def test_get_nested_from_dicts(self) -> None:
        """get_nested() navigates through nested dicts."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {
            "choices": [{"message": {"content": "hello"}}],
        }
        accessor = SafeAccessor(data)

        result = accessor.get_nested("choices", 0, "message", "content")

        assert result == "hello"

    def test_get_nested_from_mixed_objects_and_dicts(self) -> None:
        """get_nested() navigates mixed object/dict structures."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        # SDK object containing a dict
        response = MockSDKResponse(
            choices=[{"message": {"content": "hello"}}],  # type: ignore[arg-type]
        )
        accessor = SafeAccessor(response)

        result = accessor.get_nested("choices", 0, "message", "content")

        assert result == "hello"

    def test_get_nested_returns_none_on_missing_key(self) -> None:
        """get_nested() returns None if any segment is missing."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"choices": [{"message": {}}]}
        accessor = SafeAccessor(data)

        result = accessor.get_nested("choices", 0, "message", "content")

        assert result is None

    def test_get_nested_returns_default_on_missing(self) -> None:
        """get_nested() returns custom default if path fails."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"choices": []}
        accessor = SafeAccessor(data)

        result = accessor.get_nested("choices", 0, "message", default="fallback")

        assert result == "fallback"

    def test_get_nested_handles_list_index_out_of_bounds(self) -> None:
        """get_nested() handles list index out of bounds gracefully."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"choices": [{"text": "first"}]}
        accessor = SafeAccessor(data)

        result = accessor.get_nested("choices", 5, "text")

        assert result is None


# ============================================================================
# Phase 3: Fluent Wrapping
# ============================================================================


class TestSafeAccessorFluent:
    """Fluent API for chaining access through nested structures."""

    def test_child_returns_accessor_for_nested_value(self) -> None:
        """child() returns a new SafeAccessor for the nested value."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"choices": [{"message": {"content": "hello"}}]}
        accessor = SafeAccessor(data)

        choice_accessor = accessor.child("choices", 0)

        assert choice_accessor.get("message") == {"content": "hello"}

    def test_child_on_missing_returns_accessor_with_none(self) -> None:
        """child() on missing key returns accessor wrapping None."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"choices": []}
        accessor = SafeAccessor(data)

        child = accessor.child("nonexistent")

        assert child.get("anything") is None

    def test_fluent_chaining(self) -> None:
        """Can chain child() calls for deep navigation."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"choices": [{"message": {"content": "hello"}}]}
        accessor = SafeAccessor(data)

        result = accessor.child("choices").child(0).child("message").get("content")

        assert result == "hello"


# ============================================================================
# Phase 4: Real-World Patterns from provider_base.py
# ============================================================================


class TestSafeAccessorProviderBasePatterns:
    """Tests mirroring actual patterns in provider_base.py."""

    def test_extract_choices_from_sdk_response(self) -> None:
        """Pattern: extract choices from either SDK object or dict."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        # SDK-style response (object)
        sdk_response = MockSDKResponse(
            choices=[MockSDKChoice(message=MockSDKMessage(content="hi"))],
        )
        accessor = SafeAccessor(sdk_response)
        choices = accessor.get("choices")
        assert choices is not None
        assert len(choices) == 1

        # Dict-style response (from JSON)
        dict_response = {"choices": [{"message": {"content": "hi"}}]}
        accessor2 = SafeAccessor(dict_response)
        choices2 = accessor2.get("choices")
        assert choices2 is not None
        assert len(choices2) == 1

    def test_extract_message_content_deeply(self) -> None:
        """Pattern: response.choices[0].message.content."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        # SDK-style
        sdk_response = MockSDKResponse(
            choices=[MockSDKChoice(message=MockSDKMessage(content="hello"))],
        )
        content = SafeAccessor(sdk_response).get_nested(
            "choices",
            0,
            "message",
            "content",
        )
        assert content == "hello"

        # Dict-style
        dict_response = {"choices": [{"message": {"content": "hello"}}]}
        content2 = SafeAccessor(dict_response).get_nested(
            "choices",
            0,
            "message",
            "content",
        )
        assert content2 == "hello"

    def test_extract_tool_calls(self) -> None:
        """Pattern: response.choices[0].message.tool_calls."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        tool_call = {"id": "call_1", "function": {"name": "foo", "arguments": "{}"}}

        # SDK-style with tool_calls
        message = MockSDKMessage(content="", tool_calls=[tool_call])
        sdk_response = MockSDKResponse(
            choices=[MockSDKChoice(message=message)],
        )
        tool_calls = SafeAccessor(sdk_response).get_nested(
            "choices",
            0,
            "message",
            "tool_calls",
        )
        assert tool_calls == [tool_call]

        # SDK-style without tool_calls
        sdk_response_no_tools = MockSDKResponse(
            choices=[MockSDKChoice(message=MockSDKMessage(content="no tools"))],
        )
        tool_calls2 = SafeAccessor(sdk_response_no_tools).get_nested(
            "choices",
            0,
            "message",
            "tool_calls",
        )
        assert tool_calls2 is None

    def test_extract_usage_tokens(self) -> None:
        """Pattern: response.usage.prompt_tokens / input_tokens."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        # SDK-style with usage
        sdk_response = MockSDKResponse(
            choices=[],
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        accessor = SafeAccessor(sdk_response)

        # Try both key names (some APIs use input_tokens, some prompt_tokens)
        prompt_tokens = accessor.get_nested("usage", "prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = accessor.get_nested("usage", "input_tokens")

        assert prompt_tokens == 100

    def test_handles_none_response(self) -> None:
        """SafeAccessor handles None gracefully."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        accessor = SafeAccessor(None)

        assert accessor.get("choices") is None
        assert accessor.get_nested("choices", 0, "message") is None
        assert accessor.child("choices").get("anything") is None


# ============================================================================
# Phase 5: Typed Accessors with Result[T, AccessError]
# Type-driven boundary pattern - typed accessors for domain/application layers
# ============================================================================


class TestTypedAccessorGetStr:
    """Tests for get_str() - type-safe string access."""

    def test_get_str_returns_ok_for_string_value(self) -> None:
        """get_str() returns Ok(str) when value is a string."""
        from rlm.domain.models.result import Ok
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"name": "hello"}
        accessor = SafeAccessor(data)

        result = accessor.get_str("name")

        assert isinstance(result, Ok)
        assert result.value == "hello"

    def test_get_str_returns_err_for_missing_key(self) -> None:
        """get_str() returns Err when key is missing."""
        from rlm.domain.models.result import Err
        from rlm.domain.models.safe_accessor import AccessError, SafeAccessor

        data = {"other": "value"}
        accessor = SafeAccessor(data)

        result = accessor.get_str("name")

        assert isinstance(result, Err)
        assert isinstance(result.error, AccessError)
        assert "not found" in str(result.error)

    def test_get_str_returns_err_for_wrong_type(self) -> None:
        """get_str() returns Err when value is not a string."""
        from rlm.domain.models.result import Err
        from rlm.domain.models.safe_accessor import AccessError, SafeAccessor

        data = {"count": 42}
        accessor = SafeAccessor(data)

        result = accessor.get_str("count")

        assert isinstance(result, Err)
        assert isinstance(result.error, AccessError)
        assert "expected str" in str(result.error)

    def test_get_str_or_returns_value_when_valid(self) -> None:
        """get_str_or() returns the string when valid."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"name": "hello"}
        accessor = SafeAccessor(data)

        result = accessor.get_str_or("name", "default")

        assert result == "hello"

    def test_get_str_or_returns_default_when_missing(self) -> None:
        """get_str_or() returns default when key is missing."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"other": "value"}
        accessor = SafeAccessor(data)

        result = accessor.get_str_or("name", "default")

        assert result == "default"

    def test_get_str_or_returns_default_for_wrong_type(self) -> None:
        """get_str_or() returns default when value is wrong type."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"count": 42}
        accessor = SafeAccessor(data)

        result = accessor.get_str_or("count", "default")

        assert result == "default"


class TestTypedAccessorGetInt:
    """Tests for get_int() - type-safe integer access."""

    def test_get_int_returns_ok_for_int_value(self) -> None:
        """get_int() returns Ok(int) when value is an int."""
        from rlm.domain.models.result import Ok
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"count": 42}
        accessor = SafeAccessor(data)

        result = accessor.get_int("count")

        assert isinstance(result, Ok)
        assert result.value == 42

    def test_get_int_returns_err_for_bool(self) -> None:
        """get_int() returns Err for bool values (bool is subclass of int)."""
        from rlm.domain.models.result import Err
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"flag": True}
        accessor = SafeAccessor(data)

        result = accessor.get_int("flag")

        assert isinstance(result, Err)
        assert "expected int, got bool" in str(result.error)

    def test_get_int_returns_err_for_missing_key(self) -> None:
        """get_int() returns Err when key is missing."""
        from rlm.domain.models.result import Err
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"other": 1}
        accessor = SafeAccessor(data)

        result = accessor.get_int("count")

        assert isinstance(result, Err)
        assert "not found" in str(result.error)

    def test_get_int_or_returns_value_when_valid(self) -> None:
        """get_int_or() returns the int when valid."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"count": 42}
        accessor = SafeAccessor(data)

        result = accessor.get_int_or("count", 0)

        assert result == 42

    def test_get_int_or_returns_default_for_bool(self) -> None:
        """get_int_or() returns default for bool values."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"flag": True}
        accessor = SafeAccessor(data)

        result = accessor.get_int_or("flag", 0)

        assert result == 0


class TestTypedAccessorGetFloat:
    """Tests for get_float() - type-safe float access."""

    def test_get_float_returns_ok_for_float_value(self) -> None:
        """get_float() returns Ok(float) when value is a float."""
        from rlm.domain.models.result import Ok
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"rate": 3.14}
        accessor = SafeAccessor(data)

        result = accessor.get_float("rate")

        assert isinstance(result, Ok)
        assert result.value == 3.14

    def test_get_float_widens_int_to_float(self) -> None:
        """get_float() accepts int and widens to float."""
        from rlm.domain.models.result import Ok
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"count": 42}
        accessor = SafeAccessor(data)

        result = accessor.get_float("count")

        assert isinstance(result, Ok)
        assert result.value == 42.0
        assert isinstance(result.value, float)

    def test_get_float_returns_err_for_bool(self) -> None:
        """get_float() returns Err for bool values."""
        from rlm.domain.models.result import Err
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"flag": True}
        accessor = SafeAccessor(data)

        result = accessor.get_float("flag")

        assert isinstance(result, Err)
        assert "expected float, got bool" in str(result.error)

    def test_get_float_or_returns_widened_value(self) -> None:
        """get_float_or() widens int to float."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"count": 42}
        accessor = SafeAccessor(data)

        result = accessor.get_float_or("count", 0.0)

        assert result == 42.0
        assert isinstance(result, float)


class TestTypedAccessorGetBool:
    """Tests for get_bool() - type-safe boolean access."""

    def test_get_bool_returns_ok_for_bool_value(self) -> None:
        """get_bool() returns Ok(bool) when value is a bool."""
        from rlm.domain.models.result import Ok
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"enabled": True}
        accessor = SafeAccessor(data)

        result = accessor.get_bool("enabled")

        assert isinstance(result, Ok)
        assert result.value is True

    def test_get_bool_returns_err_for_int(self) -> None:
        """get_bool() returns Err for int values (0/1 not treated as bool)."""
        from rlm.domain.models.result import Err
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"value": 1}
        accessor = SafeAccessor(data)

        result = accessor.get_bool("value")

        assert isinstance(result, Err)
        assert "expected bool" in str(result.error)

    def test_get_bool_or_returns_value_when_valid(self) -> None:
        """get_bool_or() returns the bool when valid."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"enabled": False}
        accessor = SafeAccessor(data)

        result = accessor.get_bool_or("enabled", True)

        assert result is False


class TestTypedAccessorGetList:
    """Tests for get_list() - type-safe list access."""

    def test_get_list_returns_ok_for_list_value(self) -> None:
        """get_list() returns Ok(list) when value is a list."""
        from rlm.domain.models.result import Ok
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"items": [1, 2, 3]}
        accessor = SafeAccessor(data)

        result = accessor.get_list("items")

        assert isinstance(result, Ok)
        assert result.value == [1, 2, 3]

    def test_get_list_returns_err_for_wrong_type(self) -> None:
        """get_list() returns Err when value is not a list."""
        from rlm.domain.models.result import Err
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"items": "not a list"}
        accessor = SafeAccessor(data)

        result = accessor.get_list("items")

        assert isinstance(result, Err)
        assert "expected list" in str(result.error)

    def test_get_list_or_returns_empty_list_when_missing(self) -> None:
        """get_list_or() returns empty list when key is missing."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"other": "value"}
        accessor = SafeAccessor(data)

        result = accessor.get_list_or("items")

        assert result == []


class TestTypedAccessorGetDict:
    """Tests for get_dict() - type-safe dict access."""

    def test_get_dict_returns_ok_for_dict_value(self) -> None:
        """get_dict() returns Ok(dict) when value is a dict."""
        from rlm.domain.models.result import Ok
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"config": {"key": "value"}}
        accessor = SafeAccessor(data)

        result = accessor.get_dict("config")

        assert isinstance(result, Ok)
        assert result.value == {"key": "value"}

    def test_get_dict_returns_err_for_wrong_type(self) -> None:
        """get_dict() returns Err when value is not a dict."""
        from rlm.domain.models.result import Err
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"config": [1, 2, 3]}
        accessor = SafeAccessor(data)

        result = accessor.get_dict("config")

        assert isinstance(result, Err)
        assert "expected dict" in str(result.error)

    def test_get_dict_or_returns_empty_dict_when_missing(self) -> None:
        """get_dict_or() returns empty dict when key is missing."""
        from rlm.domain.models.safe_accessor import SafeAccessor

        data = {"other": "value"}
        accessor = SafeAccessor(data)

        result = accessor.get_dict_or("config")

        assert result == {}


class TestTypedAccessorWithSDKObjects:
    """Tests that typed accessors work with SDK objects, not just dicts."""

    def test_get_str_from_sdk_object(self) -> None:
        """Typed accessors work with SDK object attributes."""
        from rlm.domain.models.result import Ok
        from rlm.domain.models.safe_accessor import SafeAccessor

        obj = MockSDKResponse(choices=[])
        accessor = SafeAccessor(obj)

        result = accessor.get_str("model")

        assert isinstance(result, Ok)
        assert result.value == "gpt-4"

    def test_get_list_from_sdk_object(self) -> None:
        """get_list() works with SDK object list attributes."""
        from rlm.domain.models.result import Ok
        from rlm.domain.models.safe_accessor import SafeAccessor

        obj = MockSDKResponse(choices=["a", "b", "c"])
        accessor = SafeAccessor(obj)

        result = accessor.get_list("choices")

        assert isinstance(result, Ok)
        assert result.value == ["a", "b", "c"]
