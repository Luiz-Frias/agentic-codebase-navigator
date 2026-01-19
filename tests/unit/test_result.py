"""
TDD tests for Result[T, E] abstraction.

These tests define the contract BEFORE implementation.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestOkBasics:
    """Ok variant - success case."""

    def test_ok_holds_value(self) -> None:
        """Ok wraps a success value."""
        from rlm.domain.models.result import Ok

        result = Ok(42)

        assert result.value == 42

    def test_ok_is_ok_returns_true(self) -> None:
        """Ok.is_ok() returns True."""
        from rlm.domain.models.result import Ok

        result = Ok("success")

        assert result.is_ok() is True

    def test_ok_is_err_returns_false(self) -> None:
        """Ok.is_err() returns False."""
        from rlm.domain.models.result import Ok

        result = Ok("success")

        assert result.is_err() is False

    def test_ok_unwrap_returns_value(self) -> None:
        """Ok.unwrap() returns the wrapped value."""
        from rlm.domain.models.result import Ok

        result = Ok(42)

        assert result.unwrap() == 42

    def test_ok_unwrap_or_returns_value(self) -> None:
        """Ok.unwrap_or() ignores default and returns value."""
        from rlm.domain.models.result import Ok

        result = Ok(42)

        assert result.unwrap_or(0) == 42


class TestErrBasics:
    """Err variant - error case."""

    def test_err_holds_error(self) -> None:
        """Err wraps an error."""
        from rlm.domain.models.result import Err

        error = ValueError("something went wrong")
        result = Err(error)

        assert result.error is error

    def test_err_is_ok_returns_false(self) -> None:
        """Err.is_ok() returns False."""
        from rlm.domain.models.result import Err

        result = Err(ValueError("oops"))

        assert result.is_ok() is False

    def test_err_is_err_returns_true(self) -> None:
        """Err.is_err() returns True."""
        from rlm.domain.models.result import Err

        result = Err(ValueError("oops"))

        assert result.is_err() is True

    def test_err_unwrap_raises_error(self) -> None:
        """Err.unwrap() raises the wrapped error."""
        from rlm.domain.models.result import Err

        error = ValueError("something went wrong")
        result = Err(error)

        with pytest.raises(ValueError, match="something went wrong"):
            result.unwrap()

    def test_err_unwrap_or_returns_default(self) -> None:
        """Err.unwrap_or() returns the default value."""
        from rlm.domain.models.result import Err

        result = Err(ValueError("oops"))

        assert result.unwrap_or(42) == 42


class TestResultMap:
    """Functor mapping over Result."""

    def test_ok_map_transforms_value(self) -> None:
        """Ok.map() applies function to value."""
        from rlm.domain.models.result import Ok

        result = Ok(2)
        mapped = result.map(lambda x: x * 3)

        assert mapped.unwrap() == 6

    def test_err_map_preserves_error(self) -> None:
        """Err.map() does not apply function, preserves error."""
        from rlm.domain.models.result import Err

        error = ValueError("oops")
        result = Err(error)
        mapped = result.map(lambda x: x * 3)

        assert mapped.is_err()
        assert mapped.error is error

    def test_map_chain(self) -> None:
        """Multiple maps can be chained."""
        from rlm.domain.models.result import Ok

        result = Ok(2).map(lambda x: x + 1).map(lambda x: x * 2).map(str)

        assert result.unwrap() == "6"


class TestTryCall:
    """Helper for wrapping exception-raising code."""

    def test_try_call_success(self) -> None:
        """try_call returns Ok on success."""
        from rlm.domain.models.result import try_call

        result = try_call(lambda: 42)

        assert result.is_ok()
        assert result.unwrap() == 42

    def test_try_call_failure(self) -> None:
        """try_call returns Err on exception."""
        from rlm.domain.models.result import try_call

        def failing():
            raise ValueError("boom")

        result = try_call(failing)

        assert result.is_err()
        assert isinstance(result.error, ValueError)
        assert str(result.error) == "boom"

    def test_try_call_with_specific_error_type(self) -> None:
        """try_call can filter by exception type."""
        from rlm.domain.models.result import try_call

        def failing():
            raise KeyError("not found")

        # Should catch KeyError
        result = try_call(failing, KeyError)
        assert result.is_err()
        assert isinstance(result.error, KeyError)

    def test_try_call_reraises_unmatched_error_type(self) -> None:
        """try_call re-raises exceptions not matching the type filter."""
        from rlm.domain.models.result import try_call

        def failing():
            raise KeyError("not found")

        # Should not catch KeyError when filtering for ValueError
        with pytest.raises(KeyError):
            try_call(failing, ValueError)


class TestResultPatternMatching:
    """Python 3.10+ match statement support."""

    def test_match_ok(self) -> None:
        """Ok can be matched in pattern matching."""
        from rlm.domain.models.result import Ok

        result = Ok(42)

        match result:
            case Ok(value=v):
                assert v == 42
            case _:
                pytest.fail("Should match Ok")

    def test_match_err(self) -> None:
        """Err can be matched in pattern matching."""
        from rlm.domain.models.result import Err

        result = Err(ValueError("oops"))

        match result:
            case Err(error=e):
                assert str(e) == "oops"
            case _:
                pytest.fail("Should match Err")


class TestResultRealWorldUseCases:
    """Tests that mirror actual codebase patterns we're replacing."""

    def test_json_serialization_fallback_pattern(self) -> None:
        """Pattern from query_metadata.py - json.dumps with fallback to repr."""
        import json

        from rlm.domain.models.result import try_call

        # This replaces: try: json.dumps(...) except Exception: repr(...)
        def safe_serialize(value: object) -> str:
            result = try_call(lambda: json.dumps(value))
            return result.unwrap_or(repr(value))

        # Normal case - serializable
        assert safe_serialize({"a": 1}) == '{"a": 1}'

        # Fallback case - not serializable
        class CustomObj:
            pass

        serialized = safe_serialize(CustomObj())
        assert "CustomObj" in serialized

    def test_sdk_error_boundary_pattern(self) -> None:
        """Pattern from LLM adapters - SDK errors → domain errors."""
        from rlm.domain.models.result import Err, Ok, try_call

        class SDKError(Exception):
            """Simulated external SDK error."""

        class LLMError(Exception):
            """Domain error."""

        def call_sdk(should_fail: bool) -> str:
            if should_fail:
                raise SDKError("API rate limit exceeded")
            return "success response"

        # Adapter boundary - convert SDK errors to domain errors
        def llm_complete(should_fail: bool) -> Ok[str] | Err[LLMError]:
            result = try_call(lambda: call_sdk(should_fail), SDKError)
            match result:
                case Ok(value=v):
                    return Ok(v)
                case Err(error=e):
                    return Err(LLMError(f"LLM call failed: {e}"))

        # Success case
        success = llm_complete(should_fail=False)
        assert success.is_ok()
        assert success.unwrap() == "success response"

        # Error case - converted to domain error
        failure = llm_complete(should_fail=True)
        assert failure.is_err()
        assert isinstance(failure.error, LLMError)
        assert "API rate limit exceeded" in str(failure.error)

    def test_type_hints_fallback_pattern(self) -> None:
        """Pattern from native.py - get_type_hints() with fallback."""
        from typing import get_type_hints

        from rlm.domain.models.result import try_call

        # This replaces: try: get_type_hints(fn) except Exception: {}
        def safe_get_hints(fn: object) -> dict[str, type]:
            result = try_call(lambda: get_type_hints(fn))
            return result.unwrap_or({})

        def typed_func(x: int, y: str) -> bool:
            return True

        # Normal case - type hints available
        hints = safe_get_hints(typed_func)
        assert hints == {"x": int, "y": str, "return": bool}

        # Fallback case - no type hints (e.g., forward reference errors)
        hints = safe_get_hints(lambda x: x)  # Lambda has no annotations
        assert hints == {}


class TestResultImmutability:
    """Result should be immutable (frozen dataclass)."""

    def test_ok_is_frozen(self) -> None:
        """Ok instances are immutable."""
        from rlm.domain.models.result import Ok

        result = Ok(42)

        with pytest.raises((AttributeError, TypeError)):  # FrozenInstanceError is a subtype
            result.value = 100  # type: ignore[misc]

    def test_err_is_frozen(self) -> None:
        """Err instances are immutable."""
        from rlm.domain.models.result import Err

        result = Err(ValueError("oops"))

        with pytest.raises((AttributeError, TypeError)):
            result.error = ValueError("different")  # type: ignore[misc]
