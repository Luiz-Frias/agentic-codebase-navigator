"""
Result type for explicit error handling.

This module provides a Result[T, E] type that forces explicit error handling
at compile time rather than relying on exception propagation.

Example:
    def parse_int(s: str) -> Result[int, ValueError]:
        result = try_call(lambda: int(s), ValueError)
        return result

    match parse_int("42"):
        case Ok(value=v):
            print(f"Parsed: {v}")
        case Err(error=e):
            print(f"Failed: {e}")

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")
E = TypeVar("E", bound=Exception)
U = TypeVar("U")


@dataclass(frozen=True, slots=True)
class Ok[T]:
    """
    Success case of Result.

    Wraps a successful value. Immutable and hashable (if value is hashable).
    """

    value: T

    def is_ok(self) -> bool:
        """Check if this is a success case."""
        return True

    def is_err(self) -> bool:
        """Check if this is an error case."""
        return False

    def unwrap(self) -> T:
        """
        Get the success value.

        Returns:
            The wrapped value

        Note:
            For Ok, this always succeeds. For Err, it raises the error.

        """
        return self.value

    def unwrap_or(self, _default: T) -> T:
        """
        Get the success value or a default.

        Args:
            _default: Value to return if this is Err (unused for Ok)

        Returns:
            The wrapped value (ignores default)

        """
        return self.value

    def map(self, fn: Callable[[T], U]) -> Ok[U]:
        """
        Transform the success value.

        Args:
            fn: Function to apply to the value

        Returns:
            New Ok with transformed value

        """
        return Ok(fn(self.value))


@dataclass(frozen=True, slots=True)
class Err[E: Exception]:
    """
    Error case of Result.

    Wraps an error. Immutable.
    """

    error: E

    def is_ok(self) -> bool:
        """Check if this is a success case."""
        return False

    def is_err(self) -> bool:
        """Check if this is an error case."""
        return True

    def unwrap(self) -> NoReturn:
        """
        Attempt to get a success value.

        Raises:
            The wrapped error

        Note:
            For Err, this always raises. Use unwrap_or() for safe access.

        """
        raise self.error

    def unwrap_or(self, default: T) -> T:
        """
        Get the success value or a default.

        Args:
            default: Value to return since this is Err

        Returns:
            The default value

        """
        return default

    def map(self, _fn: Callable[[T], U]) -> Err[E]:
        """
        Transform the success value (no-op for Err).

        Args:
            fn: Function that would be applied to a success value

        Returns:
            Self unchanged (error propagates)

        """
        return self


# Type alias for Result
Result = Ok[T] | Err[E]


def try_call[T, E: Exception](
    fn: Callable[[], T],
    error_type: type[E] = Exception,  # type: ignore[assignment]
) -> Result[T, E]:
    """
    Execute a function and wrap the result in Ok/Err.

    This is the bridge between exception-based code and Result-based code.

    Args:
        fn: Zero-argument function to execute
        error_type: Exception type to catch (default: Exception)
                   Other exceptions will propagate

    Returns:
        Ok(value) on success, Err(exception) on failure

    Example:
        result = try_call(lambda: int("42"))           # Ok(42)
        result = try_call(lambda: int("bad"))          # Err(ValueError)
        result = try_call(lambda: 1/0, ZeroDivisionError)  # Err(ZeroDivisionError)

    Note:
        Exceptions not matching error_type will propagate (re-raise).

    """
    try:
        return Ok(fn())
    except error_type as e:
        return Err(e)
