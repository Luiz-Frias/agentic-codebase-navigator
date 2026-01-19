"""
Composable validation for domain invariants.

This module provides a Validator[T] abstraction that replaces scattered
isinstance() checks and validation guards with a declarative, testable pattern.

Example:
    validator = (
        Validator[object]()
        .is_type(str, "Must be a string")
        .not_blank("Must not be empty")
    )

    validator.validate("hello")  # Returns "hello"
    validator.validate("")       # Raises ValidationError
    validator.validate(42)       # Raises ValidationError

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from rlm.domain.errors import ValidationError
from rlm.domain.models.result import Err, Ok

if TYPE_CHECKING:
    from collections.abc import Callable, Sized

    from rlm.domain.models.result import Result

T = TypeVar("T")


@dataclass
class Validator[T]:
    """
    Composable validator for domain values.

    Chains multiple validation rules that are checked in order.
    Fails fast on first rule violation.

    Design notes:
    - Rules are checked in registration order (fail fast)
    - Returns the validated value on success (enables fluent usage)
    - Raises ValidationError on failure with the rule's error message
    - Integrates with Result[T, E] via validate_to_result()

    Thread safety:
    - NOT thread-safe during rule addition (build validators at startup)
    - Thread-safe during validate() calls (read-only after construction)
    """

    _rules: list[tuple[Callable[[T], bool], str]] = field(default_factory=list)

    def is_type(self, type_: type, error_message: str) -> Validator[T]:
        """
        Add a type check rule.

        Args:
            type_: Expected type (uses isinstance, so subclasses match)
            error_message: Error message if check fails

        Returns:
            Self for method chaining

        """
        # Capture type_ in closure with explicit typing for mypy
        expected_type = type_

        def _check_type(x: T) -> bool:
            return isinstance(x, expected_type)

        self._rules.append((_check_type, error_message))
        return self

    def satisfies(
        self,
        predicate: Callable[[T], bool],
        error_message: str,
    ) -> Validator[T]:
        """
        Add a custom predicate rule.

        Args:
            predicate: Function that returns True if value is valid
            error_message: Error message if predicate returns False

        Returns:
            Self for method chaining

        """
        self._rules.append((predicate, error_message))
        return self

    def not_blank(self, error_message: str) -> Validator[T]:
        """
        Add a non-blank string check.

        Checks that the value (assumed to be a string) is not empty
        and not whitespace-only.

        Args:
            error_message: Error message if check fails

        Returns:
            Self for method chaining

        """
        self._rules.append((lambda x: bool(str(x).strip()), error_message))
        return self

    def not_empty(self, error_message: str) -> Validator[T]:
        """
        Add a non-empty collection check.

        Checks that the value has length > 0.

        Args:
            error_message: Error message if check fails

        Returns:
            Self for method chaining

        """

        def _check_not_empty(x: Sized) -> bool:
            return len(x) > 0

        self._rules.append((_check_not_empty, error_message))  # type: ignore[arg-type]
        return self

    def each(self, element_validator: Validator[T]) -> Validator[T]:
        """
        Add a rule that validates each element in a collection.

        Args:
            element_validator: Validator to apply to each element

        Returns:
            Self for method chaining

        """

        def _validate_each(collection: T) -> bool:
            for item in collection:  # type: ignore[attr-defined]
                element_validator.validate(item)
            return True

        # Note: We let the element_validator raise on failure
        # so we don't need an error message here
        self._rules.append((_validate_each, ""))
        return self

    def _check_rule(
        self,
        value: T,
        predicate: Callable[[T], bool],
        error_message: str,
    ) -> None:
        """Check a single validation rule, raising ValidationError on failure."""
        result = predicate(value)
        if not result:
            raise ValidationError(error_message)

    def validate(self, value: T) -> T:
        """
        Run all validation rules against a value.

        Args:
            value: The value to validate

        Returns:
            The same value if all validations pass

        Raises:
            ValidationError: If any rule fails

        """
        for predicate, error_message in self._rules:
            self._check_rule(value, predicate, error_message)

        return value

    def validate_to_result(self, value: T) -> Result[T, ValidationError]:
        """
        Run all validation rules and return Result instead of raising.

        Args:
            value: The value to validate

        Returns:
            Ok(value) if all validations pass, Err(ValidationError) otherwise

        """
        try:
            return Ok(self.validate(value))
        except ValidationError as e:
            return Err(e)


# ============================================================================
# Pre-built validator factories for common patterns
# ============================================================================


def non_empty_string(error_message: str) -> Validator[object]:
    """
    Create a validator for non-empty strings.

    Args:
        error_message: Error message for any validation failure

    Returns:
        Validator that checks: is string AND not blank

    """
    return Validator[object]().is_type(str, error_message).not_blank(error_message)


def tuple_of[T](element_validator: Validator[T]) -> Validator[object]:
    """
    Create a validator for tuples where all elements must be valid.

    Args:
        element_validator: Validator to apply to each element

    Returns:
        Validator that checks: is tuple AND each element is valid

    """
    return (
        Validator[object]().is_type(tuple, "Must be a tuple").each(element_validator)  # type: ignore[arg-type]
    )
