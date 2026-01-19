"""
TDD tests for Validator abstraction.

These tests define the contract BEFORE implementation.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestValidatorBasicPredicates:
    """Core predicate validation behavior."""

    def test_is_type_passes_for_correct_type(self) -> None:
        """is_type(str) passes for string values."""
        from rlm.domain.models.validation import Validator

        validator = Validator[object]().is_type(str, "Must be a string")

        # Should not raise
        result = validator.validate("hello")
        assert result == "hello"

    def test_is_type_fails_for_wrong_type(self) -> None:
        """is_type(str) fails for non-string values."""
        from rlm.domain.errors import ValidationError
        from rlm.domain.models.validation import Validator

        validator = Validator[object]().is_type(str, "Must be a string")

        with pytest.raises(ValidationError, match="Must be a string"):
            validator.validate(42)

    def test_satisfies_with_custom_predicate(self) -> None:
        """satisfies() allows custom predicate functions."""
        from rlm.domain.models.validation import Validator

        validator = Validator[int]().satisfies(lambda x: x > 0, "Must be positive")

        assert validator.validate(5) == 5

    def test_satisfies_fails_when_predicate_false(self) -> None:
        """satisfies() raises when predicate returns False."""
        from rlm.domain.errors import ValidationError
        from rlm.domain.models.validation import Validator

        validator = Validator[int]().satisfies(lambda x: x > 0, "Must be positive")

        with pytest.raises(ValidationError, match="Must be positive"):
            validator.validate(-1)


class TestValidatorStringPredicates:
    """String-specific validation predicates."""

    def test_not_blank_passes_for_non_empty_string(self) -> None:
        """not_blank() passes for strings with content."""
        from rlm.domain.models.validation import Validator

        validator = Validator[str]().not_blank("Must not be empty")

        assert validator.validate("hello") == "hello"

    def test_not_blank_fails_for_empty_string(self) -> None:
        """not_blank() fails for empty strings."""
        from rlm.domain.errors import ValidationError
        from rlm.domain.models.validation import Validator

        validator = Validator[str]().not_blank("Must not be empty")

        with pytest.raises(ValidationError, match="Must not be empty"):
            validator.validate("")

    def test_not_blank_fails_for_whitespace_only(self) -> None:
        """not_blank() fails for whitespace-only strings."""
        from rlm.domain.errors import ValidationError
        from rlm.domain.models.validation import Validator

        validator = Validator[str]().not_blank("Must not be empty")

        with pytest.raises(ValidationError, match="Must not be empty"):
            validator.validate("   \t\n  ")


class TestValidatorChaining:
    """Multiple validations chained together."""

    def test_chained_validators_all_pass(self) -> None:
        """Multiple chained validators all run in sequence."""
        from rlm.domain.models.validation import Validator

        validator = (
            Validator[object]().is_type(str, "Must be string").not_blank("Must not be empty")
        )

        assert validator.validate("hello") == "hello"

    def test_chained_validators_fail_on_first_failure(self) -> None:
        """Chain stops at first failure."""
        from rlm.domain.errors import ValidationError
        from rlm.domain.models.validation import Validator

        validator = (
            Validator[object]().is_type(str, "Must be string").not_blank("Must not be empty")
        )

        # Fails on type check (first rule)
        with pytest.raises(ValidationError, match="Must be string"):
            validator.validate(42)

    def test_chained_validators_fail_on_second_rule(self) -> None:
        """Chain fails on second rule if first passes."""
        from rlm.domain.errors import ValidationError
        from rlm.domain.models.validation import Validator

        validator = (
            Validator[object]().is_type(str, "Must be string").not_blank("Must not be empty")
        )

        # Passes type check, fails on not_blank
        with pytest.raises(ValidationError, match="Must not be empty"):
            validator.validate("")

    def test_returns_self_for_method_chaining(self) -> None:
        """All methods return self for fluent API."""
        from rlm.domain.models.validation import Validator

        validator = Validator[object]()
        result1 = validator.is_type(str, "err")
        result2 = result1.not_blank("err")
        result3 = result2.satisfies(lambda x: True, "err")

        assert result1 is validator
        assert result2 is validator
        assert result3 is validator


class TestValidatorCollectionPredicates:
    """Collection-specific validation predicates."""

    def test_not_empty_passes_for_non_empty_collection(self) -> None:
        """not_empty() passes for collections with items."""
        from rlm.domain.models.validation import Validator

        validator = Validator[tuple]().not_empty("Must have items")  # type: ignore[type-arg]

        assert validator.validate((1, 2, 3)) == (1, 2, 3)

    def test_not_empty_fails_for_empty_collection(self) -> None:
        """not_empty() fails for empty collections."""
        from rlm.domain.errors import ValidationError
        from rlm.domain.models.validation import Validator

        validator = Validator[tuple]().not_empty("Must have items")  # type: ignore[type-arg]

        with pytest.raises(ValidationError, match="Must have items"):
            validator.validate(())

    def test_each_validates_all_elements(self) -> None:
        """each() applies validator to every element."""
        from rlm.domain.models.validation import Validator

        element_validator = Validator[object]().is_type(str, "Element must be string")
        validator = Validator[tuple]().each(element_validator)  # type: ignore[type-arg]

        assert validator.validate(("a", "b", "c")) == ("a", "b", "c")

    def test_each_fails_on_invalid_element(self) -> None:
        """each() fails if any element is invalid."""
        from rlm.domain.errors import ValidationError
        from rlm.domain.models.validation import Validator

        element_validator = Validator[object]().is_type(str, "Element must be string")
        validator = Validator[tuple]().each(element_validator)  # type: ignore[type-arg]

        with pytest.raises(ValidationError, match="Element must be string"):
            validator.validate(("a", 42, "c"))


class TestValidatorResultIntegration:
    """Integration with Result[T, E] pattern."""

    def test_validate_to_result_returns_ok_on_success(self) -> None:
        """validate_to_result() returns Ok on valid input."""
        from rlm.domain.models.result import Ok
        from rlm.domain.models.validation import Validator

        validator = Validator[str]().not_blank("err")

        result = validator.validate_to_result("hello")

        assert isinstance(result, Ok)
        assert result.value == "hello"

    def test_validate_to_result_returns_err_on_failure(self) -> None:
        """validate_to_result() returns Err on invalid input."""
        from rlm.domain.models.result import Err
        from rlm.domain.models.validation import Validator

        validator = Validator[str]().not_blank("Must not be empty")

        result = validator.validate_to_result("")

        assert isinstance(result, Err)
        assert "Must not be empty" in str(result.error)


class TestValidatorRealWorldUseCases:
    """Tests that mirror actual codebase patterns we're replacing."""

    def test_model_spec_name_validation_pattern(self) -> None:
        """Pattern from model_spec.py - validate name is non-empty string."""
        from rlm.domain.errors import ValidationError
        from rlm.domain.models.validation import Validator

        # This replaces:
        # if not isinstance(self.name, str) or not self.name.strip():
        #     raise ValidationError("ModelSpec.name must be a non-empty string")

        name_validator = (
            Validator[object]()
            .is_type(str, "ModelSpec.name must be a non-empty string")
            .not_blank("ModelSpec.name must be a non-empty string")
        )

        # Valid cases
        assert name_validator.validate("gpt-4") == "gpt-4"
        assert name_validator.validate("claude-3") == "claude-3"

        # Invalid cases
        with pytest.raises(ValidationError):
            name_validator.validate(123)  # Not a string

        with pytest.raises(ValidationError):
            name_validator.validate("")  # Empty

        with pytest.raises(ValidationError):
            name_validator.validate("   ")  # Whitespace only

    def test_model_spec_aliases_validation_pattern(self) -> None:
        """Pattern from model_spec.py - validate aliases is tuple of non-empty strings."""
        from rlm.domain.errors import ValidationError
        from rlm.domain.models.validation import Validator

        # This replaces:
        # if not isinstance(self.aliases, tuple):
        #     raise ValidationError("ModelSpec.aliases must be a tuple of strings")
        # for a in self.aliases:
        #     if not isinstance(a, str) or not a.strip():
        #         raise ValidationError("...")

        alias_element = (
            Validator[object]()
            .is_type(str, "Alias must be a string")
            .not_blank("Alias must not be empty")
        )

        aliases_validator = (
            Validator[object]()
            .is_type(tuple, "ModelSpec.aliases must be a tuple")
            .each(alias_element)
        )

        # Valid cases
        assert aliases_validator.validate(()) == ()
        assert aliases_validator.validate(("gpt4", "gpt-4-turbo")) == ("gpt4", "gpt-4-turbo")

        # Invalid cases
        with pytest.raises(ValidationError, match="must be a tuple"):
            aliases_validator.validate(["gpt4"])  # List, not tuple

        with pytest.raises(ValidationError, match="Alias must be a string"):
            aliases_validator.validate((123,))  # Non-string element

        with pytest.raises(ValidationError, match="Alias must not be empty"):
            aliases_validator.validate(("gpt4", ""))  # Empty string element

    def test_routing_rules_models_validation_pattern(self) -> None:
        """Pattern from model_spec.py - validate models is non-empty tuple of ModelSpec."""
        from dataclasses import dataclass

        from rlm.domain.errors import ValidationError
        from rlm.domain.models.validation import Validator

        @dataclass
        class MockModelSpec:
            name: str

        spec_validator = Validator[object]().is_type(
            MockModelSpec,
            "Must be a ModelSpec instance",
        )

        models_validator = (
            Validator[object]()
            .is_type(tuple, "models must be a tuple")
            .not_empty("models must not be empty")
            .each(spec_validator)
        )

        # Valid case
        specs = (MockModelSpec("gpt-4"), MockModelSpec("claude-3"))
        assert models_validator.validate(specs) == specs

        # Invalid: not a tuple
        with pytest.raises(ValidationError, match="must be a tuple"):
            models_validator.validate([MockModelSpec("gpt-4")])

        # Invalid: empty tuple
        with pytest.raises(ValidationError, match="must not be empty"):
            models_validator.validate(())

        # Invalid: contains non-ModelSpec
        with pytest.raises(ValidationError, match="Must be a ModelSpec"):
            models_validator.validate((MockModelSpec("gpt-4"), "not a spec"))


class TestPrebuiltValidators:
    """Pre-built validator factories for common patterns."""

    def test_non_empty_string_validator(self) -> None:
        """non_empty_string() is a convenient factory."""
        from rlm.domain.errors import ValidationError
        from rlm.domain.models.validation import non_empty_string

        validator = non_empty_string("Field must be non-empty string")

        assert validator.validate("hello") == "hello"

        with pytest.raises(ValidationError):
            validator.validate(42)

        with pytest.raises(ValidationError):
            validator.validate("")

    def test_tuple_of_validator(self) -> None:
        """tuple_of() validates a tuple where all elements match a validator."""
        from rlm.domain.errors import ValidationError
        from rlm.domain.models.validation import non_empty_string, tuple_of

        validator = tuple_of(non_empty_string("Element error"))

        assert validator.validate(("a", "b")) == ("a", "b")

        with pytest.raises(ValidationError):
            validator.validate(["a", "b"])  # List, not tuple

        with pytest.raises(ValidationError):
            validator.validate(("a", ""))  # Empty element
