"""
Model specification and routing rules.

This module defines the domain models for model selection and routing,
using the Validator pattern for clean, declarative validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from rlm.domain.errors import ValidationError
from rlm.domain.models.validation import Validator, non_empty_string

if TYPE_CHECKING:
    from collections.abc import Iterable, Sized


# ============================================================================
# Validators - declarative validation rules
# ============================================================================

_name_validator = non_empty_string("ModelSpec.name must be a non-empty string")

_alias_element_validator = non_empty_string(
    "ModelSpec.aliases must contain only non-empty strings",
)

_aliases_validator = (
    Validator[object]()
    .is_type(tuple, "ModelSpec.aliases must be a tuple of strings")
    .each(_alias_element_validator)  # type: ignore[arg-type]
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """
    Domain model describing a selectable model name and its aliases.

    Notes:
    - `name` is the canonical routing key used by the broker/LLM adapters.
    - `aliases` are alternate user-facing names that should resolve to `name`.
    - Exactly one `ModelSpec` in a set should be marked as `is_default=True`.

    """

    name: str
    aliases: tuple[str, ...] = ()
    is_default: bool = False

    def __post_init__(self) -> None:
        _name_validator.validate(self.name)
        _aliases_validator.validate(self.aliases)


# ============================================================================
# ModelRoutingRules validators
# ============================================================================

_spec_validator = Validator[object]().is_type(
    ModelSpec,
    "ModelRoutingRules.models must contain only ModelSpec instances",
)


def _is_non_empty_tuple(x: object) -> bool:
    """Check if value is a non-empty tuple (for Validator predicate)."""
    return len(cast("Sized", x)) > 0


_models_validator = (
    Validator[object]()
    .is_type(tuple, "ModelRoutingRules.models must be a non-empty tuple[ModelSpec, ...]")
    .satisfies(
        _is_non_empty_tuple,
        "ModelRoutingRules.models must be a non-empty tuple[ModelSpec, ...]",
    )
    .each(_spec_validator)  # type: ignore[arg-type]
)


@dataclass(frozen=True, slots=True)
class ModelRoutingRules:
    """
    Routing rules for model selection.

    Rules:
    - If no model is requested: use `default_model`.
    - If requested model is allowed (including aliases): use the resolved canonical name.
    - If requested model is not allowed:
      - If `fallback_model` is set: use that
      - Else: raise ValidationError
    """

    models: tuple[ModelSpec, ...]
    fallback_model: str | None = None

    # Cached lookup built at init time (kept immutable for thread safety).
    _lookup: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _default_model: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        # Validate models tuple structure
        _models_validator.validate(self.models)

        # Build lookup table and find default
        lookup: dict[str, str] = {}
        default: str | None = None

        for spec in self.models:
            # Canonical name mapping
            if spec.name in lookup and lookup[spec.name] != spec.name:
                raise ValidationError(f"Duplicate model name: {spec.name!r}")
            lookup[spec.name] = spec.name

            # Alias mapping
            for alias in spec.aliases:
                if alias in lookup and lookup[alias] != spec.name:
                    raise ValidationError(f"Alias {alias!r} is ambiguous across models")
                lookup[alias] = spec.name

            # Track default
            if spec.is_default:
                if default is not None and default != spec.name:
                    raise ValidationError("Exactly one ModelSpec must have is_default=True")
                default = spec.name

        if default is None:
            raise ValidationError(
                "ModelRoutingRules requires exactly one default ModelSpec (is_default=True)",
            )

        # Validate fallback_model if provided
        if self.fallback_model is not None:
            non_empty_string(
                "ModelRoutingRules.fallback_model must be a non-empty string when provided",
            ).validate(self.fallback_model)
            if self.fallback_model not in lookup:
                raise ValidationError(
                    f"ModelRoutingRules.fallback_model {self.fallback_model!r} is not in allowed models",
                )

        object.__setattr__(self, "_lookup", lookup)
        object.__setattr__(self, "_default_model", default)

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def allowed_models(self) -> set[str]:
        # Only canonical names (not aliases).
        return {spec.name for spec in self.models}

    def resolve(self, requested_model: str | None, /) -> str:
        """
        Resolve a requested model name (or alias) to a canonical model name.

        Raises ValidationError if the model is not allowed and no fallback is set.
        """
        if requested_model is None:
            return self._default_model

        # Validate requested_model is a string
        Validator[object]().is_type(str, "Requested model must be a string or None").validate(
            requested_model,
        )

        model = requested_model.strip()
        if not model:
            return self._default_model

        resolved = self._lookup.get(model)
        if resolved is not None:
            return resolved

        if self.fallback_model is not None:
            return self.fallback_model

        raise ValidationError(
            f"Unknown model {requested_model!r}. Allowed: {sorted(self.allowed_models)}",
        )


def build_routing_rules(
    specs: Iterable[ModelSpec],
    /,
    *,
    fallback_model: str | None = None,
) -> ModelRoutingRules:
    """
    Convenience builder to construct routing rules from any iterable.

    Keeps call sites simple when building from config.
    """
    return ModelRoutingRules(models=tuple(specs), fallback_model=fallback_model)
