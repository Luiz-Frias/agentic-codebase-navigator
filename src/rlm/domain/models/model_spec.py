from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

from rlm.domain.errors import ValidationError


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Domain model describing a selectable model name and its aliases.

    Notes:
    - `name` is the canonical routing key used by the broker/LLM adapters.
    - `aliases` are alternate user-facing names that should resolve to `name`.
    - Exactly one `ModelSpec` in a set should be marked as `is_default=True`.

    """

    name: str
    aliases: tuple[str, ...] = ()
    is_default: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValidationError("ModelSpec.name must be a non-empty string")
        if not isinstance(self.aliases, tuple):
            raise ValidationError("ModelSpec.aliases must be a tuple of strings")
        for a in self.aliases:
            if not isinstance(a, str) or not a.strip():
                raise ValidationError("ModelSpec.aliases must contain only non-empty strings")


@dataclass(frozen=True, slots=True)
class ModelRoutingRules:
    """Routing rules for model selection.

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
        if not isinstance(self.models, tuple) or not self.models:
            raise ValidationError(
                "ModelRoutingRules.models must be a non-empty tuple[ModelSpec, ...]",
            )

        lookup: dict[str, str] = {}
        default: str | None = None

        for spec in self.models:
            if not isinstance(spec, ModelSpec):
                raise ValidationError(
                    "ModelRoutingRules.models must contain only ModelSpec instances",
                )

            # Canonical name mapping.
            if spec.name in lookup and lookup[spec.name] != spec.name:
                raise ValidationError(f"Duplicate model name: {spec.name!r}")
            lookup[spec.name] = spec.name

            # Alias mapping.
            for alias in spec.aliases:
                if alias in lookup and lookup[alias] != spec.name:
                    raise ValidationError(f"Alias {alias!r} is ambiguous across models")
                lookup[alias] = spec.name

            if spec.is_default:
                if default is not None and default != spec.name:
                    raise ValidationError("Exactly one ModelSpec must have is_default=True")
                default = spec.name

        if default is None:
            raise ValidationError(
                "ModelRoutingRules requires exactly one default ModelSpec (is_default=True)",
            )

        if self.fallback_model is not None:
            if not isinstance(self.fallback_model, str) or not self.fallback_model.strip():
                raise ValidationError(
                    "ModelRoutingRules.fallback_model must be a non-empty string when provided",
                )
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
        """Resolve a requested model name (or alias) to a canonical model name.

        Raises ValidationError if the model is not allowed and no fallback is set.
        """
        if requested_model is None:
            return self._default_model
        if not isinstance(requested_model, str):
            raise ValidationError("Requested model must be a string or None")
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
    """Convenience builder to construct routing rules from any iterable.

    Keeps call sites simple when building from config.
    """
    return ModelRoutingRules(models=tuple(specs), fallback_model=fallback_model)
