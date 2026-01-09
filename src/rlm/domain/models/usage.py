from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from rlm.domain.errors import ValidationError


def _as_non_negative_int(value: Any, field_name: str, /) -> int:
    """
    Coerce a value to a non-negative int for usage accounting.

    Notes:
    - `None` is treated as 0 (log/back-compat friendliness)
    - Negative values are rejected (domain invariant)
    """

    if value is None:
        return 0
    try:
        iv = int(value)
    except Exception as exc:  # noqa: BLE001 - boundary: normalize to ValidationError
        raise ValidationError(
            f"Invalid {field_name}: expected int-like value, got {type(value).__name__}"
        ) from exc
    if iv < 0:
        raise ValidationError(f"Invalid {field_name}: must be >= 0")
    return iv


@dataclass(slots=True)
class ModelUsageSummary:
    """Usage totals for a specific model."""

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def __post_init__(self) -> None:
        # Enforce invariants and normalize any int-like inputs.
        self.total_calls = _as_non_negative_int(self.total_calls, "total_calls")
        self.total_input_tokens = _as_non_negative_int(
            self.total_input_tokens, "total_input_tokens"
        )
        self.total_output_tokens = _as_non_negative_int(
            self.total_output_tokens, "total_output_tokens"
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelUsageSummary:
        return cls(
            total_calls=_as_non_negative_int(data.get("total_calls", 0), "total_calls"),
            total_input_tokens=_as_non_negative_int(
                data.get("total_input_tokens", 0), "total_input_tokens"
            ),
            total_output_tokens=_as_non_negative_int(
                data.get("total_output_tokens", 0), "total_output_tokens"
            ),
        )


@dataclass(slots=True)
class UsageSummary:
    """Aggregated usage totals across models."""

    model_usage_summaries: dict[str, ModelUsageSummary] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_usage_summaries, dict):
            raise ValidationError(
                "UsageSummary.model_usage_summaries must be a dict[str, ModelUsageSummary]"
            )
        for k, v in self.model_usage_summaries.items():
            if not isinstance(k, str) or not k.strip():
                raise ValidationError(
                    "UsageSummary.model_usage_summaries keys must be non-empty strings"
                )
            if not isinstance(v, ModelUsageSummary):
                raise ValidationError(
                    "UsageSummary.model_usage_summaries values must be ModelUsageSummary instances"
                )

    @property
    def total_calls(self) -> int:
        return sum(m.total_calls for m in self.model_usage_summaries.values())

    @property
    def total_input_tokens(self) -> int:
        return sum(m.total_input_tokens for m in self.model_usage_summaries.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(m.total_output_tokens for m in self.model_usage_summaries.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_usage_summaries": {
                model: self.model_usage_summaries[model].to_dict()
                for model in sorted(self.model_usage_summaries)
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageSummary:
        raw = data.get("model_usage_summaries", {}) or {}
        if not isinstance(raw, dict):
            raise TypeError(
                "UsageSummary.from_dict expects 'model_usage_summaries' to be a dict "
                f"(got {type(raw).__name__})"
            )
        return cls(
            model_usage_summaries={
                str(model): ModelUsageSummary.from_dict(summary or {})
                for model, summary in sorted(raw.items(), key=lambda kv: str(kv[0]))
            }
        )


def merge_usage_summaries(summaries: Iterable[UsageSummary], /) -> UsageSummary:
    """
    Deterministically merge usage summaries across models.

    Behavior:
    - Sums totals for the same model key across inputs.
    - Returns a new UsageSummary with keys inserted in sorted order.

    Notes:
    - The returned ModelUsageSummary objects are new instances (no aliasing).
    """
    totals: dict[str, ModelUsageSummary] = {}
    for summary in summaries:
        for model, mus in summary.model_usage_summaries.items():
            current = totals.get(model)
            if current is None:
                totals[model] = ModelUsageSummary(
                    total_calls=mus.total_calls,
                    total_input_tokens=mus.total_input_tokens,
                    total_output_tokens=mus.total_output_tokens,
                )
            else:
                current.total_calls += mus.total_calls
                current.total_input_tokens += mus.total_input_tokens
                current.total_output_tokens += mus.total_output_tokens

    return UsageSummary(model_usage_summaries={m: totals[m] for m in sorted(totals)})
