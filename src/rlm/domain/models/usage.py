from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ModelUsageSummary:
    """Usage totals for a specific model."""

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelUsageSummary:
        return cls(
            total_calls=int(data.get("total_calls", 0)),
            total_input_tokens=int(data.get("total_input_tokens", 0)),
            total_output_tokens=int(data.get("total_output_tokens", 0)),
        )


@dataclass(slots=True)
class UsageSummary:
    """Aggregated usage totals across models."""

    model_usage_summaries: dict[str, ModelUsageSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_usage_summaries": {
                model: summary.to_dict() for model, summary in self.model_usage_summaries.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageSummary:
        raw = data.get("model_usage_summaries", {}) or {}
        return cls(
            model_usage_summaries={
                str(model): ModelUsageSummary.from_dict(summary) for model, summary in raw.items()
            }
        )
