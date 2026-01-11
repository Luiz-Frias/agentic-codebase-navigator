from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rlm.domain.models.serialization import serialize_value
from rlm.domain.models.usage import ModelUsageSummary, UsageSummary


@dataclass(slots=True)
class ChatCompletion:
    """
    A single LLM call result.

    Mirrors the shape of the legacy `RLMChatCompletion`, but is dependency-free
    and owned by the domain layer.
    """

    root_model: str
    prompt: Any
    response: str
    usage_summary: UsageSummary
    execution_time: float

    def to_dict(self) -> dict[str, Any]:
        mus = self.usage_summary.model_usage_summaries.get(self.root_model)
        prompt_tokens = mus.total_input_tokens if mus is not None else 0
        completion_tokens = mus.total_output_tokens if mus is not None else 0
        return {
            "root_model": self.root_model,
            "prompt": serialize_value(self.prompt),
            "response": self.response,
            "usage_summary": self.usage_summary.to_dict(),
            # Back-compat / visualizer convenience: legacy schema expects flat token counts.
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "execution_time": self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatCompletion:
        raw_usage = data.get("usage_summary")
        usage = (
            UsageSummary.from_dict(raw_usage)
            if isinstance(raw_usage, dict)
            else UsageSummary(model_usage_summaries={})
        )
        # Back-compat: legacy logs may store flat token counts instead of a UsageSummary.
        if not usage.model_usage_summaries:
            pt = data.get("prompt_tokens")
            ct = data.get("completion_tokens")
            if isinstance(pt, (int, float)) and isinstance(ct, (int, float)):
                model = str(data.get("root_model", "") or "unknown")
                usage = UsageSummary(
                    model_usage_summaries={
                        model: ModelUsageSummary(
                            total_calls=1,
                            total_input_tokens=int(pt),
                            total_output_tokens=int(ct),
                        )
                    }
                )
        return cls(
            root_model=str(data.get("root_model", "")),
            prompt=data.get("prompt"),
            response=str(data.get("response", "")),
            usage_summary=usage,
            execution_time=float(data.get("execution_time", 0.0)),
        )
