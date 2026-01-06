from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rlm.domain.models.serialization import serialize_value
from rlm.domain.models.usage import UsageSummary


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
        return {
            "root_model": self.root_model,
            "prompt": serialize_value(self.prompt),
            "response": self.response,
            "usage_summary": self.usage_summary.to_dict(),
            "execution_time": self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatCompletion:
        return cls(
            root_model=str(data.get("root_model", "")),
            prompt=data.get("prompt"),
            response=str(data.get("response", "")),
            usage_summary=UsageSummary.from_dict(data.get("usage_summary", {}) or {}),
            execution_time=float(data.get("execution_time", 0.0)),
        )
