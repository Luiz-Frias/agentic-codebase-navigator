from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlm.domain.models.repl import ReplResult
from rlm.domain.models.serialization import serialize_value
from rlm.domain.models.usage import UsageSummary


@dataclass(slots=True)
class CodeBlock:
    """A fenced code block extracted from a model response, plus its execution result."""

    code: str
    result: ReplResult

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "result": self.result.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeBlock:
        return cls(
            code=str(data.get("code", "")),
            result=ReplResult.from_dict(data.get("result", {}) or {}),
        )


@dataclass(slots=True)
class Iteration:
    """A single orchestrator iteration step (prompt → response → optional code execution)."""

    prompt: Any
    response: str
    correlation_id: str | None = None
    code_blocks: list[CodeBlock] = field(default_factory=list)
    final_answer: str | None = None
    iteration_time: float = 0.0
    iteration_usage_summary: UsageSummary | None = None
    cumulative_usage_summary: UsageSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "prompt": serialize_value(self.prompt),
            "response": self.response,
            "code_blocks": [b.to_dict() for b in self.code_blocks],
            "final_answer": self.final_answer,
            "iteration_time": self.iteration_time,
        }
        if self.correlation_id is not None:
            d["correlation_id"] = self.correlation_id
        if self.iteration_usage_summary is not None:
            d["iteration_usage_summary"] = self.iteration_usage_summary.to_dict()
        if self.cumulative_usage_summary is not None:
            d["cumulative_usage_summary"] = self.cumulative_usage_summary.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Iteration:
        raw_blocks = data.get("code_blocks", []) or []
        correlation_id = data.get("correlation_id")
        raw_iter_usage = data.get("iteration_usage_summary")
        raw_cum_usage = data.get("cumulative_usage_summary")
        return cls(
            correlation_id=str(correlation_id) if correlation_id is not None else None,
            prompt=data.get("prompt"),
            response=str(data.get("response", "")),
            code_blocks=[CodeBlock.from_dict(b) for b in raw_blocks],
            final_answer=data.get("final_answer"),
            iteration_time=float(data.get("iteration_time", 0.0)),
            iteration_usage_summary=(
                UsageSummary.from_dict(raw_iter_usage) if isinstance(raw_iter_usage, dict) else None
            ),
            cumulative_usage_summary=(
                UsageSummary.from_dict(raw_cum_usage) if isinstance(raw_cum_usage, dict) else None
            ),
        )
