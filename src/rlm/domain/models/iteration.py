from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rlm.domain.models.repl import ReplResult
from rlm.domain.models.serialization import SerializedValue, serialize_value
from rlm.domain.models.usage import UsageSummary

if TYPE_CHECKING:
    from rlm.domain.types import Prompt


@dataclass(slots=True)
class CodeBlock:
    """A fenced code block extracted from a model response, plus its execution result."""

    code: str
    result: ReplResult

    def to_dict(self) -> dict[str, SerializedValue]:
        return {"code": self.code, "result": serialize_value(self.result.to_dict())}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CodeBlock:
        """
        Create CodeBlock from dict.

        Type-driven boundary: accepts dict[str, object], validates internally.
        """
        # Narrow code to str
        raw_code = data.get("code", "")
        code = str(raw_code) if raw_code else ""

        # Narrow result to dict
        raw_result = data.get("result")
        result_dict: dict[str, object] = raw_result if isinstance(raw_result, dict) else {}

        return cls(
            code=code,
            result=ReplResult.from_dict(result_dict),
        )


@dataclass(slots=True)
class Iteration:
    """A single orchestrator iteration step (prompt → response → optional code execution)."""

    prompt: Prompt
    response: str
    correlation_id: str | None = None
    code_blocks: list[CodeBlock] = field(default_factory=list)
    final_answer: str | None = None
    iteration_time: float = 0.0
    iteration_usage_summary: UsageSummary | None = None
    cumulative_usage_summary: UsageSummary | None = None

    def to_dict(self) -> dict[str, SerializedValue]:
        d: dict[str, SerializedValue] = {
            "prompt": serialize_value(self.prompt),
            "response": self.response,
            "code_blocks": [serialize_value(b.to_dict()) for b in self.code_blocks],
            "final_answer": self.final_answer,
            "iteration_time": self.iteration_time,
        }
        if self.correlation_id is not None:
            d["correlation_id"] = self.correlation_id
        if self.iteration_usage_summary is not None:
            d["iteration_usage_summary"] = serialize_value(self.iteration_usage_summary.to_dict())
        if self.cumulative_usage_summary is not None:
            d["cumulative_usage_summary"] = serialize_value(self.cumulative_usage_summary.to_dict())
        return d

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Iteration:
        """
        Create Iteration from dict.

        Type-driven boundary: accepts dict[str, object], validates internally.
        """
        # Narrow code_blocks to list
        raw_blocks = data.get("code_blocks")
        blocks_list: list[object] = raw_blocks if isinstance(raw_blocks, list) else []

        # Narrow correlation_id to str | None
        raw_correlation_id = data.get("correlation_id")
        correlation_id = str(raw_correlation_id) if raw_correlation_id is not None else None

        # Narrow usage summaries to dict | None
        raw_iter_usage = data.get("iteration_usage_summary")
        raw_cum_usage = data.get("cumulative_usage_summary")

        # Narrow prompt - follows Prompt type (str | dict | list)
        raw_prompt = data.get("prompt")
        prompt: Prompt
        if isinstance(raw_prompt, (str, dict, list)):
            prompt = raw_prompt
        else:
            prompt = str(raw_prompt) if raw_prompt is not None else ""

        # Narrow response to str
        raw_response = data.get("response", "")
        response = str(raw_response) if raw_response else ""

        # Narrow final_answer to str | None
        raw_final_answer = data.get("final_answer")
        final_answer = str(raw_final_answer) if raw_final_answer is not None else None

        # Narrow iteration_time to float
        raw_iteration_time = data.get("iteration_time", 0.0)
        iteration_time: float
        if isinstance(raw_iteration_time, (int, float)):
            iteration_time = float(raw_iteration_time)
        else:
            iteration_time = 0.0

        return cls(
            correlation_id=correlation_id,
            prompt=prompt,
            response=response,
            code_blocks=[CodeBlock.from_dict(b) for b in blocks_list if isinstance(b, dict)],
            final_answer=final_answer,
            iteration_time=iteration_time,
            iteration_usage_summary=(
                UsageSummary.from_dict(raw_iter_usage) if isinstance(raw_iter_usage, dict) else None
            ),
            cumulative_usage_summary=(
                UsageSummary.from_dict(raw_cum_usage) if isinstance(raw_cum_usage, dict) else None
            ),
        )
