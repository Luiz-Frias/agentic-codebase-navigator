from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlm.domain.models.repl import ReplResult


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
    code_blocks: list[CodeBlock] = field(default_factory=list)
    final_answer: str | None = None
    iteration_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "code_blocks": [b.to_dict() for b in self.code_blocks],
            "final_answer": self.final_answer,
            "iteration_time": self.iteration_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Iteration:
        raw_blocks = data.get("code_blocks", []) or []
        return cls(
            prompt=data.get("prompt"),
            response=str(data.get("response", "")),
            code_blocks=[CodeBlock.from_dict(b) for b in raw_blocks],
            final_answer=data.get("final_answer"),
            iteration_time=float(data.get("iteration_time", 0.0)),
        )
