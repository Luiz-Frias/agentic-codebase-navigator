from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlm.domain.models.completion import ChatCompletion
from rlm.domain.models.serialization import serialize_value


@dataclass(slots=True)
class ReplResult:
    """Result of executing a code block in an environment."""

    stdout: str = ""
    stderr: str = ""
    locals: dict[str, Any] = field(default_factory=dict)
    llm_calls: list[ChatCompletion] = field(default_factory=list)
    execution_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "locals": {k: serialize_value(v) for k, v in self.locals.items()},
            "execution_time": self.execution_time,
            "llm_calls": [c.to_dict() for c in self.llm_calls],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplResult:
        raw_calls = data.get("llm_calls", []) or []
        return cls(
            stdout=str(data.get("stdout", "")),
            stderr=str(data.get("stderr", "")),
            locals=dict(data.get("locals", {}) or {}),
            llm_calls=[ChatCompletion.from_dict(c) for c in raw_calls],
            execution_time=float(data.get("execution_time", 0.0)),
        )
