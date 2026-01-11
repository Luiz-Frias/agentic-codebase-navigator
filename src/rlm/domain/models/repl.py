from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlm.domain.models.completion import ChatCompletion
from rlm.domain.models.serialization import serialize_value


@dataclass(slots=True)
class ReplResult:
    """Result of executing a code block in an environment."""

    correlation_id: str | None = None
    stdout: str = ""
    stderr: str = ""
    locals: dict[str, Any] = field(default_factory=dict)
    llm_calls: list[ChatCompletion] = field(default_factory=list)
    execution_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "locals": {k: serialize_value(v) for k, v in self.locals.items()},
            "execution_time": self.execution_time,
            # Keep upstream key name `rlm_calls` for log/schema compatibility.
            "rlm_calls": [c.to_dict() for c in self.llm_calls],
        }
        if self.correlation_id is not None:
            d["correlation_id"] = self.correlation_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplResult:
        # Back-compat: accept either key name (canonical: `rlm_calls`).
        if "rlm_calls" in data:
            raw_calls = data.get("rlm_calls") or []
        else:
            raw_calls = data.get("llm_calls") or []
        correlation_id = data.get("correlation_id")
        return cls(
            correlation_id=str(correlation_id) if correlation_id is not None else None,
            stdout=str(data.get("stdout", "")),
            stderr=str(data.get("stderr", "")),
            locals=dict(data.get("locals", {}) or {}),
            llm_calls=[ChatCompletion.from_dict(c) for c in raw_calls],
            execution_time=float(data.get("execution_time", 0.0)),
        )
