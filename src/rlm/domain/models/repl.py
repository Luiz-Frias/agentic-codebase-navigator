from __future__ import annotations

from dataclasses import dataclass, field

from rlm.domain.models.completion import ChatCompletion
from rlm.domain.models.serialization import SerializedValue, serialize_value


@dataclass(slots=True)
class ReplResult:
    """Result of executing a code block in an environment."""

    correlation_id: str | None = None
    stdout: str = ""
    stderr: str = ""
    locals: dict[str, object] = field(default_factory=dict)
    llm_calls: list[ChatCompletion] = field(default_factory=list)
    execution_time: float = 0.0

    def to_dict(self) -> dict[str, SerializedValue]:
        d: dict[str, SerializedValue] = {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "locals": {k: serialize_value(v) for k, v in self.locals.items()},
            "execution_time": self.execution_time,
            # Keep upstream key name `rlm_calls` for log/schema compatibility.
            "rlm_calls": [serialize_value(c.to_dict()) for c in self.llm_calls],
        }
        if self.correlation_id is not None:
            d["correlation_id"] = self.correlation_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ReplResult:
        """
        Create ReplResult from dict.

        Type-driven boundary: accepts dict[str, object], validates internally.
        """
        # Back-compat: accept either key name (canonical: `rlm_calls`).
        raw_calls: list[object]
        if "rlm_calls" in data:
            raw_rlm = data.get("rlm_calls")
            raw_calls = raw_rlm if isinstance(raw_rlm, list) else []
        else:
            raw_llm = data.get("llm_calls")
            raw_calls = raw_llm if isinstance(raw_llm, list) else []

        # Narrow correlation_id
        raw_correlation_id = data.get("correlation_id")
        correlation_id = str(raw_correlation_id) if raw_correlation_id is not None else None

        # Narrow stdout and stderr
        raw_stdout = data.get("stdout", "")
        raw_stderr = data.get("stderr", "")
        stdout = str(raw_stdout) if raw_stdout else ""
        stderr = str(raw_stderr) if raw_stderr else ""

        # Narrow locals to dict
        raw_locals = data.get("locals")
        locals_dict: dict[str, object] = raw_locals if isinstance(raw_locals, dict) else {}

        # Narrow execution_time
        raw_execution_time = data.get("execution_time", 0.0)
        execution_time: float
        if isinstance(raw_execution_time, (int, float)):
            execution_time = float(raw_execution_time)
        else:
            execution_time = 0.0

        return cls(
            correlation_id=correlation_id,
            stdout=stdout,
            stderr=stderr,
            locals=locals_dict,
            llm_calls=[ChatCompletion.from_dict(c) for c in raw_calls if isinstance(c, dict)],
            execution_time=execution_time,
        )
