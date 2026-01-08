from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlm.domain.models.serialization import serialize_value


@dataclass(slots=True, frozen=True)
class RunMetadata:
    """
    Metadata about a completion run.

    This mirrors the legacy `RLMMetadata` shape but lives in the domain layer.
    """

    root_model: str
    max_depth: int
    max_iterations: int
    backend: str
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    environment_type: str = "local"
    environment_kwargs: dict[str, Any] = field(default_factory=dict)
    other_backends: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_model": self.root_model,
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
            "backend": self.backend,
            "backend_kwargs": {k: serialize_value(v) for k, v in self.backend_kwargs.items()},
            "environment_type": self.environment_type,
            "environment_kwargs": {
                k: serialize_value(v) for k, v in self.environment_kwargs.items()
            },
            "other_backends": self.other_backends,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMetadata:
        return cls(
            root_model=str(data.get("root_model", "")),
            max_depth=int(data.get("max_depth", 0)),
            max_iterations=int(data.get("max_iterations", 0)),
            backend=str(data.get("backend", "")),
            backend_kwargs=dict(data.get("backend_kwargs", {}) or {}),
            environment_type=str(data.get("environment_type", "local")),
            environment_kwargs=dict(data.get("environment_kwargs", {}) or {}),
            other_backends=list(data.get("other_backends")) if data.get("other_backends") else None,
        )
