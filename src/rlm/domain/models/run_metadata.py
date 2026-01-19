from __future__ import annotations

from dataclasses import dataclass, field

from rlm.domain.models.serialization import SerializedValue, serialize_value


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
    backend_kwargs: dict[str, object] = field(default_factory=dict)
    environment_type: str = "local"
    environment_kwargs: dict[str, object] = field(default_factory=dict)
    other_backends: list[str] | None = None
    correlation_id: str | None = None

    def to_dict(self) -> dict[str, SerializedValue]:
        d: dict[str, SerializedValue] = {
            "root_model": self.root_model,
            "correlation_id": self.correlation_id,
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
            "backend": self.backend,
            "backend_kwargs": {k: serialize_value(v) for k, v in self.backend_kwargs.items()},
            "environment_type": self.environment_type,
            "environment_kwargs": {
                k: serialize_value(v) for k, v in self.environment_kwargs.items()
            },
            "other_backends": serialize_value(self.other_backends),
        }
        # Keep JSON payloads compact by omitting null correlation IDs.
        if d.get("correlation_id") is None:
            d.pop("correlation_id", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RunMetadata:
        """
        Create RunMetadata from dict.

        Type-driven boundary: accepts dict[str, object], validates internally.
        """
        # Narrow correlation_id
        raw_correlation_id = data.get("correlation_id")
        correlation_id = str(raw_correlation_id) if raw_correlation_id is not None else None

        # Narrow other_backends
        raw_other = data.get("other_backends")
        other_backends: list[str] | None
        if raw_other is None:
            other_backends = None
        elif isinstance(raw_other, (list, tuple)):
            other_backends = [str(x) for x in raw_other]
        else:
            # Back-compat: tolerate unexpected shapes by dropping the field.
            other_backends = None

        # Narrow root_model
        raw_root_model = data.get("root_model", "")
        root_model = str(raw_root_model) if raw_root_model else ""

        # Narrow max_depth
        raw_max_depth = data.get("max_depth", 0)
        max_depth: int
        if isinstance(raw_max_depth, int) and not isinstance(raw_max_depth, bool):
            max_depth = raw_max_depth
        elif isinstance(raw_max_depth, float):
            max_depth = int(raw_max_depth)
        else:
            max_depth = 0

        # Narrow max_iterations
        raw_max_iterations = data.get("max_iterations", 0)
        max_iterations: int
        if isinstance(raw_max_iterations, int) and not isinstance(raw_max_iterations, bool):
            max_iterations = raw_max_iterations
        elif isinstance(raw_max_iterations, float):
            max_iterations = int(raw_max_iterations)
        else:
            max_iterations = 0

        # Narrow backend
        raw_backend = data.get("backend", "")
        backend = str(raw_backend) if raw_backend else ""

        # Narrow backend_kwargs
        raw_backend_kwargs = data.get("backend_kwargs")
        backend_kwargs: dict[str, object] = (
            raw_backend_kwargs if isinstance(raw_backend_kwargs, dict) else {}
        )

        # Narrow environment_type
        raw_environment_type = data.get("environment_type", "local")
        environment_type = str(raw_environment_type) if raw_environment_type else "local"

        # Narrow environment_kwargs
        raw_environment_kwargs = data.get("environment_kwargs")
        environment_kwargs: dict[str, object] = (
            raw_environment_kwargs if isinstance(raw_environment_kwargs, dict) else {}
        )

        return cls(
            root_model=root_model,
            correlation_id=correlation_id,
            max_depth=max_depth,
            max_iterations=max_iterations,
            backend=backend,
            backend_kwargs=backend_kwargs,
            environment_type=environment_type,
            environment_kwargs=environment_kwargs,
            other_backends=other_backends,
        )
