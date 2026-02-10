from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TraceEntry:
    state_name: str
    status: str
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PipelineTrace:
    entries: tuple[TraceEntry, ...] = field(default_factory=tuple)

    def add(self, entry: TraceEntry) -> PipelineTrace:
        return PipelineTrace(entries=(*self.entries, entry))
