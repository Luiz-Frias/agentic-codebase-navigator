from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from rlm.domain.relay.pipeline import Pipeline


@dataclass(frozen=True, slots=True)
class PipelineTemplate[InputT, OutputT]:
    name: str
    description: str
    input_type: type[InputT]
    output_type: type[OutputT]
    factory: Callable[[], Pipeline]
    tags: tuple[str, ...] = ()


class PipelineRegistry(Protocol):
    def register(self, template: PipelineTemplate[object, object], /) -> None: ...

    def list(self) -> tuple[PipelineTemplate[object, object], ...]: ...

    def search(self, query: str, /) -> tuple[PipelineTemplate[object, object], ...]: ...


@dataclass(slots=True)
class InMemoryPipelineRegistry(PipelineRegistry):
    _templates: dict[str, PipelineTemplate[object, object]] = field(default_factory=dict)

    def register(self, template: PipelineTemplate[object, object], /) -> None:
        self._templates[template.name] = template

    def list(self) -> tuple[PipelineTemplate[object, object], ...]:
        return tuple(self._templates.values())

    def search(self, query: str, /) -> tuple[PipelineTemplate[object, object], ...]:
        needle = query.strip().lower()
        if not needle:
            return self.list()
        tokens = tuple(part for part in needle.split() if part)
        matches: list[PipelineTemplate[object, object]] = []
        for template in self._templates.values():
            haystacks = (template.name, template.description, " ".join(template.tags))
            if any(token in text.lower() for token in tokens for text in haystacks if text):
                matches.append(template)
        return tuple(matches)
