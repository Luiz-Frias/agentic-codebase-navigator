from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlm.domain.errors import ValidationError

if TYPE_CHECKING:
    from rlm.domain.relay.pipeline import Pipeline
    from rlm.domain.relay.registry import PipelineRegistry


@dataclass(frozen=True, slots=True)
class WorkflowSeed:
    entry_pipeline: str
    fallback_pipelines: tuple[str, ...] = ()

    def resolve(self, registry: PipelineRegistry, /) -> Pipeline:
        candidates = registry.list()
        by_name = {template.name: template for template in candidates}
        for name in (self.entry_pipeline, *self.fallback_pipelines):
            template = by_name.get(name)
            if template is not None:
                return template.factory()
        raise ValidationError(f"Unknown pipeline: {self.entry_pipeline}")


@dataclass(frozen=True, slots=True)
class ComposablePipeline[InputT, OutputT]:
    pipeline: Pipeline
    input_type: type[InputT]
    output_type: type[OutputT]
