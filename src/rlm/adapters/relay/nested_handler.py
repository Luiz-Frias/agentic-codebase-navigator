from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from rlm.adapters.relay.states.pipeline_state import SyncPipelineStateExecutor
from rlm.domain.agent_ports import NestedCallPolicy, NestedConfig
from rlm.domain.errors import ValidationError
from rlm.domain.models.nested_call import NestedCallResponse
from rlm.domain.models.result import Err
from rlm.domain.ports import NestedCallHandlerPort
from rlm.domain.relay.baton import Baton
from rlm.domain.relay.errors import StateError

if TYPE_CHECKING:
    from rlm.domain.relay.pipeline import Pipeline
    from rlm.domain.relay.registry import PipelineRegistry
    from rlm.domain.types import Prompt


class ComposerPort(Protocol):
    def compose(self, task: Prompt, /) -> Pipeline: ...


@dataclass(frozen=True, slots=True)
class RelayNestedCallHandler(NestedCallHandlerPort, NestedCallPolicy):
    registry: PipelineRegistry
    composer: ComposerPort
    max_depth: int = 3

    def should_orchestrate(self, prompt: str, depth: int) -> bool:
        if depth >= self.max_depth:
            return False
        return bool(self.registry.search(str(prompt)))

    def get_nested_config(self) -> NestedConfig:
        return NestedConfig(max_depth=self.max_depth)

    def handle(
        self,
        prompt: Prompt,
        /,
        *,
        depth: int,
        correlation_id: str | None,
        model: str | None,
    ) -> NestedCallResponse:
        _ = (correlation_id, model)
        if not self.should_orchestrate(str(prompt), depth):
            return NestedCallResponse.not_handled()

        candidates = self.registry.search(str(prompt))
        if not candidates:
            return NestedCallResponse.not_handled()

        try:
            pipeline = self.composer.compose(prompt)
        except ValidationError:
            return NestedCallResponse.not_handled()

        return self._run_pipeline(pipeline, prompt)

    def _run_pipeline(self, pipeline: Pipeline, prompt: Prompt) -> NestedCallResponse:
        if pipeline.entry_state is None:
            return NestedCallResponse.not_handled()
        terminals = pipeline.terminal_states
        if len(terminals) != 1:
            return NestedCallResponse.not_handled()

        input_payload: object
        if pipeline.entry_state.input_type is str:
            input_payload = str(prompt)
        else:
            input_payload = {"prompt": prompt}

        result = Baton.create(input_payload, pipeline.entry_state.input_type)
        if isinstance(result, Err):
            return NestedCallResponse.not_handled()

        executor: SyncPipelineStateExecutor[object, object] = SyncPipelineStateExecutor(pipeline)
        state = pipeline.as_state(
            name="nested_pipeline",
            input_type=pipeline.entry_state.input_type,
            output_type=terminals[0].output_type,
            executor=executor,
        )
        outcome = executor.execute(state, result.value)
        if isinstance(outcome, Err):
            error = outcome.error
            if isinstance(error, StateError):
                return NestedCallResponse.handled_response(f"Error: {error.message}")
            return NestedCallResponse.handled_response("Error: Nested pipeline failed")

        return NestedCallResponse.handled_response(str(outcome.value.payload))
