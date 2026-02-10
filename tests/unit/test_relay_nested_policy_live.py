from __future__ import annotations

import pytest

from rlm.adapters.relay import FunctionStateExecutor
from rlm.application.relay.root_composer import RootAgentComposer
from rlm.domain.errors import LLMError
from rlm.domain.relay import (
    InMemoryPipelineRegistry,
    PipelineTemplate,
    StateSpec,
    has_pydantic,
)
from tests.live_llm import LiveLLMSettings


@pytest.mark.unit
@pytest.mark.live_llm
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_root_composer_chains_pipelines_with_live_llm(
    live_llm_settings: LiveLLMSettings | None,
) -> None:
    if live_llm_settings is None:
        pytest.skip("Live LLM tests disabled")

    llm = live_llm_settings.build_openai_adapter(
        request_kwargs={
            "temperature": 0,
            "max_tokens": 64,
        }
    )

    registry = InMemoryPipelineRegistry()

    def _build_pipeline(label: str):
        start = StateSpec(
            name=f"{label}_start",
            input_type=str,
            output_type=str,
            executor=FunctionStateExecutor(lambda text, l=label: f"{l}:{text}"),
        )
        end = StateSpec(
            name=f"{label}_end",
            input_type=str,
            output_type=str,
            executor=FunctionStateExecutor(lambda text, l=label: f"{text}:{l}"),
        )
        return start >> end

    registry.register(
        PipelineTemplate(
            name="alpha",
            description="Alpha pipeline for chaining.",
            input_type=str,
            output_type=str,
            factory=lambda: _build_pipeline("alpha"),
        )
    )
    registry.register(
        PipelineTemplate(
            name="beta",
            description="Beta pipeline for chaining.",
            input_type=str,
            output_type=str,
            factory=lambda: _build_pipeline("beta"),
        )
    )

    composer = RootAgentComposer(registry=registry, llm=llm)
    try:
        pipeline = composer.compose("Use both alpha and beta pipelines in order.")
    except LLMError as exc:
        pytest.skip(str(exc))

    state_names = {state.name for state in pipeline.states}
    assert "alpha_start" in state_names
    assert "beta_start" in state_names

    edges = {(edge.from_state.name, edge.to_state.name) for edge in pipeline.edges}
    assert ("alpha_end", "beta_start") in edges
