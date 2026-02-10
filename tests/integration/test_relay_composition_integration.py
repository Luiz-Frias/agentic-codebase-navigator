from __future__ import annotations

import pytest

from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.application.relay.root_composer import RootAgentComposer
from rlm.domain.relay import InMemoryPipelineRegistry, PipelineTemplate, StateSpec
from tests.live_llm import LiveLLMSettings


@pytest.mark.integration
def test_root_agent_composes_subpipeline_from_registry(
    live_llm_settings: LiveLLMSettings | None,
) -> None:
    registry = InMemoryPipelineRegistry()
    registry.register(
        PipelineTemplate(
            name="alpha",
            description="Use for alpha tasks.",
            input_type=str,
            output_type=str,
            factory=lambda: StateSpec(name="a", input_type=str, output_type=str)
            >> StateSpec(name="b", input_type=str, output_type=str),
        )
    )
    registry.register(
        PipelineTemplate(
            name="beta",
            description="Use for beta tasks.",
            input_type=str,
            output_type=str,
            factory=lambda: StateSpec(name="c", input_type=str, output_type=str)
            >> StateSpec(name="d", input_type=str, output_type=str),
        )
    )

    if live_llm_settings is not None:
        llm = live_llm_settings.build_openai_adapter(
            request_kwargs={"temperature": 0, "max_tokens": 64}
        )
    else:
        llm = MockLLMAdapter(model="mock", script=["alpha"])

    composer = RootAgentComposer(registry=registry, llm=llm)
    pipeline = composer.compose("Select the alpha pipeline")
    assert pipeline.entry_state is not None
