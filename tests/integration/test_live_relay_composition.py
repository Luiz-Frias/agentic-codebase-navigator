from __future__ import annotations

import importlib.util
import os

import pytest

from rlm.application.relay.root_composer import RootAgentComposer
from rlm.domain.relay import InMemoryPipelineRegistry, PipelineTemplate, StateSpec


@pytest.mark.integration
@pytest.mark.live_llm
def test_live_root_composer_selects_pipeline() -> None:
    if importlib.util.find_spec("openai") is None:
        pytest.skip("openai package not installed")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_MODEL") or "gpt-5-nano"
    base_url = os.environ.get("OPENAI_BASE_URL")

    from rlm.adapters.llm.openai import OpenAIAdapter

    llm = OpenAIAdapter(
        model=model,
        api_key=api_key,
        base_url=base_url,
        default_request_kwargs={
            "temperature": 0,
            "max_tokens": 16,
        },
    )

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

    composer = RootAgentComposer(registry=registry, llm=llm)
    pipeline = composer.compose("Select the alpha pipeline")
    assert pipeline.entry_state is not None
