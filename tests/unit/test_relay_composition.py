from __future__ import annotations

import pytest

from rlm.adapters.relay import FunctionStateExecutor
from rlm.adapters.relay.states import SyncPipelineStateExecutor
from rlm.domain.errors import ValidationError
from rlm.domain.models import ChatCompletion
from rlm.domain.models.result import Ok
from rlm.domain.models.usage import ModelUsageSummary, UsageSummary
from rlm.domain.relay import (
    Baton,
    InMemoryPipelineRegistry,
    PipelineTemplate,
    StateSpec,
    WorkflowSeed,
    has_pydantic,
)


@pytest.mark.unit
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_pipeline_registry_search() -> None:
    registry = InMemoryPipelineRegistry()
    template = PipelineTemplate(
        name="alpha",
        description="Alpha pipeline",
        input_type=int,
        output_type=int,
        factory=lambda: StateSpec(name="s", input_type=int, output_type=int) >> StateSpec(
            name="t",
            input_type=int,
            output_type=int,
        ),
    )
    registry.register(template)

    matches = registry.search("alpha")
    assert matches and matches[0].name == "alpha"


@pytest.mark.unit
def test_workflow_seed_resolve() -> None:
    registry = InMemoryPipelineRegistry()
    registry.register(
        PipelineTemplate(
            name="primary",
            description="Primary pipeline",
            input_type=str,
            output_type=str,
            factory=lambda: StateSpec(name="s", input_type=str, output_type=str)
            >> StateSpec(name="t", input_type=str, output_type=str),
        )
    )
    seed = WorkflowSeed(entry_pipeline="primary")
    pipeline = seed.resolve(registry)
    assert pipeline is not None


@pytest.mark.unit
def test_pipeline_as_state_executes_pipeline() -> None:
    s1 = StateSpec(
        name="s1",
        input_type=int,
        output_type=int,
        executor=FunctionStateExecutor(lambda x: x + 1),
    )
    s2 = StateSpec(
        name="s2",
        input_type=int,
        output_type=int,
        executor=FunctionStateExecutor(lambda x: x * 2),
    )
    pipeline = s1 >> s2

    state = pipeline.as_state(
        name="pipe",
        input_type=int,
        output_type=int,
        executor=SyncPipelineStateExecutor(pipeline),
    )

    baton = Baton.create(1, int).unwrap()
    result = state.executor.execute(state, baton)
    assert isinstance(result, Ok)
    assert result.value.payload == 4


@pytest.mark.unit
def test_pipeline_chain_type_mismatch() -> None:
    from rlm.application.relay.root_composer import _merge_pipeline

    left = StateSpec(name="left", input_type=int, output_type=int)
    right = StateSpec(name="right", input_type=str, output_type=str)
    pipeline_left = left >> StateSpec(name="left_end", input_type=int, output_type=int)
    pipeline_right = right >> StateSpec(name="right_end", input_type=str, output_type=str)

    with pytest.raises(ValidationError):
        _merge_pipeline(pipeline_left, pipeline_right)


@pytest.mark.unit
def test_root_composer_selects_pipeline() -> None:
    from rlm.application.relay.root_composer import RootAgentComposer

    class FakeLLM:
        model_name = "fake"

        def complete(self, request, /):
            usage = UsageSummary(
                model_usage_summaries={
                    "fake": ModelUsageSummary(
                        total_calls=1,
                        total_input_tokens=1,
                        total_output_tokens=1,
                    )
                }
            )
            return ChatCompletion(
                root_model="fake",
                prompt=request.prompt,
                response="alpha",
                usage_summary=usage,
                execution_time=0.0,
            )

        async def acomplete(self, request, /):
            return self.complete(request)

        def get_usage_summary(self):
            return UsageSummary(model_usage_summaries={})

        def get_last_usage(self):
            return UsageSummary(model_usage_summaries={})

    registry = InMemoryPipelineRegistry()
    registry.register(
        PipelineTemplate(
            name="alpha",
            description="Alpha pipeline",
            input_type=str,
            output_type=str,
            factory=lambda: StateSpec(name="a", input_type=str, output_type=str)
            >> StateSpec(name="b", input_type=str, output_type=str),
        )
    )
    composer = RootAgentComposer(registry=registry, llm=FakeLLM())
    pipeline = composer.compose("do alpha")
    assert pipeline.entry_state is not None
