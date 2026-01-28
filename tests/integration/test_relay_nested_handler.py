from __future__ import annotations

import pytest

from rlm.adapters.relay import FunctionStateExecutor, RelayNestedCallHandler
from rlm.application.relay.root_composer import RootAgentComposer
from rlm.domain.models import ChatCompletion
from rlm.domain.models.usage import ModelUsageSummary, UsageSummary
from rlm.domain.relay import InMemoryPipelineRegistry, PipelineTemplate, StateSpec


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


@pytest.mark.integration
def test_relay_nested_call_handler_executes_pipeline() -> None:
    registry = InMemoryPipelineRegistry()
    start = StateSpec(
        name="start",
        input_type=str,
        output_type=str,
        executor=FunctionStateExecutor(lambda text: text.upper()),
    )
    end = StateSpec(
        name="end",
        input_type=str,
        output_type=str,
        executor=FunctionStateExecutor(lambda text: f"{text}-DONE"),
    )
    pipeline = start >> end
    registry.register(
        PipelineTemplate(
            name="alpha",
            description="Alpha pipeline",
            input_type=str,
            output_type=str,
            factory=lambda: pipeline,
        )
    )

    composer = RootAgentComposer(registry=registry, llm=FakeLLM())
    handler = RelayNestedCallHandler(registry=registry, composer=composer, max_depth=2)

    response = handler.handle("alpha task", depth=0, correlation_id=None, model=None)
    assert response.handled
    assert response.response == "ALPHA TASK-DONE"
