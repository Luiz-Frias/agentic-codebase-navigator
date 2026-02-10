from __future__ import annotations

import pytest

from rlm.adapters.relay import FunctionStateExecutor, SyncPipelineExecutor
from rlm.adapters.relay.states.pipeline_state import SyncPipelineStateExecutor
from rlm.domain.models import LLMRequest
from rlm.domain.models.result import Ok
from rlm.domain.relay import Baton, Pipeline, StateSpec, has_pydantic
from tests.live_llm import LiveLLMSettings

if has_pydantic():
    from pydantic import BaseModel
else:
    BaseModel = object  # type: ignore[assignment]


@pytest.mark.e2e
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_multi_level_pipeline_composition_with_typed_baton_flow(
    live_llm_settings: LiveLLMSettings | None,
) -> None:
    class Seed(BaseModel):
        text: str

    class InnerOutput(BaseModel):
        value: str

    class Final(BaseModel):
        result: str

    if live_llm_settings is not None:
        llm = live_llm_settings.build_openai_adapter(
            request_kwargs={"temperature": 0, "max_tokens": 64}
        )

        def _inner_text(value: str) -> str:
            prompt = f"Echo this exactly: {value}"
            return llm.complete(LLMRequest(prompt=prompt)).response.strip() or value

    else:
        def _inner_text(value: str) -> str:
            return f"inner:{value}"

    def _wrap_inner(value: str) -> InnerOutput:
        return InnerOutput(value=value)

    inner_start = StateSpec(
        name="inner_start",
        input_type=str,
        output_type=str,
        executor=FunctionStateExecutor(_inner_text),
    )
    inner_wrap = StateSpec(
        name="inner_wrap",
        input_type=str,
        output_type=InnerOutput,
        executor=FunctionStateExecutor(_wrap_inner),
    )
    inner_pipeline = inner_start >> inner_wrap

    inner_state = inner_pipeline.as_state(
        name="inner_state",
        input_type=str,
        output_type=InnerOutput,
        executor=SyncPipelineStateExecutor(inner_pipeline),
    )

    def _start(seed: Seed) -> str:
        return seed.text

    def _final(inner: InnerOutput) -> Final:
        return Final(result=inner.value)

    outer_start = StateSpec(
        name="outer_start",
        input_type=Seed,
        output_type=str,
        executor=FunctionStateExecutor(_start),
    )
    outer_end = StateSpec(
        name="outer_end",
        input_type=InnerOutput,
        output_type=Final,
        executor=FunctionStateExecutor(_final),
    )

    pipeline = outer_start >> inner_state
    pipeline.add_edge(inner_state, outer_end)
    initial = Baton.create(Seed(text="relay"), Seed).unwrap()

    executor = SyncPipelineExecutor(pipeline, initial)
    last: Baton[object] | None = None
    for step in executor:
        state_executor = step.state.executor
        assert state_executor is not None
        result = state_executor.execute(step.state, step.baton)
        executor.advance(result)
        if isinstance(result, Ok):
            last = result.value

    assert executor.failed is None
    assert last is not None
    assert isinstance(last.payload, Final)
    assert last.payload.result.strip()
