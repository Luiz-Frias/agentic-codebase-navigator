from __future__ import annotations

import pytest

from rlm.adapters.relay import FunctionStateExecutor, RelayNestedCallHandler
from rlm.api.rlm import RLM
from rlm.application.relay.root_composer import RootAgentComposer
from rlm.domain.errors import LLMError
from rlm.domain.relay import (
    InMemoryPipelineRegistry,
    Pipeline,
    PipelineTemplate,
    StateSpec,
    has_pydantic,
)
from tests.live_llm import LiveLLMSettings


def _constant_pipeline(name: str, value: str) -> Pipeline:
    state = StateSpec(
        name=f"{name}_start",
        input_type=str,
        output_type=str,
        executor=FunctionStateExecutor(lambda _text, v=value: v),
    )
    return Pipeline().add_state(state)


@pytest.mark.integration
@pytest.mark.live_llm
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_relay_nested_policy_handles_llm_query_live(
    live_llm_settings: LiveLLMSettings | None,
) -> None:
    if live_llm_settings is None:
        pytest.skip("Live LLM tests disabled")

    llm = live_llm_settings.build_openai_adapter(
        request_kwargs={
            "temperature": 0,
            "max_tokens": 160,
        }
    )

    registry = InMemoryPipelineRegistry()
    registry.register(
        PipelineTemplate(
            name="alpha",
            description="Alpha pipeline",
            input_type=str,
            output_type=str,
            factory=lambda: _constant_pipeline("alpha", "ALPHA_DONE"),
        )
    )
    registry.register(
        PipelineTemplate(
            name="beta",
            description="Beta pipeline",
            input_type=str,
            output_type=str,
            factory=lambda: _constant_pipeline("beta", "BETA_DONE"),
        )
    )

    composer = RootAgentComposer(registry=registry, llm=llm)
    policy = RelayNestedCallHandler(registry=registry, composer=composer, max_depth=2)

    rlm = RLM(
        llm,
        environment="local",
        max_iterations=2,
        nested_call_policy=policy,
    )

    prompt = (
        "Return only a repl code block. In it, set "
        "resp = llm_query('Choose between alpha and beta pipelines. Prefer alpha.'), "
        "then call FINAL_VAR('resp')."
    )
    try:
        result = rlm.completion(prompt)
    except LLMError as exc:
        pytest.skip(str(exc))
    assert result.response.strip() in {"ALPHA_DONE", "BETA_DONE"}
