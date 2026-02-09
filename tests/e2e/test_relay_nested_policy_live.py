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


def _suffix_pipeline(name: str, suffix: str) -> Pipeline:
    state = StateSpec(
        name=f"{name}_start",
        input_type=str,
        output_type=str,
        executor=FunctionStateExecutor(lambda text, s=suffix: f"{text}{s}"),
    )
    return Pipeline().add_state(state)


@pytest.mark.e2e
@pytest.mark.live_llm
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_relay_nested_policy_composes_chain_live(
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
            description="Alpha pipeline appends |alpha.",
            input_type=str,
            output_type=str,
            factory=lambda: _suffix_pipeline("alpha", "|alpha"),
        )
    )
    registry.register(
        PipelineTemplate(
            name="beta",
            description="Beta pipeline appends |beta.",
            input_type=str,
            output_type=str,
            factory=lambda: _suffix_pipeline("beta", "|beta"),
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
        "resp = llm_query('Use alpha and beta pipelines in order. Reply with: alpha, beta.'), "
        "then call FINAL_VAR('resp')."
    )
    try:
        result = rlm.completion(prompt)
    except LLMError as exc:
        pytest.skip(str(exc))
    response = result.response.strip().lower()
    assert "|alpha" in response
    assert "|beta" in response
