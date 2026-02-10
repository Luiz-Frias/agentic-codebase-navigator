from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.adapters.relay import RLMStateExecutor, SyncPipelineExecutor
from rlm.adapters.tools import InMemoryToolRegistry
from rlm.domain.models import ChatCompletion
from rlm.domain.models.result import Ok
from rlm.domain.relay import Baton, Pipeline, StateSpec, has_pydantic
from rlm.domain.services.rlm_orchestrator import RLMOrchestrator
from tests.fakes_ports import QueueEnvironment
from tests.live_llm import LiveLLMSettings


@dataclass
class TrackingStoppingPolicy:
    invocation_log: list[dict[str, object]] = field(default_factory=list)

    def should_stop(self, context: dict[str, object]) -> bool:
        self.invocation_log.append({"method": "should_stop", "context": dict(context)})
        return False

    def on_iteration_complete(self, context: dict[str, object], result: ChatCompletion) -> None:
        self.invocation_log.append(
            {
                "method": "on_iteration_complete",
                "context": dict(context),
                "result_response": result.response,
            }
        )


@pytest.mark.integration
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_relay_pipeline_stopping_policy_integration(
    live_llm_settings: LiveLLMSettings | None,
) -> None:
    def echo(value: str) -> str:
        return value

    registry = InMemoryToolRegistry()
    registry.register(echo)

    if live_llm_settings is not None:
        llm = live_llm_settings.build_openai_adapter(
            request_kwargs={"temperature": 0, "max_tokens": 128}
        )
    else:
        llm = MockLLMAdapter(
            model="mock-tool",
            script=[
                {
                    "tool_calls": [
                        {"id": "call_1", "name": "echo", "arguments": {"value": "hi"}},
                    ],
                    "response": "",
                    "finish_reason": "tool_calls",
                },
                "hi",
            ],
        )

    policy = TrackingStoppingPolicy()
    orchestrator = RLMOrchestrator(
        llm=llm,
        environment=QueueEnvironment(),
        tool_registry=registry,
        agent_mode="tools",
        stopping_policy=policy,
    )

    state = StateSpec(
        name="policy_state",
        input_type=str,
        output_type=ChatCompletion,
        executor=RLMStateExecutor(
            orchestrator=orchestrator,
            max_iterations=3,
            max_depth=1,
            tool_choice="required",
        ),
    )

    pipeline = Pipeline().add_state(state)
    initial = Baton.create("Use the tool to echo 'hi'.", str).unwrap()

    executor = SyncPipelineExecutor(pipeline, initial)
    last: ChatCompletion | None = None
    for step in executor:
        result = step.state.executor.execute(step.state, step.baton)
        executor.advance(result)
        if isinstance(result, Ok):
            last = result.value.payload

    assert executor.failed is None
    assert last is not None
    assert last.response.strip()
    assert policy.invocation_log
