from __future__ import annotations

import pytest

from rlm.adapters.relay import FunctionStateExecutor, RLMStateExecutor, SyncPipelineExecutor
from rlm.adapters.tools import InMemoryToolRegistry
from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.domain.models import ChatCompletion, LLMRequest
from rlm.domain.models.result import Ok
from rlm.domain.relay import Baton, Pipeline, StateSpec, has_pydantic
from rlm.domain.services.rlm_orchestrator import RLMOrchestrator
from tests.fakes_ports import QueueEnvironment
from tests.live_llm import LiveLLMSettings

if has_pydantic():
    from pydantic import BaseModel
else:
    BaseModel = object  # type: ignore[assignment]


def _run_pipeline(pipeline, initial: Baton[object]) -> Baton[object]:
    executor = SyncPipelineExecutor(pipeline, initial)
    last: Baton[object] | None = None
    for step in executor:
        state_executor = step.state.executor
        assert state_executor is not None
        result = state_executor.execute(step.state, step.baton)
        executor.advance(result)
        if isinstance(result, Ok):
            last = result.value
    if executor.failed is not None:
        raise AssertionError(f"Pipeline failed: {executor.failed}")
    if last is None:
        raise AssertionError("Pipeline produced no result")
    return last


@pytest.mark.e2e
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_relay_research_pipeline_e2e(live_llm_settings: LiveLLMSettings | None) -> None:
    class Query(BaseModel):
        text: str

    class Sources(BaseModel):
        documents: list[str]

    class Findings(BaseModel):
        insights: list[str]

    class Summary(BaseModel):
        answer: str

    if live_llm_settings is not None:
        llm = live_llm_settings.build_openai_adapter(
            request_kwargs={"temperature": 0, "max_tokens": 128}
        )

        def _sources(query: Query) -> Sources:
            prompt = f"List two short sources about {query.text} as comma-separated phrases."
            response = llm.complete(LLMRequest(prompt=prompt)).response
            docs = [part.strip() for part in response.split(",") if part.strip()]
            if not docs:
                docs = ["source"]
            return Sources(documents=docs[:2])

        def _findings(sources: Sources) -> Findings:
            return Findings(insights=[f"Insight from {doc}" for doc in sources.documents])

        def _summary(findings: Findings) -> Summary:
            answer = "; ".join(findings.insights)
            return Summary(answer=answer)

    else:
        def _sources(query: Query) -> Sources:
            return Sources(documents=[f"doc:{query.text}", "doc:extra"])

        def _findings(sources: Sources) -> Findings:
            return Findings(insights=[f"Insight from {doc}" for doc in sources.documents])

        def _summary(findings: Findings) -> Summary:
            return Summary(answer="; ".join(findings.insights))

    start = StateSpec(
        name="query",
        input_type=Query,
        output_type=Sources,
        executor=FunctionStateExecutor(_sources),
    )
    analyze = StateSpec(
        name="analyze",
        input_type=Sources,
        output_type=Findings,
        executor=FunctionStateExecutor(_findings),
    )
    summarize = StateSpec(
        name="summary",
        input_type=Findings,
        output_type=Summary,
        executor=FunctionStateExecutor(_summary),
    )

    pipeline = start >> analyze
    pipeline.add_edge(analyze, summarize)
    initial = Baton.create(Query(text="rlm relay"), Query).unwrap()
    final_baton = _run_pipeline(pipeline, initial)

    assert isinstance(final_baton.payload, Summary)
    assert final_baton.payload.answer.strip()


@pytest.mark.e2e
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_relay_parallel_join_e2e(live_llm_settings: LiveLLMSettings | None) -> None:
    class BranchNote(BaseModel):
        note: str

    class Joined(BaseModel):
        combined: str

    if live_llm_settings is not None:
        llm = live_llm_settings.build_openai_adapter(
            request_kwargs={"temperature": 0, "max_tokens": 64}
        )

        def _left(value: int) -> BranchNote:
            prompt = f"Return a short note about {value}."
            response = llm.complete(LLMRequest(prompt=prompt)).response
            return BranchNote(note=response.strip() or "left")

        def _right(value: int) -> BranchNote:
            return BranchNote(note=f"right-{value}")

    else:
        def _left(value: int) -> BranchNote:
            return BranchNote(note=f"left-{value}")

        def _right(value: int) -> BranchNote:
            return BranchNote(note=f"right-{value}")

    def _join(payload: dict[str, BranchNote]) -> Joined:
        left = payload.get("left")
        right = payload.get("right")
        combined = " | ".join(
            note.note for note in (left, right) if isinstance(note, BranchNote)
        )
        return Joined(combined=combined)

    left_state = StateSpec(
        name="left",
        input_type=int,
        output_type=BranchNote,
        executor=FunctionStateExecutor(_left),
    )
    right_state = StateSpec(
        name="right",
        input_type=int,
        output_type=BranchNote,
        executor=FunctionStateExecutor(_right),
    )
    join_state = StateSpec(
        name="join",
        input_type=dict,
        output_type=Joined,
        executor=FunctionStateExecutor(_join),
    )

    pipeline = (left_state | right_state).join(mode="all") >> join_state
    initial = Baton.create(3, int).unwrap()
    final_baton = _run_pipeline(pipeline, initial)

    assert isinstance(final_baton.payload, Joined)
    assert final_baton.payload.combined.strip()


@pytest.mark.e2e
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_relay_conditional_routing_e2e() -> None:
    start = StateSpec(
        name="start",
        input_type=int,
        output_type=int,
        executor=FunctionStateExecutor(lambda value: value),
    )
    yes = StateSpec(
        name="yes",
        input_type=int,
        output_type=str,
        executor=FunctionStateExecutor(lambda value: f"yes-{value}"),
    )
    no = StateSpec(
        name="no",
        input_type=int,
        output_type=str,
        executor=FunctionStateExecutor(lambda value: f"no-{value}"),
    )

    pipeline = start.when(lambda b: b.payload > 0) >> yes
    pipeline = pipeline.otherwise(no)

    initial = Baton.create(1, int).unwrap()
    final_baton = _run_pipeline(pipeline, initial)
    assert isinstance(final_baton.payload, str)
    assert final_baton.payload.startswith("yes-")


@pytest.mark.e2e
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_relay_rlm_state_executor_e2e(live_llm_settings: LiveLLMSettings | None) -> None:
    def shout(value: str) -> str:
        return value.upper()

    registry = InMemoryToolRegistry()
    registry.register(shout)

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
                        {"id": "call_1", "name": "shout", "arguments": {"value": "hi"}},
                    ],
                    "response": "",
                    "finish_reason": "tool_calls",
                },
                "HI",
            ],
        )

    orchestrator = RLMOrchestrator(
        llm=llm,
        environment=QueueEnvironment(),
        tool_registry=registry,
        agent_mode="tools",
    )

    state = StateSpec(
        name="tool_state",
        input_type=str,
        output_type=ChatCompletion,
        executor=RLMStateExecutor(
            orchestrator=orchestrator,
            max_iterations=4,
            max_depth=1,
            tool_choice="required",
        ),
    )

    pipeline = Pipeline().add_state(state)
    initial = Baton.create("Use the tool to shout 'hi'.", str).unwrap()
    final_baton = _run_pipeline(pipeline, initial)
    assert str(final_baton.payload.response).strip()
