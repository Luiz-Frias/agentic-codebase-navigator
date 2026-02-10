from __future__ import annotations

from rlm.adapters.relay import FunctionStateExecutor, SyncPipelineExecutor
from rlm.domain.models.result import Ok
from rlm.domain.relay import Baton, StateSpec, has_pydantic

if not has_pydantic():
    raise SystemExit("Pydantic is required. Install with `rlm[pydantic]`.")

from pydantic import BaseModel


class Query(BaseModel):
    text: str


class Sources(BaseModel):
    documents: list[str]


class Findings(BaseModel):
    insights: list[str]


class Summary(BaseModel):
    answer: str


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

initial = Baton.create(Query(text="relay"), Query).unwrap()
executor = SyncPipelineExecutor(pipeline, initial)
last = None
for step in executor:
    result = step.state.executor.execute(step.state, step.baton)
    executor.advance(result)
    if isinstance(result, Ok):
        last = result.value

if last is None:
    raise SystemExit("Pipeline produced no result.")

print(last.payload.answer)  # noqa: T201
