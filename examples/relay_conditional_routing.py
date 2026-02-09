from __future__ import annotations

from rlm.adapters.relay import FunctionStateExecutor, SyncPipelineExecutor
from rlm.domain.models.result import Ok
from rlm.domain.relay import Baton, StateSpec, has_pydantic

if not has_pydantic():
    raise SystemExit("Pydantic is required. Install with `rlm[pydantic]`.")


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

pipeline = start.when(lambda baton: baton.payload > 0) >> yes
pipeline = pipeline.otherwise(no)

initial = Baton.create(1, int).unwrap()
executor = SyncPipelineExecutor(pipeline, initial)

last = None
for step in executor:
    result = step.state.executor.execute(step.state, step.baton)
    executor.advance(result)
    if isinstance(result, Ok):
        last = result.value

if last is None:
    raise SystemExit("Pipeline produced no result.")

print(last.payload)  # noqa: T201
