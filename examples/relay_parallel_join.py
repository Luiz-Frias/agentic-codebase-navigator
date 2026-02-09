from __future__ import annotations

from rlm.adapters.relay import FunctionStateExecutor, SyncPipelineExecutor
from rlm.domain.models.result import Ok
from rlm.domain.relay import Baton, StateSpec, has_pydantic

if not has_pydantic():
    raise SystemExit("Pydantic is required. Install with `rlm[pydantic]`.")

from pydantic import BaseModel


class BranchNote(BaseModel):
    note: str


class Joined(BaseModel):
    combined: str


def _left(value: int) -> BranchNote:
    return BranchNote(note=f"left-{value}")


def _right(value: int) -> BranchNote:
    return BranchNote(note=f"right-{value}")


def _join(payload: dict[str, BranchNote]) -> Joined:
    left = payload.get("left")
    right = payload.get("right")
    combined = " | ".join(note.note for note in (left, right) if isinstance(note, BranchNote))
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
executor = SyncPipelineExecutor(pipeline, initial)

last = None
for step in executor:
    result = step.state.executor.execute(step.state, step.baton)
    executor.advance(result)
    if isinstance(result, Ok):
        last = result.value

if last is None:
    raise SystemExit("Pipeline produced no result.")

print(last.payload.combined)  # noqa: T201
