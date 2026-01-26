from __future__ import annotations

import pytest

from rlm.domain.errors import ValidationError
from rlm.domain.relay import JoinSpec, Pipeline, StateSpec


def _guard(_: object) -> bool:
    return True


@pytest.mark.unit
def test_pipeline_add_edge_sets_entry_and_terminals() -> None:
    start = StateSpec(name="start", input_type=str, output_type=str)
    end = StateSpec(name="end", input_type=str, output_type=int)

    pipeline = Pipeline().add_edge(start, end)

    assert pipeline.entry_state == start
    assert pipeline.terminal_states == (end,)
    assert len(pipeline.edges) == 1


@pytest.mark.unit
def test_state_rshift_builds_pipeline() -> None:
    left = StateSpec(name="left", input_type=str, output_type=str)
    right = StateSpec(name="right", input_type=str, output_type=int)

    pipeline = left >> right

    assert isinstance(pipeline, Pipeline)
    assert pipeline.entry_state == left
    assert pipeline.terminal_states == (right,)


@pytest.mark.unit
def test_parallel_group_join_updates_spec() -> None:
    left = StateSpec(name="left", input_type=str, output_type=str)
    right = StateSpec(name="right", input_type=str, output_type=str)

    group = left | right
    assert group.join_spec == JoinSpec()

    joined = group.join(mode="race", timeout_seconds=1.0)
    assert joined.join_spec.mode == "race"
    assert joined.join_spec.timeout_seconds == 1.0


@pytest.mark.unit
def test_when_otherwise_builds_conditional_edges() -> None:
    start = StateSpec(name="start", input_type=str, output_type=str)
    yes = StateSpec(name="yes", input_type=str, output_type=str)
    no = StateSpec(name="no", input_type=str, output_type=str)

    pipeline = start.when(_guard) >> yes
    assert len(pipeline.edges) == 1
    assert pipeline.edges[0].guard is _guard

    pipeline = pipeline.otherwise(no)
    assert len(pipeline.edges) == 2
    guards = [edge.guard for edge in pipeline.edges]
    assert guards.count(None) == 1


@pytest.mark.unit
def test_otherwise_requires_distinct_default_state() -> None:
    start = StateSpec(name="start", input_type=str, output_type=str)
    yes = StateSpec(name="yes", input_type=str, output_type=str)

    pipeline = start.when(_guard) >> yes
    with pytest.raises(ValidationError):
        pipeline.otherwise(yes)
