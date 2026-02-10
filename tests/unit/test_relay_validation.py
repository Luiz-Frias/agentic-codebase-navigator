from __future__ import annotations

import pytest

from rlm.domain.relay import Pipeline, PipelineDefinitionError, StateSpec, allow_cycles


def _make_state(name: str, input_type: type, output_type: type) -> StateSpec:
    return StateSpec(name=name, input_type=input_type, output_type=output_type)


@pytest.mark.unit
def test_validate_detects_type_mismatch() -> None:
    left = _make_state("left", str, str)
    right = _make_state("right", int, int)

    pipeline = Pipeline().add_edge(left, right)

    with pytest.raises(PipelineDefinitionError, match=r"Type mismatch"):
        pipeline.validate()


@pytest.mark.unit
def test_validate_detects_cycle_by_default() -> None:
    a = _make_state("a", str, str)
    b = _make_state("b", str, str)

    pipeline = Pipeline().add_edge(a, b).add_edge(b, a)

    with pytest.raises(PipelineDefinitionError, match=r"Cycle detected"):
        pipeline.validate()


@pytest.mark.unit
def test_validate_allows_cycle_with_decorator() -> None:
    a = _make_state("a", str, str)
    b = _make_state("b", str, str)

    pipeline = Pipeline().add_edge(a, b).add_edge(b, a)
    allow_cycles(max_iterations=3)(pipeline)

    with pytest.raises(PipelineDefinitionError, match=r"terminal state"):
        pipeline.validate()


@pytest.mark.unit
def test_validate_requires_max_iterations_when_allow_cycles() -> None:
    a = _make_state("a", str, str)
    b = _make_state("b", str, str)

    pipeline = Pipeline().add_edge(a, b).add_edge(b, a)
    pipeline.set_cycle_policy(allow_cycles=True, max_iterations=0)

    with pytest.raises(PipelineDefinitionError, match=r"max_cycle_iterations"):
        pipeline.validate()


@pytest.mark.unit
def test_validate_detects_unreachable_states() -> None:
    start = _make_state("start", str, str)
    end = _make_state("end", str, str)
    orphan = _make_state("orphan", str, str)

    pipeline = Pipeline().add_edge(start, end).add_state(orphan)

    with pytest.raises(PipelineDefinitionError, match=r"Unreachable states"):
        pipeline.validate()
