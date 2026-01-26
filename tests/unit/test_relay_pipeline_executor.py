from __future__ import annotations

import pytest

from rlm.adapters.relay import FunctionStateExecutor, SyncPipelineExecutor
from rlm.domain.models.result import Ok
from rlm.domain.relay import Baton, JoinSpec, StateSpec, has_pydantic


@pytest.mark.unit
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_sync_pipeline_executor_sequential() -> None:
    s1 = StateSpec(name="s1", input_type=int, output_type=int)
    s2 = StateSpec(name="s2", input_type=int, output_type=int)

    pipeline = s1 >> s2
    executor = SyncPipelineExecutor(pipeline, Baton.create(1, int).unwrap())

    steps: list[str] = []
    for step in executor:
        steps.append(step.state.name)
        result = FunctionStateExecutor(lambda x: x + 1).execute(step.state, step.baton)
        executor.advance(result)

    assert steps == ["s1", "s2"]


@pytest.mark.unit
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_sync_pipeline_executor_conditional() -> None:
    start = StateSpec(name="start", input_type=int, output_type=int)
    yes = StateSpec(name="yes", input_type=int, output_type=int)
    no = StateSpec(name="no", input_type=int, output_type=int)

    pipeline = start.when(lambda b: b.payload > 0) >> yes
    pipeline = pipeline.otherwise(no)

    executor = SyncPipelineExecutor(pipeline, Baton.create(1, int).unwrap())

    steps: list[str] = []
    for step in executor:
        steps.append(step.state.name)
        result = FunctionStateExecutor(lambda x: x).execute(step.state, step.baton)
        executor.advance(result)

    assert steps == ["start", "yes"]


@pytest.mark.unit
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_sync_pipeline_executor_parallel_join_all() -> None:
    left = StateSpec(name="left", input_type=int, output_type=int)
    right = StateSpec(name="right", input_type=int, output_type=int)
    join = StateSpec(name="join", input_type=int, output_type=int)

    pipeline = (left | right).join(mode="all") >> join

    executor = SyncPipelineExecutor(pipeline, Baton.create(1, int).unwrap())

    steps: list[str] = []
    for step in executor:
        steps.append(step.state.name)
        result = FunctionStateExecutor(lambda x: x + 1).execute(step.state, step.baton)
        executor.advance(result)

    assert steps.count("left") == 1
    assert steps.count("right") == 1
    assert steps.count("join") == 1


@pytest.mark.unit
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_sync_pipeline_executor_run_parallel_modes() -> None:
    left = StateSpec(name="left", input_type=int, output_type=int)
    right = StateSpec(name="right", input_type=int, output_type=int)
    pipeline = left >> right

    executor = SyncPipelineExecutor(pipeline, Baton.create(1, int).unwrap())

    def _execute(state: StateSpec[object, object], baton: Baton[object]):
        return FunctionStateExecutor(lambda x: x).execute(state, baton)

    all_results = executor.run_parallel([left, right], Baton.create(1, int).unwrap(), _execute)
    assert len(all_results) == 2
    assert all(isinstance(result, Ok) for result in all_results)

    race_results = executor.run_parallel(
        [left, right],
        Baton.create(1, int).unwrap(),
        _execute,
        mode="race",
    )
    assert len(race_results) == 1
    assert isinstance(race_results[0], Ok)
