from __future__ import annotations

import pytest

from rlm.adapters.relay import FunctionStateExecutor, SyncPipelineExecutor
from rlm.domain.models.result import Err, Ok
from rlm.domain.relay import Baton, BatonMetadata, StateSpec, TokenBudget, has_pydantic


@pytest.mark.unit
def test_token_budget_remaining_and_consume() -> None:
    budget = TokenBudget(max_tokens=10, per_state_estimates={"s1": 3}, consumed=4)
    assert budget.remaining == 6
    assert budget.estimate_for("s1") == 3
    assert budget.can_consume(3)
    updated = budget.with_consumed(2)
    assert updated.consumed == 6


@pytest.mark.unit
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_sync_executor_enforces_budget_and_traces() -> None:
    state = StateSpec(name="s1", input_type=int, output_type=int)
    budget = TokenBudget(max_tokens=1, per_state_estimates={"s1": 2}, consumed=0)
    metadata = BatonMetadata.create(budget=budget)

    baton_result = Baton.create(1, int, metadata=metadata)
    assert isinstance(baton_result, Ok)

    executor = SyncPipelineExecutor(state >> state, baton_result.value)

    with pytest.raises(StopIteration):
        next(executor)

    assert executor.failed is not None


@pytest.mark.unit
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_sync_executor_records_trace_on_completion() -> None:
    state = StateSpec(name="s1", input_type=int, output_type=int)
    from rlm.domain.relay.pipeline import Pipeline

    pipeline = Pipeline().add_state(state)

    baton_result = Baton.create(1, int)
    assert isinstance(baton_result, Ok)

    executor = SyncPipelineExecutor(pipeline, baton_result.value)
    step = next(executor)
    result = FunctionStateExecutor(lambda x: x + 1).execute(step.state, step.baton)
    executor.advance(result)
    with pytest.raises(StopIteration):
        next(executor)

    assert executor.trace.entries
    assert executor.trace.entries[-1].state_name == "s1"
    assert executor.trace.entries[-1].status == "completed"
