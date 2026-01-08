from __future__ import annotations

import pytest

from rlm.domain.models import (
    BatchedLLMRequest,
    Iteration,
    LLMRequest,
    ReplResult,
    RunMetadata,
)
from tests.fakes_ports import (
    CollectingLogger,
    FakeClock,
    InMemoryBroker,
    QueueEnvironment,
    QueueLLM,
    SequenceIdGenerator,
)


@pytest.mark.unit
def test_llm_port_contract_complete_updates_last_usage_and_returns_completion() -> None:
    llm = QueueLLM(model_name="mock", responses=["ok"])
    req = LLMRequest(prompt="hi")

    cc = llm.complete(req)

    assert cc.response == "ok"
    assert cc.prompt == "hi"
    assert cc.root_model == "mock"
    assert llm.get_last_usage() == cc.usage_summary
    assert llm.get_usage_summary().model_usage_summaries["mock"].total_calls == 1


@pytest.mark.unit
def test_llm_port_contract_propagates_errors() -> None:
    err = RuntimeError("boom")
    llm = QueueLLM(model_name="mock", responses=[err])

    with pytest.raises(RuntimeError, match="boom"):
        llm.complete(LLMRequest(prompt="hi"))


@pytest.mark.unit
def test_broker_port_contract_routes_by_model_name_and_defaults() -> None:
    a = QueueLLM(model_name="a", responses=["a1"])
    b = QueueLLM(model_name="b", responses=["b1"])
    broker = InMemoryBroker(default_llm=a)
    broker.register_llm("b", b)

    cc_b = broker.complete(LLMRequest(prompt="p", model="b"))
    assert cc_b.response == "b1"

    # Unknown model falls back to default LLM.
    cc_default = broker.complete(LLMRequest(prompt="p2", model="unknown"))
    assert cc_default.response == "a1"


@pytest.mark.unit
def test_broker_port_contract_batched_preserves_order() -> None:
    llm = QueueLLM(model_name="mock", responses=["r1", "r2", "r3"])
    broker = InMemoryBroker(default_llm=llm)

    req = BatchedLLMRequest(prompts=["p1", "p2", "p3"])
    results = broker.complete_batched(req)

    assert [r.response for r in results] == ["r1", "r2", "r3"]
    assert [r.prompt for r in results] == ["p1", "p2", "p3"]
    assert llm.get_usage_summary().model_usage_summaries["mock"].total_calls == 3


@pytest.mark.unit
def test_broker_port_contract_propagates_errors_from_llm() -> None:
    llm = QueueLLM(model_name="mock", responses=["ok1", ValueError("bad")])
    broker = InMemoryBroker(default_llm=llm)

    # First request ok, second raises.
    broker.complete(LLMRequest(prompt="p1"))
    with pytest.raises(ValueError, match="bad"):
        broker.complete(LLMRequest(prompt="p2"))


@pytest.mark.unit
def test_environment_port_contract_executes_in_order_and_propagates_errors() -> None:
    env = QueueEnvironment(
        results=[
            ReplResult(stdout="s1"),
            ValueError("exec_failed"),
        ]
    )

    r1 = env.execute_code("print(1)")
    assert r1.stdout == "s1"
    with pytest.raises(ValueError, match="exec_failed"):
        env.execute_code("print(2)")

    assert env.executed_code == ["print(1)", "print(2)"]


@pytest.mark.unit
def test_logger_port_contract_accepts_metadata_and_iterations() -> None:
    logger = CollectingLogger()

    md = RunMetadata(
        root_model="m",
        max_depth=1,
        max_iterations=2,
        backend="dummy",
        environment_type="local",
    )
    it = Iteration(prompt="p", response="r", iteration_time=0.0)

    logger.log_metadata(md)
    logger.log_iteration(it)

    assert logger.metadata == [md]
    assert logger.iterations == [it]


@pytest.mark.unit
def test_clock_and_id_generator_ports_are_deterministic() -> None:
    clock = FakeClock(start=100.0, step=0.5)
    assert clock.now() == 100.5
    assert clock.now() == 101.0

    ids = SequenceIdGenerator(prefix="run", start=7)
    assert ids.new_id() == "run-7"
    assert ids.new_id() == "run-8"
