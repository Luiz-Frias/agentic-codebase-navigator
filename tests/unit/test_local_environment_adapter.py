from __future__ import annotations

from pathlib import Path

import pytest

from rlm.adapters.environments.local import LocalEnvironmentAdapter
from tests.fakes_ports import InMemoryBroker, QueueLLM


@pytest.mark.unit
def test_local_environment_state_persists_across_execute_code_calls() -> None:
    env = LocalEnvironmentAdapter()
    try:
        r1 = env.execute_code("x = 1")
        assert r1.locals["x"] == 1

        r2 = env.execute_code("print(x)")
        assert r2.stdout == "1\n"
        assert r2.stderr == ""
        assert r2.locals["x"] == 1
    finally:
        env.cleanup()


@pytest.mark.unit
def test_local_environment_captures_stdout_and_stderr() -> None:
    env = LocalEnvironmentAdapter()
    try:
        r = env.execute_code('print("hi")')
        assert r.stdout == "hi\n"
        assert r.stderr == ""
    finally:
        env.cleanup()


@pytest.mark.unit
def test_local_environment_captures_exceptions_in_stderr() -> None:
    env = LocalEnvironmentAdapter()
    try:
        r = env.execute_code("1/0")
        assert r.stdout == ""
        assert "ZeroDivisionError" in r.stderr
    finally:
        env.cleanup()


@pytest.mark.unit
def test_local_environment_load_context_sets_context_variable() -> None:
    env = LocalEnvironmentAdapter()
    try:
        env.load_context({"name": "O'Brien"})
        r = env.execute_code('print(context["name"])')
        assert r.stdout == "O'Brien\n"
        assert r.stderr == ""
    finally:
        env.cleanup()


@pytest.mark.unit
def test_local_environment_exec_runs_in_session_cwd() -> None:
    env = LocalEnvironmentAdapter()
    try:
        r = env.execute_code("from pathlib import Path\nprint(Path.cwd())")
        assert Path(r.stdout.strip()).resolve() == env.session_dir.resolve()
        assert env.session_dir.is_dir()
    finally:
        env.cleanup()


@pytest.mark.unit
def test_local_environment_llm_query_records_llm_calls_when_broker_provided() -> None:
    broker = InMemoryBroker(default_llm=QueueLLM(model_name="sub", responses=["pong"]))
    env = LocalEnvironmentAdapter(broker=broker, broker_address=("127.0.0.1", 0))
    try:
        r = env.execute_code("resp = llm_query('ping', model='sub')\nprint(resp)")
        assert r.stdout == "pong\n"
        assert r.stderr == ""
        assert [c.response for c in r.llm_calls] == ["pong"]
        assert r.locals["resp"] == "pong"
    finally:
        env.cleanup()


@pytest.mark.unit
def test_local_environment_llm_query_batched_preserves_order_and_records_only_successes() -> None:
    broker = InMemoryBroker(default_llm=QueueLLM(model_name="sub", responses=["r1", "r2", "r3"]))
    env = LocalEnvironmentAdapter(broker=broker, broker_address=("127.0.0.1", 0))
    try:
        r = env.execute_code("out = llm_query_batched(['a','b','c'], model='sub')\nprint(out)")
        assert r.stderr == ""
        assert r.locals["out"] == ["r1", "r2", "r3"]
        assert [c.response for c in r.llm_calls] == ["r1", "r2", "r3"]
    finally:
        env.cleanup()


@pytest.mark.unit
def test_local_environment_llm_query_batched_returns_error_strings_when_broker_batch_fails() -> (
    None
):
    broker = InMemoryBroker(
        default_llm=QueueLLM(model_name="sub", responses=[RuntimeError("boom")])
    )
    env = LocalEnvironmentAdapter(broker=broker, broker_address=("127.0.0.1", 0))
    try:
        r = env.execute_code("out = llm_query_batched(['a','b','c'], model='sub')\nprint(out)")
        assert r.stderr == ""
        assert isinstance(r.locals["out"], list)
        assert len(r.locals["out"]) == 3
        assert all("Error: LM query failed" in s for s in r.locals["out"])
        assert r.llm_calls == []
    finally:
        env.cleanup()


@pytest.mark.unit
def test_local_environment_cleanup_removes_session_dir() -> None:
    env = LocalEnvironmentAdapter()
    session_dir = env.session_dir
    assert session_dir.exists()
    env.cleanup()
    assert not session_dir.exists()
