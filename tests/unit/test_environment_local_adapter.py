from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import rlm.adapters.environments.local as local_mod
from rlm.adapters.environments.local import LocalEnvironmentAdapter, _execution_timeout
from rlm.domain.models import ChatCompletion
from rlm.domain.models.usage import UsageSummary
from rlm.infrastructure.comms.messages import WireResult
from tests.fakes_ports import InMemoryBroker, QueueLLM


@pytest.mark.unit
def test_execution_timeout_context_manager_noops_in_expected_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _execution_timeout(None):
        pass

    ran: list[bool] = []

    def _worker() -> None:
        # Not main thread => no SIGALRM usage, just yields.
        with _execution_timeout(0.001):
            ran.append(True)

    t = threading.Thread(target=_worker)
    t.start()
    t.join(timeout=1)
    assert ran == [True]

    # Simulate platforms without SIGALRM by replacing the module-level `signal`.
    monkeypatch.setattr(local_mod, "signal", SimpleNamespace())
    with _execution_timeout(0.001):
        pass


@pytest.mark.unit
def test_local_environment_allowed_import_roots_overrides_policy() -> None:
    env = LocalEnvironmentAdapter(allowed_import_roots={"math"}, execute_timeout_s=None)
    try:
        ok = env.execute_code("import math\nprint(math.sqrt(9))")
        assert ok.stderr == ""
        assert ok.stdout.strip() == "3.0"

        blocked = env.execute_code("import json\nprint('x')")
        assert blocked.stdout == ""
        assert "ImportError" in blocked.stderr
    finally:
        env.cleanup()


@pytest.mark.unit
def test_local_environment_context_payload_and_setup_code_are_applied() -> None:
    env = LocalEnvironmentAdapter(
        context_payload="ctx",
        setup_code="x = 2\nprint('setup')",
        execute_timeout_s=None,
    )
    try:
        r = env.execute_code("print(context)\nprint(x)")
        assert r.stderr == ""
        assert r.stdout == "ctx\n2\n"
    finally:
        env.cleanup()


@pytest.mark.unit
def test_local_environment_final_var_formats_missing_and_bad_str() -> None:
    env = LocalEnvironmentAdapter(execute_timeout_s=None)
    try:
        env.execute_code("x = 123")
        assert env._final_var("x") == "123"
        assert "not found" in env._final_var("missing")

        class BadStr:
            def __str__(self) -> str:
                raise RuntimeError("nope")

        env._ns["bad"] = BadStr()
        msg = env._final_var("bad")
        assert "Failed to stringify variable" in msg
    finally:
        env.cleanup()


@pytest.mark.unit
def test_local_environment_llm_query_errors_are_returned_as_strings() -> None:
    env = LocalEnvironmentAdapter(execute_timeout_s=None)
    try:
        invalid = env.execute_code("print(llm_query('hi', correlation_id=123))")
        assert "Error: Invalid correlation_id" in invalid.stdout
        assert invalid.llm_calls == []

        no_broker = env.execute_code("print(llm_query('hi'))")
        assert "Error: LM query failed" in no_broker.stdout
        assert no_broker.llm_calls == []
    finally:
        env.cleanup()


@pytest.mark.unit
def test_local_environment_llm_query_success_records_llm_calls() -> None:
    llm = QueueLLM(responses=["ok"])
    broker = InMemoryBroker(llm)
    env = LocalEnvironmentAdapter(broker=broker, execute_timeout_s=None)
    try:
        r = env.execute_code("print(llm_query('hi'))")
        assert r.stderr == ""
        assert r.stdout.strip() == "ok"
        assert len(r.llm_calls) == 1
        assert r.llm_calls[0].response == "ok"
    finally:
        env.cleanup()


@pytest.mark.unit
def test_local_environment_llm_query_batched_handles_invalid_inputs_and_errors() -> None:
    env = LocalEnvironmentAdapter(execute_timeout_s=None)
    try:
        invalid_prompts = env.execute_code("print(llm_query_batched('nope'))")
        assert "Invalid prompts" in invalid_prompts.stdout

        invalid_cid = env.execute_code("print(llm_query_batched(['a','b'], correlation_id=123))")
        assert "Invalid correlation_id" in invalid_cid.stdout

        no_broker = env.execute_code("print(llm_query_batched(['a']))")
        assert "Error: LM query failed" in no_broker.stdout
    finally:
        env.cleanup()


@pytest.mark.unit
def test_local_environment_llm_query_batched_success_records_llm_calls() -> None:
    llm = QueueLLM(responses=["r1", "r2"])
    broker = InMemoryBroker(llm)
    env = LocalEnvironmentAdapter(broker=broker, execute_timeout_s=None)
    try:
        r = env.execute_code("print(llm_query_batched(['a','b']))")
        assert "r1" in r.stdout and "r2" in r.stdout
        assert len(r.llm_calls) == 2
        assert [c.response for c in r.llm_calls] == ["r1", "r2"]
    finally:
        env.cleanup()


@pytest.mark.unit
def test_local_environment_broker_address_paths_use_wire_protocol_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _stub_request_completion(*args, **kwargs):
        captured["rc_args"] = args
        captured["rc_kwargs"] = kwargs
        return ChatCompletion(
            root_model="m",
            prompt=kwargs.get("prompt"),
            response="ok",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    monkeypatch.setattr(local_mod, "request_completion", _stub_request_completion)

    env = LocalEnvironmentAdapter(
        broker_address=("127.0.0.1", 1234),
        correlation_id="cid",
        execute_timeout_s=None,
    )
    try:
        cc = env._request_completion(prompt="hi", model=None, correlation_id=None)
        assert cc.response == "ok"
        assert captured["rc_kwargs"]["correlation_id"] == "cid"
    finally:
        env.cleanup()

    def _stub_request_completions_batched(*args, **kwargs):
        return [
            WireResult(error="oops", chat_completion=None),
            WireResult(
                error=None,
                chat_completion=ChatCompletion(
                    root_model="m",
                    prompt="p2",
                    response="r2",
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=0.0,
                ),
            ),
        ]

    monkeypatch.setattr(local_mod, "request_completions_batched", _stub_request_completions_batched)

    env2 = LocalEnvironmentAdapter(
        broker_address=("127.0.0.1", 1234),
        correlation_id="cid",
        execute_timeout_s=None,
    )
    try:
        out, calls = env2._request_completions_batched(
            prompts=["p1", "p2"],
            model=None,
            correlation_id=None,
        )
        assert out == ["Error: oops", "r2"]
        assert len(calls) == 1 and calls[0].response == "r2"
    finally:
        env2.cleanup()


@pytest.mark.unit
def test_local_environment_cleanup_is_best_effort_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = LocalEnvironmentAdapter(execute_timeout_s=None)
    monkeypatch.setattr(env._tmp, "cleanup", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    env.cleanup()  # must not raise
    assert env._ns == {}
    assert env._pending_llm_calls == []
