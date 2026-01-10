from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlm._legacy.core.types import ModelUsageSummary, RLMChatCompletion, UsageSummary
from rlm._legacy.environments.local_repl import LocalREPL
from tests.fakes_ports import InMemoryBroker, QueueLLM


@pytest.mark.unit
def test_localrepl_captures_llm_calls_in_deterministic_order_with_broker_injection() -> None:
    llm = QueueLLM(model_name="dummy", responses=["r1", "r2", "b1", "b2"])
    broker = InMemoryBroker(default_llm=llm)

    with LocalREPL(broker=broker) as env:
        res = env.execute_code(
            "a = llm_query('p1')\n"
            "b = llm_query('p2')\n"
            "cs = llm_query_batched(['p3','p4'])\n"
            "print(a, b, cs)\n"
        )

    assert [c.response for c in res.llm_calls] == ["r1", "r2", "b1", "b2"]


def _chat_completion(response: str) -> RLMChatCompletion:
    return RLMChatCompletion(
        root_model="dummy",
        prompt="p",
        response=response,
        usage_summary=UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(1, 0, 0)}),
        execution_time=0.0,
    )


@pytest.mark.unit
def test_dockerrepl_execute_code_copies_pending_calls_in_order_and_clears_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlm._legacy.environments.docker_repl as docker_mod

    # Avoid real docker/proxy startup.
    def _fake_setup(self) -> None:
        self.container_id = "cid"
        self.proxy_port = 1234

    monkeypatch.setattr(docker_mod.DockerREPL, "setup", _fake_setup)
    monkeypatch.setattr(docker_mod.DockerREPL, "cleanup", lambda self: None)

    env = docker_mod.DockerREPL(image="python:3.12-slim", lm_handler_address=("127.0.0.1", 0))

    def _fake_run(*_args, **_kwargs):
        with env._calls_lock:
            env.pending_calls.extend([_chat_completion("c1"), _chat_completion("c2")])
        return SimpleNamespace(
            stdout='{"stdout":"out","stderr":"","locals":{}}',
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(docker_mod.subprocess, "run", _fake_run)

    res = env.execute_code("print('hi')\n")
    assert [c.response for c in res.llm_calls] == ["c1", "c2"]
    assert env.pending_calls == []
