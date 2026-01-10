from __future__ import annotations

import signal
import subprocess
import threading
from dataclasses import dataclass

import pytest

from rlm.adapters.broker.tcp import TcpBrokerAdapter
from rlm.domain.errors import BrokerError, LLMError
from rlm.domain.models import BatchedLLMRequest, ChatCompletion, LLMRequest, UsageSummary
from rlm.domain.policies.timeouts import BrokerTimeouts, CancellationPolicy
from rlm.domain.ports import LLMPort
from rlm.infrastructure.comms.protocol import request_completions_batched


@dataclass
class _HangingAsyncLLM(LLMPort):
    """
    LLMPort whose async path never completes (until cancelled).

    Used to test broker batched timeouts and cancellation behavior.
    """

    _model_name: str = "default"

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, request: LLMRequest, /) -> ChatCompletion:  # pragma: no cover
        raise AssertionError("_HangingAsyncLLM.complete should not be called in these tests")

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        import asyncio

        # Sleep "forever" (test timeouts will cancel this).
        await asyncio.sleep(3600)
        return ChatCompletion(
            root_model=request.model or self._model_name,
            prompt=request.prompt,
            response="unreachable",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})

    def get_last_usage(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})


@pytest.mark.unit
def test_tcp_broker_complete_batched_times_out_fast_and_raises_llm_error() -> None:
    llm = _HangingAsyncLLM("default")
    broker = TcpBrokerAdapter(
        llm,
        timeouts=BrokerTimeouts(batched_completion_timeout_s=0.05),
        cancellation=CancellationPolicy(grace_timeout_s=0.01),
    )
    broker.start()
    try:
        with pytest.raises(LLMError, match=r"timed out"):
            broker.complete_batched(BatchedLLMRequest(prompts=["p1", "p2"]))
    finally:
        broker.stop()


@pytest.mark.unit
def test_tcp_broker_wire_protocol_batched_request_times_out_with_safe_error() -> None:
    llm = _HangingAsyncLLM("default")
    broker = TcpBrokerAdapter(
        llm,
        timeouts=BrokerTimeouts(batched_completion_timeout_s=0.05),
        cancellation=CancellationPolicy(grace_timeout_s=0.01),
    )
    addr = broker.start()
    try:
        with pytest.raises(BrokerError, match=r"Request timed out"):
            request_completions_batched(addr, ["p1", "p2"], timeout_s=1.0)
    finally:
        broker.stop()


@pytest.mark.unit
def test_localrepl_execute_code_respects_execution_timeout_when_available() -> None:
    # The legacy LocalREPL timeout watchdog is best-effort and uses SIGALRM only.
    if not (hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")):
        pytest.skip("SIGALRM/setitimer not available on this platform")
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("requires main thread for SIGALRM")

    from rlm._legacy.environments.local_repl import LocalREPL

    env = LocalREPL(execute_timeout_s=0.1)
    try:
        res = env.execute_code("import time\ntime.sleep(1)\n")
        assert "TimeoutError" in res.stderr
        assert "Execution timed out" in res.stderr
    finally:
        env.cleanup()


@pytest.mark.unit
def test_dockerrepl_execute_code_timeout_returns_error_result_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlm._legacy.environments.docker_repl as docker_mod

    def _fake_setup(self) -> None:
        # Minimal fields required by execute_code().
        self.container_id = "cid"
        self.proxy_port = 1234

    monkeypatch.setattr(docker_mod.DockerREPL, "setup", _fake_setup)

    calls: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):  # noqa: ANN001
        cmd_list = list(cmd)
        calls.append(cmd_list)
        if cmd_list[:2] == ["docker", "exec"]:
            raise subprocess.TimeoutExpired(cmd=cmd_list, timeout=kwargs.get("timeout"))
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(docker_mod.subprocess, "run", _fake_run)

    env = docker_mod.DockerREPL(
        image="python:3.12-slim",
        lm_handler_address=("127.0.0.1", 0),
        subprocess_timeout_s=0.01,
    )
    res = env.execute_code("print('hi')\n")

    assert "TimeoutExpired" in res.stderr
    assert "docker exec exceeded" in res.stderr
    assert env.container_id is None  # cleanup() should have cleared it
    assert any(cmd[:2] == ["docker", "exec"] for cmd in calls)
