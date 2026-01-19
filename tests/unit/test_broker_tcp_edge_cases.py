from __future__ import annotations

import asyncio

import pytest

from rlm.adapters.broker.tcp import (
    TcpBrokerAdapter,
    _AsyncLoopThread,
    _safe_error_message,
)
from rlm.domain.errors import LLMError, ValidationError
from rlm.domain.models import (
    BatchedLLMRequest,
    ChatCompletion,
    LLMRequest,
    ModelUsageSummary,
    UsageSummary,
)
from rlm.domain.policies.timeouts import CancellationPolicy
from rlm.infrastructure.comms.messages import WireRequest


class _AsyncLLM:
    def __init__(self) -> None:
        self.model_name = "m"

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        raise AssertionError("not used")

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        if request.prompt == "boom":
            raise ValueError("boom")
        usage = UsageSummary(model_usage_summaries={"m": ModelUsageSummary(1, 0, 0)})
        return ChatCompletion(
            root_model=request.model or self.model_name,
            prompt=request.prompt,
            response=f"r:{request.prompt}",
            usage_summary=usage,
            execution_time=0.0,
        )

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})

    def get_last_usage(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})


@pytest.mark.unit
def test_safe_error_message_classifies_errors() -> None:
    assert _safe_error_message(LLMError("x")) == "x"
    assert _safe_error_message(ValidationError("x")) == "x"
    assert _safe_error_message(ValueError("x")) == "x"
    assert _safe_error_message(TypeError("x")) == "x"
    assert _safe_error_message(TimeoutError()) == "Request timed out"
    assert _safe_error_message(ConnectionError("x")) == "Connection error"
    assert _safe_error_message(OSError("x")) == "Connection error"
    assert _safe_error_message(RuntimeError("x")) == "Internal broker error"


@pytest.mark.unit
def test_async_loop_thread_start_stop_idempotent_and_run_timeout_path() -> None:
    loop = _AsyncLoopThread()

    coro = asyncio.sleep(0)
    try:
        with pytest.raises(RuntimeError, match="not started"):
            loop.run(coro)
    finally:
        coro.close()

    loop.start()
    loop.start()  # idempotent

    async def _slow() -> None:
        await asyncio.sleep(1)

    with pytest.raises(TimeoutError, match="Batched request timed out"):
        loop.run(
            _slow(),
            timeout_s=0.01,
            cancellation=CancellationPolicy(grace_timeout_s=0.01),
        )

    loop.stop()
    loop.stop()  # idempotent


@pytest.mark.unit
def test_tcp_broker_address_before_start_uses_configured_host_and_port() -> None:
    broker = TcpBrokerAdapter(_AsyncLLM(), host="127.0.0.1", port=12345)
    assert broker.address == ("127.0.0.1", 12345)


@pytest.mark.unit
def test_tcp_broker_register_llm_validations() -> None:
    broker = TcpBrokerAdapter(_AsyncLLM())

    with pytest.raises(ValidationError, match="non-empty model_name"):
        broker.register_llm("", _AsyncLLM())  # type: ignore[arg-type]

    other = _AsyncLLM()
    other.model_name = "other"
    with pytest.raises(ValidationError, match="must match llm.model_name"):
        broker.register_llm("mismatch", other)


@pytest.mark.unit
def test_tcp_broker_complete_batched_success_and_error_paths() -> None:
    llm = _AsyncLLM()
    broker = TcpBrokerAdapter(llm)
    addr = broker.start()
    assert isinstance(addr, tuple)
    try:
        out = broker.complete_batched(BatchedLLMRequest(prompts=["a", "b"], model=None))
        assert [c.response for c in out] == ["r:a", "r:b"]

        with pytest.raises(LLMError, match="boom"):
            broker.complete_batched(BatchedLLMRequest(prompts=["a", "boom"], model=None))
    finally:
        broker.stop()


@pytest.mark.unit
def test_tcp_broker_handle_wire_request_missing_prompts_and_timeout() -> None:
    broker = TcpBrokerAdapter(_AsyncLLM())

    # Defensive branch: bypass WireRequest validation by constructing directly.
    resp = broker._handle_wire_request(WireRequest(prompt=None, prompts=None))  # type: ignore[arg-type]
    assert resp.error == "WireRequest missing prompts"

    broker.start()
    try:
        # Force timeout branch in _handle_wire_request for batched prompts.
        def _timeout(coro, *_a, **_k):
            try:
                close = getattr(coro, "close", None)
                if callable(close):
                    close()
            finally:
                raise TimeoutError

        broker._async_loop.run = _timeout  # type: ignore[method-assign]
        resp2 = broker._handle_wire_request(WireRequest(prompts=["a"]))
        assert resp2.error == "Request timed out"
    finally:
        broker.stop()


@pytest.mark.unit
def test_tcp_broker_handle_wire_request_batched_records_per_item_errors() -> None:
    broker = TcpBrokerAdapter(_AsyncLLM())
    broker.start()
    try:
        resp = broker._handle_wire_request(WireRequest(prompts=["a", "boom", "c"]))
        assert resp.error is None
        assert resp.results is not None
        assert [r.error is None for r in resp.results] == [True, False, True]
        assert resp.results[1].error == "boom"
    finally:
        broker.stop()
