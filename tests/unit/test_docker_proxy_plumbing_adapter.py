from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlm.adapters.environments.docker import DockerLLMProxyHandler, _build_exec_script
from tests.fakes_ports import InMemoryBroker, QueueLLM


@pytest.mark.unit
def test_docker_proxy_handler_single_requires_broker_or_broker_address() -> None:
    import threading

    dummy = SimpleNamespace(
        broker=None,
        broker_address=None,
        pending_calls=[],
        lock=threading.Lock(),
        timeout_s=0.1,
    )
    out = DockerLLMProxyHandler._handle_single(dummy, {"prompt": "hi"})  # type: ignore[arg-type]
    assert out == {"error": "No broker configured"}


@pytest.mark.unit
def test_docker_proxy_handler_single_uses_broker_and_records_pending_call() -> None:
    import threading

    broker = InMemoryBroker(default_llm=QueueLLM(model_name="dummy", responses=["ok"]))
    dummy = SimpleNamespace(
        broker=broker,
        broker_address=None,
        pending_calls=[],
        lock=threading.Lock(),
        timeout_s=0.1,
    )
    out = DockerLLMProxyHandler._handle_single(dummy, {"prompt": "ping", "model": "dummy"})  # type: ignore[arg-type]
    assert out == {"response": "ok"}
    assert [c.response for c in dummy.pending_calls] == ["ok"]


@pytest.mark.unit
def test_docker_proxy_handler_batched_uses_broker_per_item_and_preserves_order() -> None:
    import threading

    broker = InMemoryBroker(
        default_llm=QueueLLM(model_name="dummy", responses=["r1", RuntimeError("boom"), "r3"])
    )
    dummy = SimpleNamespace(
        broker=broker,
        broker_address=None,
        pending_calls=[],
        lock=threading.Lock(),
        timeout_s=0.1,
    )

    out = DockerLLMProxyHandler._handle_batched(  # type: ignore[arg-type]
        dummy, {"prompts": ["a", "b", "c"], "model": "dummy"}
    )
    assert out == {"responses": ["r1", "Error: boom", "r3"]}
    assert [c.response for c in dummy.pending_calls] == ["r1", "r3"]


@pytest.mark.unit
def test_docker_exec_script_passes_correlation_id_to_proxy_requests() -> None:
    script = _build_exec_script("print('hi')", 1234)
    assert "RUN_CORRELATION_ID" in script
    assert "correlation_id" in script
    assert "def llm_query(prompt, model=None, correlation_id=None):" in script
    assert "def llm_query_batched(prompts, model=None, correlation_id=None):" in script
    assert '"correlation_id": cid' in script


@pytest.mark.unit
def test_docker_exec_script_llm_query_preserves_empty_string_response() -> None:
    """
    Regression: `llm_query()` must not treat an empty-string LLM response as an error.

    The container-side script should explicitly branch on `error` rather than
    using `d.get("response") or ...`, because `""` is falsy.
    """

    script = _build_exec_script("print('hi')", 1234)
    assert 'return d.get("response") or' not in script
    assert 'return d.get("responses") or' not in script
    assert 'err = d.get("error")' in script
    assert "if err is not None:" in script


@pytest.mark.unit
def test_docker_proxy_handler_validates_model_and_correlation_id_types_in_broker_mode() -> None:
    import threading

    broker = InMemoryBroker(default_llm=QueueLLM(responses=["ok"]))
    dummy = SimpleNamespace(
        broker=broker,
        broker_address=None,
        pending_calls=[],
        lock=threading.Lock(),
        timeout_s=0.1,
    )

    out = DockerLLMProxyHandler._handle_single(dummy, {"prompt": "hi", "model": 123})  # type: ignore[arg-type]
    assert out == {"error": "Invalid model"}

    out = DockerLLMProxyHandler._handle_single(dummy, {"prompt": "hi", "correlation_id": 123})  # type: ignore[arg-type]
    assert out == {"error": "Invalid correlation_id"}


@pytest.mark.unit
def test_docker_proxy_handler_validates_prompts_list_in_broker_mode() -> None:
    import threading

    broker = InMemoryBroker(default_llm=QueueLLM(responses=["ok"]))
    dummy = SimpleNamespace(
        broker=broker,
        broker_address=None,
        pending_calls=[],
        lock=threading.Lock(),
        timeout_s=0.1,
    )

    out = DockerLLMProxyHandler._handle_batched(dummy, {"prompts": "nope"})  # type: ignore[arg-type]
    assert out == {"error": "Invalid prompts"}
