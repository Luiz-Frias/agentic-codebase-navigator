from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

import rlm.adapters.environments.docker as docker_mod
from rlm.adapters.environments.docker import DockerLLMProxyHandler, _build_exec_script
from rlm.domain.models import ChatCompletion
from rlm.domain.models.usage import UsageSummary
from rlm.infrastructure.comms.messages import WireResult
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
        default_llm=QueueLLM(model_name="dummy", responses=["r1", RuntimeError("boom"), "r3"]),
    )
    dummy = SimpleNamespace(
        broker=broker,
        broker_address=None,
        pending_calls=[],
        lock=threading.Lock(),
        timeout_s=0.1,
    )

    out = DockerLLMProxyHandler._handle_batched(  # type: ignore[arg-type]
        dummy,
        {"prompts": ["a", "b", "c"], "model": "dummy"},
    )
    assert out == {"responses": ["r1", "Error: boom", "r3"]}
    assert [c.response for c in dummy.pending_calls] == ["r1", "r3"]


@pytest.mark.unit
def test_docker_exec_script_passes_correlation_id_to_proxy_requests() -> None:
    script = _build_exec_script("print('hi')", 1234)
    assert "RUN_CORRELATION_ID" in script
    assert "correlation_id" in script
    assert "def llm_query(prompt, model=None, correlation_id=None, depth=None):" in script
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


@pytest.mark.unit
def test_docker_proxy_handler_do_post_validations_and_routing() -> None:
    recorded: list[tuple[int, dict]] = []

    def _respond(status: int, data: dict) -> None:
        recorded.append((status, data))

    # Missing Content-Length
    dummy = SimpleNamespace(
        headers={},
        rfile=SimpleNamespace(read=lambda _n: b""),
        path="/llm_query",
        _respond=_respond,
    )
    DockerLLMProxyHandler.do_POST(dummy)  # type: ignore[arg-type]
    assert recorded.pop() == (400, {"error": "Missing Content-Length"})

    # Invalid Content-Length
    dummy = SimpleNamespace(
        headers={"Content-Length": "nope"},
        rfile=SimpleNamespace(read=lambda _n: b""),
        path="/llm_query",
        _respond=_respond,
    )
    DockerLLMProxyHandler.do_POST(dummy)  # type: ignore[arg-type]
    assert recorded.pop() == (400, {"error": "Invalid Content-Length"})

    # Missing body
    dummy = SimpleNamespace(
        headers={"Content-Length": "0"},
        rfile=SimpleNamespace(read=lambda _n: b""),
        path="/llm_query",
        _respond=_respond,
    )
    DockerLLMProxyHandler.do_POST(dummy)  # type: ignore[arg-type]
    assert recorded.pop() == (400, {"error": "Missing request body"})

    # Invalid JSON payload
    dummy = SimpleNamespace(
        headers={"Content-Length": "1"},
        rfile=SimpleNamespace(read=lambda _n: b"{"),
        path="/llm_query",
        _respond=_respond,
    )
    DockerLLMProxyHandler.do_POST(dummy)  # type: ignore[arg-type]
    assert recorded.pop() == (400, {"error": "Invalid JSON payload"})

    # Body must be dict
    body = json.dumps(["nope"]).encode("utf-8")
    dummy = SimpleNamespace(
        headers={"Content-Length": str(len(body))},
        rfile=SimpleNamespace(read=lambda _n: body),
        path="/llm_query",
        _respond=_respond,
    )
    DockerLLMProxyHandler.do_POST(dummy)  # type: ignore[arg-type]
    assert recorded.pop() == (400, {"error": "Request body must be a JSON object"})

    # Not found path
    body2 = json.dumps({"prompt": "hi"}).encode("utf-8")
    dummy = SimpleNamespace(
        headers={"Content-Length": str(len(body2))},
        rfile=SimpleNamespace(read=lambda _n: body2),
        path="/nope",
        _respond=_respond,
        _handle_single=lambda _b: {"response": "ok"},
        _handle_batched=lambda _b: {"responses": []},
    )
    DockerLLMProxyHandler.do_POST(dummy)  # type: ignore[arg-type]
    assert recorded.pop() == (404, {"error": "Not found"})

    # Routes to llm_query
    dummy = SimpleNamespace(
        headers={"Content-Length": str(len(body2))},
        rfile=SimpleNamespace(read=lambda _n: body2),
        path="/llm_query",
        _respond=_respond,
        _handle_single=lambda _b: {"response": "ok"},
        _handle_batched=lambda _b: {"responses": []},
    )
    DockerLLMProxyHandler.do_POST(dummy)  # type: ignore[arg-type]
    assert recorded.pop() == (200, {"response": "ok"})

    # Routes to llm_query_batched
    body3 = json.dumps({"prompts": ["a", "b"]}).encode("utf-8")
    dummy = SimpleNamespace(
        headers={"Content-Length": str(len(body3))},
        rfile=SimpleNamespace(read=lambda _n: body3),
        path="/llm_query_batched",
        _respond=_respond,
        _handle_single=lambda _b: {"response": "nope"},
        _handle_batched=lambda _b: {"responses": ["r1", "r2"]},
    )
    DockerLLMProxyHandler.do_POST(dummy)  # type: ignore[arg-type]
    assert recorded.pop() == (200, {"responses": ["r1", "r2"]})


@pytest.mark.unit
def test_docker_proxy_handler_do_post_handles_unexpected_exception() -> None:
    recorded: list[tuple[int, dict]] = []

    def _respond(status: int, data: dict) -> None:
        recorded.append((status, data))

    dummy = SimpleNamespace(
        headers={"Content-Length": "1"},
        rfile=SimpleNamespace(read=lambda _n: (_ for _ in ()).throw(RuntimeError("boom"))),
        path="/llm_query",
        _respond=_respond,
    )
    DockerLLMProxyHandler.do_POST(dummy)  # type: ignore[arg-type]
    assert recorded.pop() == (400, {"error": "boom"})


@pytest.mark.unit
def test_docker_proxy_handler_respond_and_log_message_are_noops() -> None:
    recorded: dict[str, object] = {
        "status": None,
        "headers": [],
        "ended": False,
        "body": b"",
    }

    def send_response(status: int) -> None:
        recorded["status"] = status

    def send_header(k: str, v: str) -> None:
        recorded["headers"].append((k, v))  # type: ignore[union-attr]

    def end_headers() -> None:
        recorded["ended"] = True

    def write(data: bytes) -> None:
        recorded["body"] += data  # type: ignore[operator]

    dummy = SimpleNamespace(
        send_response=send_response,
        send_header=send_header,
        end_headers=end_headers,
        wfile=SimpleNamespace(write=write),
    )
    DockerLLMProxyHandler._respond(dummy, 200, {"x": 1})  # type: ignore[arg-type]
    assert recorded["status"] == 200
    assert ("Content-Type", "application/json") in recorded["headers"]  # type: ignore[operator]
    assert recorded["ended"] is True
    assert b'"x": 1' in recorded["body"]

    assert DockerLLMProxyHandler.log_message(dummy, "ignored") is None  # type: ignore[arg-type]


@pytest.mark.unit
def test_docker_proxy_handler_single_and_batched_can_use_broker_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _stub_request_completion(*_a, **_k):
        return ChatCompletion(
            root_model="m",
            prompt=_k.get("prompt"),
            response="ok",
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    monkeypatch.setattr(docker_mod, "request_completion", _stub_request_completion)

    dummy = SimpleNamespace(
        broker=None,
        broker_address=("127.0.0.1", 1234),
        pending_calls=[],
        lock=threading.Lock(),
        timeout_s=0.1,
    )
    out = DockerLLMProxyHandler._handle_single(dummy, {"prompt": "hi"})  # type: ignore[arg-type]
    assert out == {"response": "ok"}
    assert [c.response for c in dummy.pending_calls] == ["ok"]

    def _stub_request_completions_batched(*_a, **_k):
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

    monkeypatch.setattr(
        docker_mod,
        "request_completions_batched",
        _stub_request_completions_batched,
    )
    out2 = DockerLLMProxyHandler._handle_batched(dummy, {"prompts": ["p1", "p2"]})  # type: ignore[arg-type]
    assert out2 == {"responses": ["Error: oops", "r2"]}
    assert [c.response for c in dummy.pending_calls] == ["ok", "r2"]
