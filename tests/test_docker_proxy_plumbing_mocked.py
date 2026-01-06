from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlm._legacy.core.comms_utils import LMResponse
from rlm._legacy.core.types import ModelUsageSummary, RLMChatCompletion, UsageSummary


def _chat_completion(response: str) -> RLMChatCompletion:
    return RLMChatCompletion(
        root_model="dummy",
        prompt="p",
        response=response,
        usage_summary=UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(1, 0, 0)}),
        execution_time=0.0,
    )


@pytest.mark.unit
def test_llm_proxy_handler_single_requires_lm_handler_address() -> None:
    from rlm._legacy.environments.docker_repl import LLMProxyHandler

    dummy = SimpleNamespace(lm_handler_address=None, pending_calls=[], lock=None)
    out = LLMProxyHandler._handle_single(dummy, {"prompt": "hi"})  # type: ignore[arg-type]
    assert out == {"error": "No LM handler configured"}


@pytest.mark.unit
def test_llm_proxy_handler_single_routes_request_and_records_pending_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import rlm._legacy.environments.docker_repl as docker_mod

    recorded: dict[str, object] = {}

    def _fake_send_lm_request(address, request, timeout: int = 300):  # noqa: ARG001
        recorded["address"] = address
        recorded["request"] = request
        return LMResponse.success_response(_chat_completion("ok"))

    monkeypatch.setattr(docker_mod, "send_lm_request", _fake_send_lm_request)

    dummy = SimpleNamespace(
        lm_handler_address=("127.0.0.1", 5555),
        pending_calls=[],
        lock=threading.Lock(),
    )

    out = docker_mod.LLMProxyHandler._handle_single(dummy, {"prompt": "ping", "model": "m1"})
    assert out == {"response": "ok"}
    assert recorded["address"] == ("127.0.0.1", 5555)
    assert recorded["request"].prompt == "ping"
    assert recorded["request"].model == "m1"
    assert len(dummy.pending_calls) == 1
    assert dummy.pending_calls[0].response == "ok"


@pytest.mark.unit
def test_llm_proxy_handler_batched_routes_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import rlm._legacy.environments.docker_repl as docker_mod

    recorded: dict[str, object] = {}

    def _fake_send_lm_request_batched(address, prompts, model=None, timeout: int = 300):  # noqa: ARG001
        recorded["address"] = address
        recorded["prompts"] = prompts
        recorded["model"] = model
        return [
            LMResponse.success_response(_chat_completion("r1")),
            LMResponse.error_response("boom"),
            LMResponse.success_response(_chat_completion("r3")),
        ]

    monkeypatch.setattr(docker_mod, "send_lm_request_batched", _fake_send_lm_request_batched)

    dummy = SimpleNamespace(
        lm_handler_address=("127.0.0.1", 7777),
        pending_calls=[],
        lock=threading.Lock(),
    )

    out = docker_mod.LLMProxyHandler._handle_batched(
        dummy, {"prompts": ["a", "b", "c"], "model": "m2"}
    )
    assert out == {"responses": ["r1", "Error: boom", "r3"]}
    assert recorded["address"] == ("127.0.0.1", 7777)
    assert recorded["prompts"] == ["a", "b", "c"]
    assert recorded["model"] == "m2"

    # Only successful responses are recorded as pending calls.
    assert [c.response for c in dummy.pending_calls] == ["r1", "r3"]
