from __future__ import annotations

import pytest

from rlm.domain.errors import BrokerError
from rlm.domain.models import ChatCompletion, ModelUsageSummary, UsageSummary
from rlm.infrastructure.comms.protocol import request_completion, request_completions_batched


def _fake_cc_dict(prompt: object = "p", response: str = "ok") -> dict:
    usage = UsageSummary(model_usage_summaries={"m": ModelUsageSummary(1, 2, 3)})
    cc = ChatCompletion(
        root_model="m",
        prompt=prompt,
        response=response,
        usage_summary=usage,
        execution_time=0.01,
    )
    return cc.to_dict()


@pytest.mark.unit
def test_request_completion_returns_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub_request_response(*args, **kwargs):
        return {
            "correlation_id": "cid",
            "error": None,
            "results": [{"error": None, "chat_completion": _fake_cc_dict(response="FINAL(ok)")}],
        }

    import rlm.infrastructure.comms.protocol as proto

    monkeypatch.setattr(proto, "request_response", _stub_request_response)

    cc = request_completion(("127.0.0.1", 1234), "hello", correlation_id="cid")
    assert cc.response == "FINAL(ok)"


@pytest.mark.unit
def test_request_completion_raises_on_request_level_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub_request_response(*args, **kwargs):
        return {"correlation_id": "cid", "error": "bad request", "results": None}

    import rlm.infrastructure.comms.protocol as proto

    monkeypatch.setattr(proto, "request_response", _stub_request_response)

    with pytest.raises(BrokerError, match="bad request"):
        request_completion(("127.0.0.1", 1234), "hello", correlation_id="cid")


@pytest.mark.unit
def test_request_completions_batched_preserves_order_and_returns_per_item_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _stub_request_response(*args, **kwargs):
        return {
            "correlation_id": "cid",
            "error": None,
            "results": [
                {"error": None, "chat_completion": _fake_cc_dict(prompt="p1", response="r1")},
                {"error": "oops", "chat_completion": None},
                {"error": None, "chat_completion": _fake_cc_dict(prompt="p3", response="r3")},
            ],
        }

    import rlm.infrastructure.comms.protocol as proto

    monkeypatch.setattr(proto, "request_response", _stub_request_response)

    results = request_completions_batched(
        ("127.0.0.1", 1234), ["p1", "p2", "p3"], correlation_id="cid"
    )
    assert [r.error for r in results] == [None, "oops", None]
    assert results[0].chat_completion is not None and results[0].chat_completion.response == "r1"
    assert results[2].chat_completion is not None and results[2].chat_completion.response == "r3"


@pytest.mark.unit
def test_request_completions_batched_raises_on_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _stub_request_response(*args, **kwargs):
        return {
            "correlation_id": "cid",
            "error": None,
            "results": [{"error": None, "chat_completion": _fake_cc_dict(response="r1")}],
        }

    import rlm.infrastructure.comms.protocol as proto

    monkeypatch.setattr(proto, "request_response", _stub_request_response)

    with pytest.raises(BrokerError, match="expected 2 results, got 1"):
        request_completions_batched(("127.0.0.1", 1234), ["p1", "p2"], correlation_id="cid")
