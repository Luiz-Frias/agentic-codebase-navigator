from __future__ import annotations

import pytest

from rlm.domain.models import ChatCompletion
from rlm.domain.models.usage import UsageSummary
from rlm.infrastructure.comms.messages import WireRequest, WireResponse, WireResult


@pytest.mark.unit
def test_wire_request_prompt_variants_and_is_batched() -> None:
    # dict prompt (legacy JSON-y dict)
    req = WireRequest.from_dict({"prompt": {"x": 1}})
    assert isinstance(req.prompt, dict)

    # list-of-dict prompt (OpenAI-style messages)
    req2 = WireRequest.from_dict({"prompt": [{"role": "user", "content": "hi"}]})
    assert isinstance(req2.prompt, list)

    # batched prompts
    req3 = WireRequest.from_dict({"prompts": ["a", {"x": 1}]})
    assert req3.is_batched is True

    assert WireRequest(prompts=[]).is_batched is False


@pytest.mark.unit
def test_wire_request_validations() -> None:
    with pytest.raises(TypeError, match="correlation_id must be a string"):
        WireRequest.from_dict({"prompt": "hi", "correlation_id": 1})

    with pytest.raises(TypeError, match="model must be a string"):
        WireRequest.from_dict({"prompt": "hi", "model": 1})

    with pytest.raises(ValueError, match="only one of 'prompt' or 'prompts'"):
        WireRequest.from_dict({"prompt": "hi", "prompts": ["x"]})

    with pytest.raises(TypeError, match="prompt must be a valid Prompt"):
        WireRequest.from_dict({"prompt": 123})  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="prompts must be a list"):
        WireRequest.from_dict({"prompts": "nope"})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must not be empty"):
        WireRequest.from_dict({"prompts": []})

    with pytest.raises(TypeError, match=r"prompts\[0\] must be a valid Prompt"):
        WireRequest.from_dict({"prompts": [123]})  # type: ignore[list-item]


@pytest.mark.unit
def test_wire_result_success_property_and_validations() -> None:
    assert WireResult(error=None, chat_completion=None).success is True
    assert WireResult(error="x", chat_completion=None).success is False

    with pytest.raises(ValueError, match="Unknown keys"):
        WireResult.from_dict({"nope": 1})

    with pytest.raises(TypeError, match="error must be a string"):
        WireResult.from_dict({"error": 1})

    with pytest.raises(ValueError, match="either 'error' or 'chat_completion'"):
        WireResult.from_dict({"error": None, "chat_completion": None})

    with pytest.raises(ValueError, match="cannot include both"):
        WireResult.from_dict({"error": "x", "chat_completion": {"root_model": "m"}})

    with pytest.raises(TypeError, match="chat_completion must be a dict"):
        WireResult.from_dict({"chat_completion": "nope"})  # type: ignore[arg-type]

    cc = ChatCompletion(
        root_model="m",
        prompt="p",
        response="r",
        usage_summary=UsageSummary(model_usage_summaries={}),
        execution_time=0.0,
    )
    ok = WireResult.from_dict({"chat_completion": cc.to_dict()})
    assert ok.chat_completion is not None
    assert ok.error is None


@pytest.mark.unit
def test_wire_response_success_property_and_validations() -> None:
    assert WireResponse(error=None, results=None).success is True
    assert WireResponse(error="x", results=None).success is False

    with pytest.raises(ValueError, match="Unknown keys"):
        WireResponse.from_dict({"nope": 1})

    with pytest.raises(TypeError, match="correlation_id must be a string"):
        WireResponse.from_dict({"correlation_id": 1, "error": "x"})

    with pytest.raises(TypeError, match="error must be a string"):
        WireResponse.from_dict({"error": 1})

    with pytest.raises(ValueError, match="cannot include both"):
        WireResponse.from_dict({"error": "x", "results": []})

    with pytest.raises(TypeError, match="results must be a list"):
        WireResponse.from_dict({"results": "nope"})  # type: ignore[arg-type]
