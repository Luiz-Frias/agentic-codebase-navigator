from __future__ import annotations

import pytest

from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.domain.errors import LLMError
from rlm.domain.models import LLMRequest


@pytest.mark.unit
def test_mock_llm_adapter_echo_is_deterministic_and_tracks_usage() -> None:
    llm = MockLLMAdapter(model="mock-model")

    cc1 = llm.complete(LLMRequest(prompt="hello world"))
    assert cc1.root_model == "mock-model"
    assert cc1.response.startswith("Mock response to: ")
    assert "hello world" in cc1.response

    last1 = llm.get_last_usage().model_usage_summaries["mock-model"]
    assert last1.total_calls == 1

    total1 = llm.get_usage_summary().model_usage_summaries["mock-model"]
    assert total1.total_calls == 1

    cc2 = llm.complete(LLMRequest(prompt="hello world"))
    assert cc2.response == cc1.response

    last2 = llm.get_last_usage().model_usage_summaries["mock-model"]
    assert last2.total_calls == 1

    total2 = llm.get_usage_summary().model_usage_summaries["mock-model"]
    assert total2.total_calls == 2


@pytest.mark.unit
def test_mock_llm_adapter_scripted_responses_pop_in_order() -> None:
    llm = MockLLMAdapter(model="dummy", script=["A", "B"])

    assert llm.complete(LLMRequest(prompt="p")).response == "A"
    assert llm.complete(LLMRequest(prompt="p")).response == "B"

    with pytest.raises(LLMError, match="no scripted responses left"):
        llm.complete(LLMRequest(prompt="p"))


@pytest.mark.unit
def test_mock_llm_adapter_script_can_raise_exceptions() -> None:
    llm = MockLLMAdapter(model="dummy", script=[ValueError("boom")])
    with pytest.raises(ValueError, match="boom"):
        llm.complete(LLMRequest(prompt="p"))
