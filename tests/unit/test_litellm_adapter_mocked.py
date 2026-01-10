from __future__ import annotations

import sys
import types

import pytest

from rlm.domain.models import LLMRequest


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Response:
    def __init__(self, content: str, *, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.choices = [_Choice(content)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


@pytest.mark.unit
def test_litellm_adapter_complete_maps_prompt_and_extracts_text_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.litellm import LiteLLMAdapter

    resp = _Response("hi", prompt_tokens=3, completion_tokens=5)
    calls: list[dict] = []

    def completion(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(dict(kwargs))
        return resp

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = completion
    fake_litellm.acompletion = object()  # not used in this test
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    llm = LiteLLMAdapter(model="litellm-test")
    cc = llm.complete(LLMRequest(prompt="hello"))

    assert cc.root_model == "litellm-test"
    assert cc.response == "hi"
    mus = cc.usage_summary.model_usage_summaries["litellm-test"]
    assert mus.total_calls == 1
    assert mus.total_input_tokens == 3
    assert mus.total_output_tokens == 5

    call = calls[-1]
    assert call["model"] == "litellm-test"
    assert call["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.unit
async def test_litellm_adapter_acomplete_maps_prompt_and_extracts_text_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.litellm import LiteLLMAdapter

    resp = _Response("ahi", prompt_tokens=1, completion_tokens=2)
    calls: list[dict] = []

    async def acompletion(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(dict(kwargs))
        return resp

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = object()  # not used in this test
    fake_litellm.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    llm = LiteLLMAdapter(model="litellm-test")
    cc = await llm.acomplete(LLMRequest(prompt="hello"))

    assert cc.root_model == "litellm-test"
    assert cc.response == "ahi"
    mus = cc.usage_summary.model_usage_summaries["litellm-test"]
    assert mus.total_calls == 1
    assert mus.total_input_tokens == 1
    assert mus.total_output_tokens == 2

    call = calls[-1]
    assert call["model"] == "litellm-test"
    assert call["messages"] == [{"role": "user", "content": "hello"}]
