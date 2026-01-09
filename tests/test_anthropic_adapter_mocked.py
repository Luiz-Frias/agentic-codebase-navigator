from __future__ import annotations

import sys
import types

import pytest

from rlm.domain.models import LLMRequest


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Block:
    def __init__(self, text: str):
        self.text = text


class _Response:
    def __init__(self, text: str, *, input_tokens: int = 0, output_tokens: int = 0):
        self.content = [_Block(text)]
        self.usage = _Usage(input_tokens, output_tokens)


class _FakeMessages:
    def __init__(self, response: _Response):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self._response


class _FakeClient:
    def __init__(self, *, response: _Response, **kwargs):
        self.kwargs = dict(kwargs)
        self.messages = _FakeMessages(response)


class _FakeAsyncMessages:
    def __init__(self, response: _Response):
        self._response = response
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self._response


class _FakeAsyncClient:
    def __init__(self, *, response: _Response, **kwargs):
        self.kwargs = dict(kwargs)
        self.messages = _FakeAsyncMessages(response)


@pytest.mark.unit
def test_anthropic_adapter_complete_maps_prompt_and_extracts_text_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.anthropic import AnthropicAdapter

    resp = _Response("hi", input_tokens=3, output_tokens=5)
    created: list[_FakeClient] = []

    class Anthropic:  # noqa: N801 - matches SDK naming
        def __init__(self, **kwargs):
            client = _FakeClient(response=resp, **kwargs)
            created.append(client)

        @property
        def messages(self):
            return created[-1].messages

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = Anthropic
    fake_anthropic.AsyncAnthropic = object()  # not used in this test
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    llm = AnthropicAdapter(model="claude-test", api_key="ak-test")
    cc = llm.complete(
        LLMRequest(
            prompt=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ]
        )
    )

    assert cc.root_model == "claude-test"
    assert cc.response == "hi"
    mus = cc.usage_summary.model_usage_summaries["claude-test"]
    assert mus.total_calls == 1
    assert mus.total_input_tokens == 3
    assert mus.total_output_tokens == 5

    assert created, "expected Anthropic client to be constructed"
    call = created[-1].messages.calls[-1]
    assert call["model"] == "claude-test"
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert call["max_tokens"] == 1024


@pytest.mark.unit
async def test_anthropic_adapter_acomplete_maps_prompt_and_extracts_text_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.anthropic import AnthropicAdapter

    resp = _Response("ahi", input_tokens=1, output_tokens=2)
    created: list[_FakeAsyncClient] = []

    class AsyncAnthropic:  # noqa: N801 - matches SDK naming
        def __init__(self, **kwargs):
            client = _FakeAsyncClient(response=resp, **kwargs)
            created.append(client)

        @property
        def messages(self):
            return created[-1].messages

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = object()  # not used in this test
    fake_anthropic.AsyncAnthropic = AsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    llm = AnthropicAdapter(model="claude-test", api_key="ak-test")
    cc = await llm.acomplete(LLMRequest(prompt="hello"))

    assert cc.root_model == "claude-test"
    assert cc.response == "ahi"
    mus = cc.usage_summary.model_usage_summaries["claude-test"]
    assert mus.total_calls == 1
    assert mus.total_input_tokens == 1
    assert mus.total_output_tokens == 2

    call = created[-1].messages.calls[-1]
    assert call["model"] == "claude-test"
    assert call["messages"] == [{"role": "user", "content": "hello"}]
