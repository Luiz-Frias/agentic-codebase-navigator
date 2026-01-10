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


class _FakeCompletions:
    def __init__(self, response: _Response):
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self._response


class _FakeAsyncCompletions:
    def __init__(self, response: _Response):
        self._response = response
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self._response


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, *, response: _Response, **kwargs):
        self.kwargs = dict(kwargs)
        self._completions = _FakeCompletions(response)
        self.chat = _FakeChat(self._completions)

    @property
    def completions(self) -> _FakeCompletions:
        return self._completions


class _FakeAsyncClient:
    def __init__(self, *, response: _Response, **kwargs):
        self.kwargs = dict(kwargs)
        self._completions = _FakeAsyncCompletions(response)
        self.chat = _FakeChat(self._completions)

    @property
    def completions(self) -> _FakeAsyncCompletions:
        return self._completions


@pytest.mark.unit
def test_portkey_adapter_complete_maps_prompt_and_extracts_text_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.portkey import PortkeyAdapter

    resp = _Response("hi", prompt_tokens=3, completion_tokens=5)
    created: list[_FakeClient] = []

    class Portkey:  # noqa: N801 - matches SDK naming
        def __init__(self, **kwargs):
            client = _FakeClient(response=resp, **kwargs)
            created.append(client)

        @property
        def chat(self):
            return created[-1].chat

    fake_portkey = types.ModuleType("portkey_ai")
    fake_portkey.Portkey = Portkey
    fake_portkey.AsyncPortkey = object()  # not used in this test
    monkeypatch.setitem(sys.modules, "portkey_ai", fake_portkey)

    llm = PortkeyAdapter(model="portkey-test", api_key="pk-test")
    cc = llm.complete(LLMRequest(prompt="hello"))

    assert cc.root_model == "portkey-test"
    assert cc.response == "hi"
    mus = cc.usage_summary.model_usage_summaries["portkey-test"]
    assert mus.total_calls == 1
    assert mus.total_input_tokens == 3
    assert mus.total_output_tokens == 5

    call = created[-1].completions.calls[-1]
    assert call["model"] == "portkey-test"
    assert call["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.unit
async def test_portkey_adapter_acomplete_maps_prompt_and_extracts_text_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.portkey import PortkeyAdapter

    resp = _Response("ahi", prompt_tokens=1, completion_tokens=2)
    created: list[_FakeAsyncClient] = []

    class AsyncPortkey:  # noqa: N801 - matches SDK naming
        def __init__(self, **kwargs):
            client = _FakeAsyncClient(response=resp, **kwargs)
            created.append(client)

        @property
        def chat(self):
            return created[-1].chat

    fake_portkey = types.ModuleType("portkey_ai")
    fake_portkey.Portkey = object()  # not used in this test
    fake_portkey.AsyncPortkey = AsyncPortkey
    monkeypatch.setitem(sys.modules, "portkey_ai", fake_portkey)

    llm = PortkeyAdapter(model="portkey-test", api_key="pk-test")
    cc = await llm.acomplete(LLMRequest(prompt="hello"))

    assert cc.root_model == "portkey-test"
    assert cc.response == "ahi"
    mus = cc.usage_summary.model_usage_summaries["portkey-test"]
    assert mus.total_calls == 1
    assert mus.total_input_tokens == 1
    assert mus.total_output_tokens == 2

    call = created[-1].completions.calls[-1]
    assert call["model"] == "portkey-test"
    assert call["messages"] == [{"role": "user", "content": "hello"}]
