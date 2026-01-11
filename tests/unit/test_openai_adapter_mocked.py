from __future__ import annotations

import sys
import types

import pytest

from rlm.domain.errors import LLMError
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


class _FakeChat:
    def __init__(self, completions: _FakeCompletions):
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, *, response: _Response, **kwargs):
        self.kwargs = dict(kwargs)
        self._completions = _FakeCompletions(response)
        self.chat = _FakeChat(self._completions)

    @property
    def completions(self) -> _FakeCompletions:
        return self._completions


class _FakeAsyncCompletions:
    def __init__(self, response: _Response):
        self._response = response
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self._response


class _FakeAsyncChat:
    def __init__(self, completions: _FakeAsyncCompletions):
        self.completions = completions


class _FakeAsyncOpenAIClient:
    def __init__(self, *, response: _Response, **kwargs):
        self.kwargs = dict(kwargs)
        self._completions = _FakeAsyncCompletions(response)
        self.chat = _FakeAsyncChat(self._completions)

    @property
    def completions(self) -> _FakeAsyncCompletions:
        return self._completions


@pytest.mark.unit
def test_openai_adapter_complete_maps_prompt_and_extracts_text_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.openai import OpenAIAdapter

    resp = _Response("hi", prompt_tokens=3, completion_tokens=5)
    created: list[_FakeOpenAIClient] = []

    class OpenAI:  # noqa: N801 - matches SDK naming
        def __init__(self, **kwargs):
            client = _FakeOpenAIClient(response=resp, **kwargs)
            created.append(client)

        @property
        def chat(self):
            return created[-1].chat

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = OpenAI
    fake_openai.AsyncOpenAI = object()  # not used in this test
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    llm = OpenAIAdapter(model="gpt-test", api_key="sk-test")
    cc = llm.complete(LLMRequest(prompt="hello"))

    assert cc.root_model == "gpt-test"
    assert cc.response == "hi"
    mus = cc.usage_summary.model_usage_summaries["gpt-test"]
    assert mus.total_calls == 1
    assert mus.total_input_tokens == 3
    assert mus.total_output_tokens == 5

    # Verify the adapter sends chat-completions with a user message.
    assert created, "expected OpenAI client to be constructed"
    call = created[-1].completions.calls[-1]
    assert call["model"] == "gpt-test"
    assert call["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.unit
async def test_openai_adapter_acomplete_maps_prompt_and_extracts_text_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.openai import OpenAIAdapter

    resp = _Response("ahi", prompt_tokens=1, completion_tokens=2)
    created: list[_FakeAsyncOpenAIClient] = []

    class AsyncOpenAI:  # noqa: N801 - matches SDK naming
        def __init__(self, **kwargs):
            client = _FakeAsyncOpenAIClient(response=resp, **kwargs)
            created.append(client)

        @property
        def chat(self):
            return created[-1].chat

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = object()  # not used in this test
    fake_openai.AsyncOpenAI = AsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    llm = OpenAIAdapter(model="gpt-test", api_key="sk-test")
    cc = await llm.acomplete(LLMRequest(prompt="hello"))

    assert cc.root_model == "gpt-test"
    assert cc.response == "ahi"
    mus = cc.usage_summary.model_usage_summaries["gpt-test"]
    assert mus.total_calls == 1
    assert mus.total_input_tokens == 1
    assert mus.total_output_tokens == 2

    call = created[-1].completions.calls[-1]
    assert call["model"] == "gpt-test"
    assert call["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.unit
def test_openai_adapter_maps_provider_errors_to_safe_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.openai import OpenAIAdapter

    class AuthenticationError(Exception):
        pass

    class _BoomCompletions:
        def create(self, **kwargs):
            raise AuthenticationError("Incorrect API key provided: sk-should-not-leak")

    class _BoomChat:
        completions = _BoomCompletions()

    class OpenAI:  # noqa: N801
        def __init__(self, **kwargs):
            pass

        chat = _BoomChat()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = OpenAI
    fake_openai.AsyncOpenAI = object()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    llm = OpenAIAdapter(model="gpt-test", api_key="sk-test")
    with pytest.raises(LLMError, match=r"OpenAI authentication failed"):
        llm.complete(LLMRequest(prompt="hello"))


@pytest.mark.unit
def test_openai_adapter_usage_accumulates_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    from rlm.adapters.llm.openai import OpenAIAdapter

    resp = _Response("hi", prompt_tokens=3, completion_tokens=5)

    class OpenAI:  # noqa: N801 - matches SDK naming
        def __init__(self, **kwargs):
            self._client = _FakeOpenAIClient(response=resp, **kwargs)

        @property
        def chat(self):
            return self._client.chat

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = OpenAI
    fake_openai.AsyncOpenAI = object()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    llm = OpenAIAdapter(model="gpt-test", api_key="sk-test")
    llm.complete(LLMRequest(prompt="hello"))
    llm.complete(LLMRequest(prompt="hello"))

    total = llm.get_usage_summary().model_usage_summaries["gpt-test"]
    assert total.total_calls == 2
    assert total.total_input_tokens == 6
    assert total.total_output_tokens == 10

    last = llm.get_last_usage().model_usage_summaries["gpt-test"]
    assert last.total_calls == 1
    assert last.total_input_tokens == 3
    assert last.total_output_tokens == 5
