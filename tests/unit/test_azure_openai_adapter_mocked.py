from __future__ import annotations

import sys
import types
from types import SimpleNamespace

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
def test_azure_openai_adapter_complete_maps_prompt_and_extracts_text_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.azure_openai import AzureOpenAIAdapter

    resp = _Response("hi", prompt_tokens=3, completion_tokens=5)
    created: list[_FakeClient] = []

    class AzureOpenAI:  # noqa: N801 - matches SDK naming
        def __init__(self, **kwargs):
            client = _FakeClient(response=resp, **kwargs)
            created.append(client)

        @property
        def chat(self):
            return created[-1].chat

    fake_openai = types.ModuleType("openai")
    fake_openai.AzureOpenAI = AzureOpenAI
    fake_openai.AsyncAzureOpenAI = object()  # not used in this test
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    llm = AzureOpenAIAdapter(
        deployment="dep-test",
        api_key="azk-test",
        endpoint="https://example.openai.azure.com",
        api_version="2024-06-01",
    )
    cc = llm.complete(LLMRequest(prompt="hello"))

    assert cc.root_model == "dep-test"
    assert cc.response == "hi"
    mus = cc.usage_summary.model_usage_summaries["dep-test"]
    assert mus.total_calls == 1
    assert mus.total_input_tokens == 3
    assert mus.total_output_tokens == 5

    call = created[-1].completions.calls[-1]
    assert call["model"] == "dep-test"
    assert call["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.unit
async def test_azure_openai_adapter_acomplete_maps_prompt_and_extracts_text_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.azure_openai import AzureOpenAIAdapter

    resp = _Response("ahi", prompt_tokens=1, completion_tokens=2)
    created: list[_FakeAsyncClient] = []

    class AsyncAzureOpenAI:  # noqa: N801 - matches SDK naming
        def __init__(self, **kwargs):
            client = _FakeAsyncClient(response=resp, **kwargs)
            created.append(client)

        @property
        def chat(self):
            return created[-1].chat

    fake_openai = types.ModuleType("openai")
    fake_openai.AzureOpenAI = object()  # not used in this test
    fake_openai.AsyncAzureOpenAI = AsyncAzureOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    llm = AzureOpenAIAdapter(
        deployment="dep-test",
        api_key="azk-test",
        endpoint="https://example.openai.azure.com",
        api_version="2024-06-01",
    )
    cc = await llm.acomplete(LLMRequest(prompt="hello"))

    assert cc.root_model == "dep-test"
    assert cc.response == "ahi"
    mus = cc.usage_summary.model_usage_summaries["dep-test"]
    assert mus.total_calls == 1
    assert mus.total_input_tokens == 1
    assert mus.total_output_tokens == 2

    call = created[-1].completions.calls[-1]
    assert call["model"] == "dep-test"
    assert call["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.unit
def test_azure_openai_adapter_validations_client_cache_and_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.azure_openai import (
        AzureOpenAIAdapter,
        build_azure_openai_adapter,
    )

    with pytest.raises(ValueError, match="non-empty 'deployment'"):
        build_azure_openai_adapter(deployment="")

    adapter = AzureOpenAIAdapter(deployment="dep")
    with pytest.raises(ImportError, match="expected `openai\\.AzureOpenAI`"):
        adapter._get_client(SimpleNamespace())
    with pytest.raises(ImportError, match="expected `openai\\.AsyncAzureOpenAI`"):
        adapter._get_async_client(SimpleNamespace())

    resp = _Response("ok", prompt_tokens=1, completion_tokens=2)
    created: list[_FakeClient] = []

    class AzureOpenAI:  # noqa: N801 - matches SDK naming
        def __init__(self, **kwargs):
            client = _FakeClient(response=resp, **kwargs)
            created.append(client)

        @property
        def chat(self):
            return created[-1].chat

    fake_openai = types.ModuleType("openai")
    fake_openai.AzureOpenAI = AzureOpenAI
    fake_openai.AsyncAzureOpenAI = object()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    llm = AzureOpenAIAdapter(
        deployment="dep-test",
        api_key="k",
        endpoint="https://example.openai.azure.com",
        api_version="2024-06-01",
    )
    cc1 = llm.complete(LLMRequest(prompt="hello"))
    cc2 = llm.complete(LLMRequest(prompt="hello"))
    assert cc1.response == "ok"
    assert cc2.response == "ok"
    assert len(created) == 1  # cached client

    # Provider exception => mapped to LLMError.
    class AzureOpenAITimeout:  # noqa: N801 - matches SDK naming
        def __init__(self, **_kwargs):
            client = _FakeClient(response=resp, **_kwargs)
            client.completions.create = lambda **_k: (_ for _ in ()).throw(TimeoutError())  # type: ignore[method-assign]
            created2.append(client)

        @property
        def chat(self):
            return created2[-1].chat

    created2: list[_FakeClient] = []
    fake_openai2 = types.ModuleType("openai")
    fake_openai2.AzureOpenAI = AzureOpenAITimeout
    fake_openai2.AsyncAzureOpenAI = object()
    monkeypatch.setitem(sys.modules, "openai", fake_openai2)

    with pytest.raises(LLMError, match="Azure OpenAI request timed out"):
        AzureOpenAIAdapter(deployment="dep-test").complete(LLMRequest(prompt="hello"))
