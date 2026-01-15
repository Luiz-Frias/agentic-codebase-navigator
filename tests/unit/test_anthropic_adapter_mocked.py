from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from rlm.domain.errors import LLMError
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
def test_anthropic_adapter_reports_tool_prompt_format() -> None:
    from rlm.adapters.llm.anthropic import AnthropicAdapter

    adapter = AnthropicAdapter(model="claude-test")
    assert adapter.tool_prompt_format == "anthropic"


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


@pytest.mark.unit
def test_anthropic_adapter_helpers_and_validations() -> None:
    from rlm.adapters.llm.anthropic import (
        AnthropicAdapter,
        _extract_text,
        _extract_usage_tokens,
        _messages_and_system,
        build_anthropic_adapter,
    )

    with pytest.raises(ValueError, match="non-empty 'model'"):
        build_anthropic_adapter(model="")
    with pytest.raises(ValueError, match="api_key must be a non-empty string"):
        build_anthropic_adapter(model="m", api_key=" ")

    msgs, system = _messages_and_system(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "system", "content": "ignored"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert system == "sys"
    assert msgs == [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]

    assert _extract_text({"content": [{"text": "hi"}]}) == "hi"

    class RespWithOutputText:
        output_text = "ok"

    assert _extract_text(RespWithOutputText()) == "ok"
    with pytest.raises(ValueError, match="missing content"):
        _extract_text({})

    assert _extract_usage_tokens({}) == (0, 0)
    assert _extract_usage_tokens({"usage": {"input_tokens": 1, "output_tokens": 2}}) == (1, 2)

    class Usage:
        input_tokens = "3"
        output_tokens = None

    class Resp:
        usage = Usage()

    assert _extract_usage_tokens(Resp()) == (3, 0)

    adapter = AnthropicAdapter(model="m")
    with pytest.raises(ImportError, match="expected `anthropic\\.Anthropic`"):
        adapter._get_client(SimpleNamespace())
    with pytest.raises(ImportError, match="expected `anthropic\\.AsyncAnthropic`"):
        adapter._get_async_client(SimpleNamespace())


@pytest.mark.unit
def test_anthropic_adapter_complete_error_mapping_and_client_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.anthropic import AnthropicAdapter

    created: list[_FakeClient] = []

    class Anthropic:  # noqa: N801 - matches SDK naming
        def __init__(self, **kwargs):
            client = _FakeClient(response=_Response("ok"), **kwargs)
            created.append(client)

        @property
        def messages(self):
            return created[-1].messages

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = Anthropic
    fake_anthropic.AsyncAnthropic = object()  # not used in this test
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    llm = AnthropicAdapter(model="claude-test", api_key="ak-test")
    cc1 = llm.complete(LLMRequest(prompt="hello"))
    cc2 = llm.complete(LLMRequest(prompt="hello"))
    assert cc1.response == "ok"
    assert cc2.response == "ok"
    # Client is cached.
    assert len(created) == 1

    class AnthropicTimeout:  # noqa: N801 - matches SDK naming
        def __init__(self, **_kwargs):
            self._messages = _FakeMessages(_Response("unused"))
            self._messages.create = lambda **_k: (_ for _ in ()).throw(TimeoutError())  # type: ignore[method-assign]

        @property
        def messages(self):
            return self._messages

    fake_anthropic2 = types.ModuleType("anthropic")
    fake_anthropic2.Anthropic = AnthropicTimeout
    fake_anthropic2.AsyncAnthropic = object()
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic2)

    with pytest.raises(LLMError, match="Anthropic request timed out"):
        AnthropicAdapter(model="claude-test").complete(LLMRequest(prompt="hello"))


@pytest.mark.unit
async def test_anthropic_adapter_acomplete_falls_back_to_thread_when_async_client_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.anthropic import AnthropicAdapter

    created: list[_FakeClient] = []

    class Anthropic:  # noqa: N801 - matches SDK naming
        def __init__(self, **kwargs):
            client = _FakeClient(response=_Response("ok"), **kwargs)
            created.append(client)

        @property
        def messages(self):
            return created[-1].messages

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = Anthropic
    # No AsyncAnthropic => acomplete should fall back to `asyncio.to_thread(self.complete, ...)`.
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    llm = AnthropicAdapter(model="claude-test")
    cc = await llm.acomplete(LLMRequest(prompt="hello"))
    assert cc.response == "ok"
