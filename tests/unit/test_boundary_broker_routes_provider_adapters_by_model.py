from __future__ import annotations

import sys
import types

import pytest

from rlm.adapters.broker.tcp import TcpBrokerAdapter
from rlm.adapters.llm.anthropic import AnthropicAdapter
from rlm.adapters.llm.openai import OpenAIAdapter
from rlm.domain.errors import BrokerError
from rlm.infrastructure.comms.protocol import request_completion


class _OpenAIMessage:
    def __init__(self, content: str):
        self.content = content


class _OpenAIChoice:
    def __init__(self, content: str):
        self.message = _OpenAIMessage(content)


class _OpenAIUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _OpenAIResponse:
    def __init__(self, content: str, *, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.choices = [_OpenAIChoice(content)]
        self.usage = _OpenAIUsage(prompt_tokens, completion_tokens)


class _OpenAICompletions:
    def __init__(self, response: _OpenAIResponse):
        self._response = response

    def create(self, **kwargs):
        return self._response


class _OpenAIChat:
    def __init__(self, response: _OpenAIResponse):
        self.completions = _OpenAICompletions(response)


class _AnthropicUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _AnthropicBlock:
    def __init__(self, text: str):
        self.text = text


class _AnthropicResponse:
    def __init__(self, text: str, *, input_tokens: int = 0, output_tokens: int = 0):
        self.content = [_AnthropicBlock(text)]
        self.usage = _AnthropicUsage(input_tokens, output_tokens)


class _AnthropicMessages:
    def __init__(self, response: _AnthropicResponse):
        self._response = response

    def create(self, **kwargs):
        return self._response


@pytest.mark.unit
def test_broker_routes_to_openai_or_anthropic_adapters_by_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub provider SDKs (no network / no optional deps required).
    openai_resp = _OpenAIResponse("OPENAI", prompt_tokens=1, completion_tokens=2)

    class OpenAI:
        def __init__(self, **kwargs):
            self.chat = _OpenAIChat(openai_resp)

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = OpenAI
    fake_openai.AsyncOpenAI = object()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    anthropic_resp = _AnthropicResponse("ANTHROPIC", input_tokens=3, output_tokens=4)

    class Anthropic:
        def __init__(self, **kwargs):
            self.messages = _AnthropicMessages(anthropic_resp)

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = Anthropic
    fake_anthropic.AsyncAnthropic = object()
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    openai_llm = OpenAIAdapter(model="gpt-test", api_key="sk-test")
    anthropic_llm = AnthropicAdapter(model="claude-test", api_key="ak-test")

    broker = TcpBrokerAdapter(openai_llm)
    broker.register_llm("claude-test", anthropic_llm)

    addr = broker.start()
    try:
        cc_default = request_completion(addr, "hi")
        assert cc_default.root_model == "gpt-test"
        assert cc_default.response == "OPENAI"

        cc_anthropic = request_completion(addr, "hi", model="claude-test")
        assert cc_anthropic.root_model == "claude-test"
        assert cc_anthropic.response == "ANTHROPIC"

        with pytest.raises(BrokerError, match="Unknown model"):
            request_completion(addr, "hi", model="unknown")
    finally:
        broker.stop()
