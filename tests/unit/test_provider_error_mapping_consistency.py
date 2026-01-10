from __future__ import annotations

import sys
import types

import pytest

from rlm.domain.errors import LLMError
from rlm.domain.models import LLMRequest


@pytest.mark.unit
def test_anthropic_adapter_maps_provider_errors_without_leaking_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.anthropic import AnthropicAdapter

    secret = "ak-should-not-leak"

    class _BoomMessages:
        def create(self, **kwargs):
            raise ValueError(f"Incorrect API key provided: {secret}")

    class Anthropic:  # noqa: N801
        def __init__(self, **kwargs):
            self.messages = _BoomMessages()

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = Anthropic
    fake_anthropic.AsyncAnthropic = object()
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    llm = AnthropicAdapter(model="claude-test", api_key="x")
    with pytest.raises(LLMError) as excinfo:
        llm.complete(LLMRequest(prompt="hello"))

    msg = str(excinfo.value)
    assert "Anthropic" in msg
    assert secret not in msg


@pytest.mark.unit
def test_gemini_adapter_maps_provider_errors_without_leaking_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.gemini import GeminiAdapter

    secret = "gk-should-not-leak"

    class _BoomModels:
        def generate_content(self, **kwargs):
            raise ValueError(f"bad key: {secret}")

    class Client:  # noqa: N801
        def __init__(self, **kwargs):
            self.models = _BoomModels()

    google = types.ModuleType("google")
    google.__path__ = []
    genai = types.ModuleType("google.genai")
    genai.Client = Client
    google.genai = genai

    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    llm = GeminiAdapter(model="gemini-test", api_key="x")
    with pytest.raises(LLMError) as excinfo:
        llm.complete(LLMRequest(prompt="hello"))

    msg = str(excinfo.value)
    assert "Gemini" in msg
    assert secret not in msg


@pytest.mark.unit
def test_portkey_adapter_maps_provider_errors_without_leaking_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.portkey import PortkeyAdapter

    secret = "pk-should-not-leak"

    class _BoomCompletions:
        def create(self, **kwargs):
            raise ValueError(f"bad key: {secret}")

    class _BoomChat:
        completions = _BoomCompletions()

    class Portkey:  # noqa: N801
        def __init__(self, **kwargs):
            self.chat = _BoomChat()

    fake_portkey = types.ModuleType("portkey_ai")
    fake_portkey.Portkey = Portkey
    fake_portkey.AsyncPortkey = object()
    monkeypatch.setitem(sys.modules, "portkey_ai", fake_portkey)

    llm = PortkeyAdapter(model="portkey-test", api_key="x")
    with pytest.raises(LLMError) as excinfo:
        llm.complete(LLMRequest(prompt="hello"))

    msg = str(excinfo.value)
    assert "Portkey" in msg
    assert secret not in msg


@pytest.mark.unit
def test_litellm_adapter_maps_provider_errors_without_leaking_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.litellm import LiteLLMAdapter

    secret = "sk-should-not-leak"

    def completion(**kwargs):  # type: ignore[no-untyped-def]
        raise ValueError(f"Incorrect API key provided: {secret}")

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = completion
    fake_litellm.acompletion = object()
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    llm = LiteLLMAdapter(model="litellm-test")
    with pytest.raises(LLMError) as excinfo:
        llm.complete(LLMRequest(prompt="hello"))

    msg = str(excinfo.value)
    assert "LiteLLM" in msg
    assert secret not in msg


@pytest.mark.unit
def test_azure_openai_adapter_maps_provider_errors_without_leaking_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.azure_openai import AzureOpenAIAdapter

    secret = "azk-should-not-leak"

    class _BoomCompletions:
        def create(self, **kwargs):
            raise ValueError(f"Incorrect API key provided: {secret}")

    class _BoomChat:
        completions = _BoomCompletions()

    class AzureOpenAI:  # noqa: N801
        def __init__(self, **kwargs):
            self.chat = _BoomChat()

    fake_openai = types.ModuleType("openai")
    fake_openai.AzureOpenAI = AzureOpenAI
    fake_openai.AsyncAzureOpenAI = object()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    llm = AzureOpenAIAdapter(deployment="dep-test", api_key="x", endpoint="e", api_version="v")
    with pytest.raises(LLMError) as excinfo:
        llm.complete(LLMRequest(prompt="hello"))

    msg = str(excinfo.value)
    assert "Azure OpenAI" in msg
    assert secret not in msg
