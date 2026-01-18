from __future__ import annotations

import sys
import types

import pytest

from rlm.domain.models import LLMRequest


class _UsageMetadata:
    def __init__(self, prompt_token_count: int, candidates_token_count: int):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count


class _Response:
    def __init__(self, text: str, *, prompt_tokens: int = 0, output_tokens: int = 0):
        self.text = text
        self.usage_metadata = _UsageMetadata(prompt_tokens, output_tokens)


class _FakeModels:
    def __init__(self, response: _Response):
        self._response = response
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self._response


class _FakeClient:
    def __init__(self, *, response: _Response, **kwargs):
        self.kwargs = dict(kwargs)
        self.models = _FakeModels(response)


@pytest.mark.unit
def test_gemini_adapter_complete_maps_prompt_and_extracts_text_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.gemini import GeminiAdapter

    resp = _Response("hi", prompt_tokens=3, output_tokens=5)
    created: list[_FakeClient] = []

    class Client:
        def __init__(self, **kwargs):
            client = _FakeClient(response=resp, **kwargs)
            created.append(client)

        @property
        def models(self):
            return created[-1].models

    google = types.ModuleType("google")
    google.__path__ = []  # make it package-like
    genai = types.ModuleType("google.genai")
    genai.Client = Client
    google.genai = genai

    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    llm = GeminiAdapter(model="gemini-test", api_key="gk-test")
    cc = llm.complete(LLMRequest(prompt="hello"))

    assert cc.root_model == "gemini-test"
    assert cc.response == "hi"
    mus = cc.usage_summary.model_usage_summaries["gemini-test"]
    assert mus.total_calls == 1
    assert mus.total_input_tokens == 3
    assert mus.total_output_tokens == 5

    call = created[-1].models.calls[-1]
    assert call["model"] == "gemini-test"
    assert call["contents"] == "hello"


@pytest.mark.unit
async def test_gemini_adapter_acomplete_runs_sync_path_in_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlm.adapters.llm.gemini import GeminiAdapter

    resp = _Response("ahi", prompt_tokens=1, output_tokens=2)

    class Client:
        def __init__(self, **kwargs):
            self.models = _FakeModels(resp)

    google = types.ModuleType("google")
    google.__path__ = []
    genai = types.ModuleType("google.genai")
    genai.Client = Client
    google.genai = genai

    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    llm = GeminiAdapter(model="gemini-test", api_key="gk-test")
    cc = await llm.acomplete(LLMRequest(prompt="hello"))

    assert cc.root_model == "gemini-test"
    assert cc.response == "ahi"
    mus = cc.usage_summary.model_usage_summaries["gemini-test"]
    assert mus.total_input_tokens == 1
    assert mus.total_output_tokens == 2
