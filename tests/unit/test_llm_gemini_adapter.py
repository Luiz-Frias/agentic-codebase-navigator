from __future__ import annotations

from types import SimpleNamespace

import pytest

import rlm.adapters.llm.gemini as gemini_mod
from rlm.adapters.llm.gemini import (
    GeminiAdapter,
    _extract_text,
    _extract_usage_tokens,
    build_gemini_adapter,
)
from rlm.domain.errors import LLMError
from rlm.domain.models import LLMRequest


@pytest.mark.unit
def test_build_gemini_adapter_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty 'model'"):
        build_gemini_adapter(model="")
    with pytest.raises(ValueError, match="api_key must be a non-empty string"):
        build_gemini_adapter(model="m", api_key=" ")


@pytest.mark.unit
def test_extract_text_supports_multiple_shapes() -> None:
    assert _extract_text({"text": "hi"}) == "hi"

    class Resp:
        text = "hello"

    assert _extract_text(Resp()) == "hello"

    payload = {"candidates": [{"content": {"parts": [{"text": "p0"}]}}]}
    assert _extract_text(payload) == "p0"

    with pytest.raises(ValueError, match="missing text"):
        _extract_text({})


@pytest.mark.unit
def test_extract_usage_tokens_supports_multiple_shapes() -> None:
    assert _extract_usage_tokens({}) == (0, 0)

    assert _extract_usage_tokens(
        {"usage_metadata": {"prompt_token_count": 1, "candidates_token_count": 2}}
    ) == (
        1,
        2,
    )
    assert _extract_usage_tokens(
        {"usage_metadata": {"input_token_count": "3", "output_token_count": "4"}}
    ) == (
        3,
        4,
    )

    class Usage:
        prompt_token_count = 5
        candidates_token_count = 6

    class Resp:
        usage_metadata = Usage()

    assert _extract_usage_tokens(Resp()) == (5, 6)

    class BadUsage:
        input_token_count = "nope"
        output_token_count = None

    class Resp2:
        usage_metadata = BadUsage()

    assert _extract_usage_tokens(Resp2()) == (0, 0)


@pytest.mark.unit
def test_gemini_adapter_complete_success_error_and_client_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict] = []

    class _Models:
        def __init__(self, *, resp, exc: Exception | None = None) -> None:
            self._resp = resp
            self._exc = exc

        def generate_content(self, **_kwargs):
            if self._exc is not None:
                raise self._exc
            return self._resp

    class Client:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.models = _Models(
                resp={
                    "text": "ok",
                    "usage_metadata": {"prompt_token_count": 1, "candidates_token_count": 2},
                }
            )

    dummy_genai = SimpleNamespace(Client=Client)
    monkeypatch.setattr(gemini_mod, "_require_google_genai", lambda: dummy_genai)

    adapter = GeminiAdapter(model="m", api_key="k")
    cc1 = adapter.complete(LLMRequest(prompt="hi"))
    cc2 = adapter.complete(LLMRequest(prompt="hi"))
    assert cc1.response == "ok"
    assert cc2.response == "ok"
    # Client created once and cached.
    assert created == [{"api_key": "k"}]

    class ClientTimeout:
        def __init__(self, **_kwargs):
            self.models = _Models(resp={}, exc=TimeoutError())

    dummy_genai2 = SimpleNamespace(Client=ClientTimeout)
    monkeypatch.setattr(gemini_mod, "_require_google_genai", lambda: dummy_genai2)

    with pytest.raises(LLMError, match="Gemini request timed out"):
        GeminiAdapter(model="m").complete(LLMRequest(prompt="hi"))

    class ClientBadText:
        def __init__(self, **_kwargs):
            self.models = _Models(resp={})

    dummy_genai3 = SimpleNamespace(Client=ClientBadText)
    monkeypatch.setattr(gemini_mod, "_require_google_genai", lambda: dummy_genai3)

    with pytest.raises(ValueError, match="missing text"):
        # _extract_text raises ValueError; adapter does not wrap it (intentional).
        GeminiAdapter(model="m").complete(LLMRequest(prompt="hi"))


@pytest.mark.unit
def test_gemini_adapter_get_client_requires_expected_sdk_api() -> None:
    adapter = GeminiAdapter(model="m")
    with pytest.raises(ImportError, match="expected `google\\.genai\\.Client`"):
        adapter._get_client(SimpleNamespace())


@pytest.mark.unit
async def test_gemini_adapter_acomplete_runs_sync_path_in_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        def __init__(self, **_kwargs):
            self.models = SimpleNamespace(
                generate_content=lambda **_k: {"text": "ok"}  # noqa: ARG005
            )

    dummy_genai = SimpleNamespace(Client=Client)
    monkeypatch.setattr(gemini_mod, "_require_google_genai", lambda: dummy_genai)

    cc = await GeminiAdapter(model="m").acomplete(LLMRequest(prompt="hi"))
    assert cc.response == "ok"
