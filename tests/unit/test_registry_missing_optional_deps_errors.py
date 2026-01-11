from __future__ import annotations

import builtins
import sys

import pytest

from rlm.api.registries import DefaultLLMRegistry
from rlm.application.config import LLMConfig
from rlm.domain.models import LLMRequest


def _block_imports(monkeypatch: pytest.MonkeyPatch, *, blocked: set[str]) -> None:
    orig_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name in blocked or any(str(name).startswith(prefix + ".") for prefix in blocked):
            raise ModuleNotFoundError(name)
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)


@pytest.mark.unit
def test_registry_openai_backend_raises_helpful_error_if_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "openai", raising=False)
    _block_imports(monkeypatch, blocked={"openai"})

    llm = DefaultLLMRegistry().build(LLMConfig(backend="openai", model_name="gpt-test"))
    with pytest.raises(ImportError, match=r"\[llm-openai\]"):
        llm.complete(LLMRequest(prompt="hello"))


@pytest.mark.unit
def test_registry_anthropic_backend_raises_helpful_error_if_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "anthropic", raising=False)
    _block_imports(monkeypatch, blocked={"anthropic"})

    llm = DefaultLLMRegistry().build(LLMConfig(backend="anthropic", model_name="claude-test"))
    with pytest.raises(ImportError, match=r"\[llm-anthropic\]"):
        llm.complete(LLMRequest(prompt="hello"))


@pytest.mark.unit
def test_registry_gemini_backend_raises_helpful_error_if_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "google", raising=False)
    monkeypatch.delitem(sys.modules, "google.genai", raising=False)
    _block_imports(monkeypatch, blocked={"google"})

    llm = DefaultLLMRegistry().build(LLMConfig(backend="gemini", model_name="gemini-test"))
    with pytest.raises(ImportError, match=r"\[llm-gemini\]"):
        llm.complete(LLMRequest(prompt="hello"))


@pytest.mark.unit
def test_registry_portkey_backend_raises_helpful_error_if_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "portkey_ai", raising=False)
    _block_imports(monkeypatch, blocked={"portkey_ai"})

    llm = DefaultLLMRegistry().build(LLMConfig(backend="portkey", model_name="portkey-test"))
    with pytest.raises(ImportError, match=r"\[llm-portkey\]"):
        llm.complete(LLMRequest(prompt="hello"))


@pytest.mark.unit
def test_registry_litellm_backend_raises_helpful_error_if_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "litellm", raising=False)
    _block_imports(monkeypatch, blocked={"litellm"})

    llm = DefaultLLMRegistry().build(LLMConfig(backend="litellm", model_name="litellm-test"))
    with pytest.raises(ImportError, match=r"\[llm-litellm\]"):
        llm.complete(LLMRequest(prompt="hello"))


@pytest.mark.unit
def test_registry_azure_openai_backend_raises_helpful_error_if_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "openai", raising=False)
    _block_imports(monkeypatch, blocked={"openai"})

    llm = DefaultLLMRegistry().build(
        LLMConfig(backend="azure_openai", model_name="deployment-test")
    )
    with pytest.raises(ImportError, match=r"\[llm-azure-openai\]"):
        llm.complete(LLMRequest(prompt="hello"))
