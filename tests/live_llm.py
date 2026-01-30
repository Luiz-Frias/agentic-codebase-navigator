from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

DEFAULT_OPENAI_MODEL = "gpt-4-mini"
_ENV_FILES = ("local.env", ".env")


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        os.environ[key] = value


def load_env_files() -> None:
    for name in _ENV_FILES:
        _load_env_file(Path(name))


def live_llm_enabled() -> bool:
    raw = (os.environ.get("RLM_RUN_LIVE_LLM_TESTS") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class LiveLLMSettings:
    api_key: str
    base_url: str | None
    model: str
    model_sub: str
    api_version: str | None

    def build_openai_adapter(
        self,
        *,
        model: str | None = None,
        request_kwargs: dict[str, object] | None = None,
    ):
        from rlm.adapters.llm.openai import OpenAIAdapter

        kwargs = dict(request_kwargs or {})
        return OpenAIAdapter(
            model=model or self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            default_request_kwargs=kwargs,
        )


def get_live_llm_settings() -> LiveLLMSettings:
    load_env_files()

    try:
        import importlib.util

        if importlib.util.find_spec("openai") is None:
            pytest.skip("openai package not installed")
    except Exception:
        pytest.skip("openai package not available")

    api_key = os.environ.get("ACN_OPENAI_API_KEY")
    if not api_key:
        pytest.skip("ACN_OPENAI_API_KEY not set")

    model = os.environ.get("ACN_OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    base_url = os.environ.get("ACN_OPENAI_BASE_URL")
    api_version = os.environ.get("ACN_OPENAI_API_VERSION")
    model_sub = os.environ.get("ACN_OPENAI_MODEL_SUB") or model

    return LiveLLMSettings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        model_sub=model_sub,
        api_version=api_version,
    )
