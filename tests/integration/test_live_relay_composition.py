from __future__ import annotations

import importlib.util
import os
from urllib.parse import urlparse

import pytest

from rlm.adapters.llm.retry import RetryConfig
from rlm.application.relay.root_composer import RootAgentComposer
from rlm.domain.relay import InMemoryPipelineRegistry, PipelineTemplate, StateSpec
from tests.live_llm import load_env_files

load_env_files()

RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay_seconds=0.5,
    max_delay_seconds=4.0,
    jitter_seconds=0.25,
)


@pytest.mark.integration
@pytest.mark.live_llm
def test_live_root_composer_selects_pipeline() -> None:
    if importlib.util.find_spec("openai") is None:
        pytest.skip("openai package not installed")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_MODEL") or "gpt-5-nano"
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_version = os.environ.get("OPENAI_API_VERSION")

    if _is_azure_endpoint(base_url):
        if api_version is None:
            pytest.skip("OPENAI_API_VERSION not set for Azure OpenAI endpoint")
        from rlm.adapters.llm.azure_openai import AzureOpenAIAdapter

        llm = AzureOpenAIAdapter(
            deployment=model,
            api_key=api_key,
            endpoint=base_url,
            api_version=api_version,
            retry_config=RETRY_CONFIG,
            default_request_kwargs={
                "temperature": 0,
                "max_tokens": 16,
            },
        )
    else:
        from rlm.adapters.llm.openai import OpenAIAdapter

        llm = OpenAIAdapter(
            model=model,
            api_key=api_key,
            base_url=base_url,
            retry_config=RETRY_CONFIG,
            default_request_kwargs={
                "temperature": 0,
                "max_tokens": 16,
            },
        )

    registry = InMemoryPipelineRegistry()
    registry.register(
        PipelineTemplate(
            name="alpha",
            description="Use for alpha tasks.",
            input_type=str,
            output_type=str,
            factory=lambda: StateSpec(name="a", input_type=str, output_type=str)
            >> StateSpec(name="b", input_type=str, output_type=str),
        )
    )
    registry.register(
        PipelineTemplate(
            name="beta",
            description="Use for beta tasks.",
            input_type=str,
            output_type=str,
            factory=lambda: StateSpec(name="c", input_type=str, output_type=str)
            >> StateSpec(name="d", input_type=str, output_type=str),
        )
    )

    composer = RootAgentComposer(registry=registry, llm=llm)
    pipeline = composer.compose("Select the alpha pipeline")
    assert pipeline.entry_state is not None


def _is_azure_endpoint(base_url: str | None) -> bool:
    if not base_url:
        return False
    parsed = urlparse(base_url)
    host = parsed.netloc.lower() if parsed.netloc else base_url.lower()
    return "azure.com" in host or "cognitiveservices.azure.com" in host
