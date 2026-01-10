from __future__ import annotations

import pytest

from rlm.api import create_rlm_from_config
from rlm.application.config import EnvironmentConfig, LLMConfig, RLMConfig


@pytest.mark.e2e
def test_default_install_can_run_with_mock_llm_without_provider_deps() -> None:
    """
    This should work on a clean install with *no* provider SDKs installed.

    The mock LLM is dependency-free and supports scripting so the orchestrator
    can complete deterministically.
    """

    cfg = RLMConfig(
        llm=LLMConfig(
            backend="mock",
            model_name="mock-model",
            backend_kwargs={"script": ["FINAL(default_ok)"]},
        ),
        env=EnvironmentConfig(environment="local"),
        max_iterations=2,
        verbose=False,
    )
    rlm = create_rlm_from_config(cfg)
    cc = rlm.completion("hello")
    assert cc.response == "default_ok"
