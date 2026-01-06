from __future__ import annotations

import pytest

from rlm.application.config import EnvironmentConfig, LLMConfig, RLMConfig


@pytest.mark.unit
def test_rlm_config_constructs() -> None:
    cfg = RLMConfig(
        llm=LLMConfig(backend="mock", model_name="dummy", backend_kwargs={"x": 1}),
        env=EnvironmentConfig(environment="local"),
        max_depth=1,
        max_iterations=2,
        verbose=False,
    )

    assert cfg.llm.backend == "mock"
    assert cfg.env.environment == "local"
