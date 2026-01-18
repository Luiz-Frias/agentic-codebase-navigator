from __future__ import annotations

import pytest

from rlm.api import create_rlm_from_config
from rlm.api.registries import DictLLMRegistry
from rlm.application.config import EnvironmentConfig, LLMConfig, RLMConfig
from tests.fakes_ports import QueueLLM


@pytest.mark.e2e
def test_create_rlm_from_config_runs_end_to_end_local() -> None:
    registry = DictLLMRegistry(
        builders={
            "mock": lambda _cfg: QueueLLM(
                responses=[
                    "```repl\nprint('HELLO_LOCAL')\n```",
                    "FINAL(ok)",
                ],
            ),
        },
    )  # type: ignore[arg-type]

    cfg = RLMConfig(
        llm=LLMConfig(backend="mock"),
        env=EnvironmentConfig(environment="local"),
        max_iterations=3,
        verbose=False,
    )

    rlm = create_rlm_from_config(cfg, llm_registry=registry)
    cc = rlm.completion("hello")
    assert cc.response == "ok"


@pytest.mark.e2e
@pytest.mark.docker
def test_create_rlm_from_config_runs_end_to_end_docker() -> None:
    registry = DictLLMRegistry(
        builders={
            "mock": lambda _cfg: QueueLLM(
                responses=[
                    "```repl\nprint('HELLO_DOCKER')\n```",
                    "FINAL(ok)",
                ],
            ),
        },
    )  # type: ignore[arg-type]

    cfg = RLMConfig(
        llm=LLMConfig(backend="mock"),
        env=EnvironmentConfig(
            environment="docker",
            environment_kwargs={"image": "python:3.12-slim"},
        ),
        max_iterations=3,
        verbose=False,
    )

    try:
        rlm = create_rlm_from_config(cfg, llm_registry=registry)
        cc = rlm.completion("hello")
        assert cc.response == "ok"
    except RuntimeError as e:
        # Docker can be present but image pulls may fail in restricted environments.
        if "Failed to start container" in str(e):
            pytest.skip(str(e))
        raise
