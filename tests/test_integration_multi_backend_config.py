from __future__ import annotations

import pytest

from rlm.api import create_rlm_from_config
from rlm.application.config import EnvironmentConfig, LLMConfig, RLMConfig


@pytest.mark.integration
def test_create_rlm_from_config_registers_other_llms_and_env_can_route_by_model() -> None:
    """
    Integration: config -> registries -> RLM facade -> broker -> local env.

    Root LLM emits a code block that calls `llm_query(..., model=<other>)`. The
    nested call should route to the configured `other_llms` entry.
    """

    root_script = "```repl\nresp = llm_query('ping', model='sub')\n```\nFINAL_VAR('resp')"

    cfg = RLMConfig(
        llm=LLMConfig(backend="mock", model_name="root", backend_kwargs={"script": [root_script]}),
        other_llms=[
            LLMConfig(backend="mock", model_name="sub", backend_kwargs={"script": ["pong"]})
        ],
        env=EnvironmentConfig(environment="local"),
        max_iterations=3,
        verbose=False,
    )

    rlm = create_rlm_from_config(cfg)
    cc = rlm.completion("hello")
    assert cc.response == "pong"
