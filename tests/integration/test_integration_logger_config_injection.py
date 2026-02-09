from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlm.api import create_rlm_from_config
from rlm.application.config import EnvironmentConfig, LLMConfig, LoggerConfig, RLMConfig
from tests.live_llm import LiveLLMSettings


@pytest.mark.integration
def test_create_rlm_from_config_can_build_jsonl_logger_and_write_events(
    tmp_path: Path,
    live_llm_settings: LiveLLMSettings | None,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg = RLMConfig(
        llm=LLMConfig(
            backend="mock",
            model_name="mock-model",
            backend_kwargs={"script": ["FINAL(ok)"]},
        ),
        env=EnvironmentConfig(environment="local"),
        logger=LoggerConfig(
            logger="jsonl",
            logger_kwargs={"log_dir": str(log_dir), "file_name": "cfg_run"},
        ),
        max_iterations=2,
        verbose=False,
    )
    if live_llm_settings is not None:
        def echo(value: str) -> str:
            return value

        llm = live_llm_settings.build_openai_adapter(
            request_kwargs={"temperature": 0, "max_tokens": 32}
        )
        live_cfg = RLMConfig(
            llm=LLMConfig(backend="openai", model_name=live_llm_settings.model),
            env=cfg.env,
            logger=cfg.logger,
            max_iterations=cfg.max_iterations,
            verbose=cfg.verbose,
            agent_mode="tools",
        )
        rlm = create_rlm_from_config(live_cfg, llm=llm, tools=[echo])
        cc = rlm.completion("Return exactly the word ok.")
        assert cc.response.strip()
    else:
        rlm = create_rlm_from_config(cfg)
        cc = rlm.completion("hello")
        assert cc.response == "ok"

    files = sorted(log_dir.glob("cfg_run_*.jsonl"))
    assert len(files) == 1
    path = files[0]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2

    meta = json.loads(lines[0])
    assert meta["type"] == "metadata"
    if live_llm_settings is not None:
        assert meta["root_model"] == live_llm_settings.model
    else:
        assert meta["root_model"] == "mock-model"

    it0 = json.loads(lines[1])
    assert it0["type"] == "iteration"
