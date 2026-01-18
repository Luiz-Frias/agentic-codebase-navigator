from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlm.adapters.logger.jsonl import JsonlLoggerAdapter
from rlm.api.factory import create_rlm
from rlm.api.registries import ensure_docker_available
from rlm.domain.errors import ExecutionError
from tests.fakes_ports import QueueLLM


@pytest.mark.integration
@pytest.mark.docker
def test_docker_completion_jsonl_logs_include_rlm_calls(tmp_path: Path) -> None:
    """Integration: docker environment + broker subcalls are persisted in JSONL logs.

    This is best-effort and should skip cleanly if Docker isn't available or
    image pulls/container startup are blocked.
    """
    try:
        ensure_docker_available(timeout_s=0.5)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLoggerAdapter(log_dir=log_dir, file_name="docker_jsonl")

    llm = QueueLLM(
        model_name="dummy",
        responses=[
            "```repl\nresp = llm_query('ping')\n```\nFINAL_VAR('resp')",
            "pong",
        ],
    )
    rlm = create_rlm(llm, environment="docker", max_iterations=3, verbose=False, logger=logger)

    try:
        cc = rlm.completion("hello")
    except ExecutionError as exc:
        cause = exc.__cause__
        if cause is not None and "Failed to start container" in str(cause):
            pytest.skip(str(cause))
        raise
    except RuntimeError as exc:
        if "Failed to start container" in str(exc):
            pytest.skip(str(exc))
        raise

    assert cc.response == "pong"

    path = logger.log_file_path
    assert path is not None
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) >= 2
    meta = json.loads(lines[0])
    assert meta["type"] == "metadata"
    entry = json.loads(lines[1])
    assert entry["type"] == "iteration"
    assert entry["code_blocks"]

    result = entry["code_blocks"][0]["result"]
    assert result["rlm_calls"]
    assert result["rlm_calls"][0]["response"] == "pong"
