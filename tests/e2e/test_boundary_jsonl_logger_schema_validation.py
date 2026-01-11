from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.adapters.logger.jsonl import JsonlLoggerAdapter
from rlm.api import create_rlm


@pytest.mark.e2e
def test_jsonl_logger_emits_schema_versioned_metadata_and_iteration(
    tmp_path: Path,
) -> None:
    """
    Boundary: public API -> use case -> logger emits schema-versioned JSONL lines.

    This validates the *artifact contract* a PyPI/uv consumer relies on.
    """

    logger = JsonlLoggerAdapter(log_dir=tmp_path)
    rlm = create_rlm(
        MockLLMAdapter(model="root", script=["FINAL(ok)"]),
        environment="local",
        max_iterations=2,
        verbose=False,
        logger=logger,
    )

    cc = rlm.completion("hello")
    assert cc.response == "ok"

    log_path = logger.log_file_path
    assert log_path is not None

    lines = Path(log_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2

    objs = [json.loads(line) for line in lines]
    assert all(obj.get("schema_version") == 1 for obj in objs)
    assert {obj.get("type") for obj in objs} >= {"metadata", "iteration"}

    metadata = next(obj for obj in objs if obj.get("type") == "metadata")
    assert metadata["root_model"] == "root"
    assert metadata["max_iterations"] == 2
    assert metadata["environment_type"] in {"local", "unknown"}  # best-effort inference

    iteration = next(obj for obj in objs if obj.get("type") == "iteration")
    assert iteration["iteration"] == 1
    assert isinstance(iteration.get("prompt"), (str, dict, list))
    assert iteration["response"] == "FINAL(ok)"
