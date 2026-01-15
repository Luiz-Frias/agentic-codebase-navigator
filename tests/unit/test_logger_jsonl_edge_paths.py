from __future__ import annotations

from pathlib import Path

import pytest

from rlm.adapters.logger.jsonl import JsonlLoggerAdapter
from rlm.domain.models import Iteration, RunMetadata


@pytest.mark.unit
def test_jsonl_logger_log_iteration_before_metadata_starts_new_run(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = JsonlLoggerAdapter(log_dir=log_dir, file_name="edge", rotate_per_run=False)
    logger.log_iteration(Iteration(prompt="p", response="r", iteration_time=0.0))

    path = logger.log_file_path
    assert path is not None
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


@pytest.mark.unit
def test_jsonl_logger_log_metadata_is_idempotent_when_rotate_per_run_false(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = JsonlLoggerAdapter(log_dir=log_dir, file_name="edge", rotate_per_run=False)
    md = RunMetadata(
        root_model="m",
        max_depth=1,
        max_iterations=1,
        backend="openai",
        backend_kwargs={},
        environment_type="local",
        environment_kwargs={},
        other_backends=None,
        correlation_id="cid",
    )
    logger.log_metadata(md)
    logger.log_metadata(md)

    path = logger.log_file_path
    assert path is not None
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
