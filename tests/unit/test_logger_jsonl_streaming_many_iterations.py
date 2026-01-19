from __future__ import annotations

from pathlib import Path

import pytest

from rlm.adapters.logger.jsonl import JsonlLoggerAdapter
from rlm.domain.models import Iteration, RunMetadata


@pytest.mark.unit
def test_jsonl_logger_can_write_many_iterations_without_buffering(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = JsonlLoggerAdapter(log_dir=log_dir, file_name="many_iters")
    # Contract: the adapter uses `__slots__` (no accidental in-memory buffering via __dict__).
    assert not hasattr(logger, "__dict__")

    logger.log_metadata(
        RunMetadata(
            root_model="root",
            max_depth=1,
            max_iterations=10_000,
            backend="mock",
            backend_kwargs={},
            environment_type="local",
            environment_kwargs={},
            other_backends=None,
            correlation_id="cid-123",
        ),
    )

    for _ in range(2_000):
        logger.log_iteration(Iteration(prompt="p", response="r", iteration_time=0.0))

    path = logger.log_file_path
    assert path is not None
    with open(path) as f:
        lines = [line for line in f if line.strip()]

    # metadata + N iteration entries
    assert len(lines) == 2_001
