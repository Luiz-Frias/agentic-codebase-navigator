from __future__ import annotations

import pytest

from rlm.adapters.logger.jsonl import JsonlLoggerAdapter


@pytest.mark.unit
def test_jsonl_logger_validates_ctor_args(tmp_path) -> None:
    with pytest.raises(ValueError, match="file_name"):
        JsonlLoggerAdapter(log_dir=tmp_path, file_name="")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rotate_per_run"):
        JsonlLoggerAdapter(log_dir=tmp_path, rotate_per_run="no")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="schema_version"):
        JsonlLoggerAdapter(log_dir=tmp_path, schema_version=0)
