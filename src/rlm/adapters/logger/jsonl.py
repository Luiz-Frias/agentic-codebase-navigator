from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rlm.adapters.base import BaseLoggerAdapter

if TYPE_CHECKING:
    from collections.abc import Callable

    from rlm.domain.models import Iteration, RunMetadata


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class JsonlLoggerAdapter(BaseLoggerAdapter):
    """
    JSONL logger adapter.

    Schema (versioned, line-oriented):
    - metadata entry:
        {"schema_version": 1, "type": "metadata", "timestamp": "...", ...RunMetadata.to_dict()}
    - iteration entry:
        {"schema_version": 1, "type": "iteration", "iteration": 1, "timestamp": "...", ...Iteration.to_dict()}

    Notes:
    - Writes are streaming (one line per call); no in-memory buffering of events.
    - By default we rotate to a new file per `log_metadata(...)` call to keep one run per file.

    """

    __slots__ = (
        "_file_name",
        "_iteration_count",
        "_lock",
        "_log_dir",
        "_log_file_path",
        "_metadata_logged",
        "_now_fn",
        "_rotate_per_run",
        "_schema_version",
    )

    def __init__(
        self,
        *,
        log_dir: str | Path,
        file_name: str = "rlm",
        rotate_per_run: bool = True,
        schema_version: int = 1,
        now_fn: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError("JsonlLoggerAdapter requires file_name to be a non-empty string")
        if not isinstance(rotate_per_run, bool):
            raise ValueError("JsonlLoggerAdapter requires rotate_per_run to be a bool")
        if not isinstance(schema_version, int) or schema_version < 1:
            raise ValueError("JsonlLoggerAdapter requires schema_version to be an int >= 1")

        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._file_name = file_name
        self._rotate_per_run = rotate_per_run
        self._schema_version = schema_version
        self._now_fn = now_fn or _utc_now_iso

        self._log_file_path: Path | None = None
        self._iteration_count = 0
        self._metadata_logged = False
        self._lock = threading.Lock()

    @property
    def log_file_path(self) -> str | None:
        """Absolute path of the active log file (or None if no events have been written yet)."""
        p = self._log_file_path
        return str(p) if p is not None else None

    def _start_new_run(self) -> None:
        ts = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
        run_id = uuid.uuid4().hex[:8]
        self._log_file_path = self._log_dir / f"{self._file_name}_{ts}_{run_id}.jsonl"
        self._iteration_count = 0
        self._metadata_logged = False

    def _ensure_path(self) -> Path:
        if self._log_file_path is None:
            self._start_new_run()
        assert self._log_file_path is not None
        return self._log_file_path

    def _write_entry(self, entry: dict[str, object]) -> None:
        path = self._ensure_path()
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")

    def log_metadata(self, metadata: RunMetadata, /) -> None:
        with self._lock:
            if self._rotate_per_run or self._log_file_path is None:
                self._start_new_run()
            if self._metadata_logged:
                return
            metadata_dict = metadata.to_dict()
            entry: dict[str, object] = {
                "schema_version": self._schema_version,
                "type": "metadata",
                "timestamp": self._now_fn(),
                **metadata_dict,
            }
            self._write_entry(entry)
            self._metadata_logged = True

    def log_iteration(self, iteration: Iteration, /) -> None:
        with self._lock:
            self._ensure_path()
            self._iteration_count += 1
            iteration_dict = iteration.to_dict()
            entry: dict[str, object] = {
                "schema_version": self._schema_version,
                "type": "iteration",
                "iteration": self._iteration_count,
                "timestamp": self._now_fn(),
                **iteration_dict,
            }
            self._write_entry(entry)
