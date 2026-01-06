from __future__ import annotations

from typing import Any

from rlm._legacy.logger.rlm_logger import RLMLogger


class LegacyLoggerAdapter:
    """Adapter: legacy `RLMLogger` -> domain `LoggerPort`."""

    def __init__(self, logger: RLMLogger):
        self._logger = logger

    def log_metadata(self, metadata: Any, /) -> None:
        # Legacy logger accepts `RLMMetadata`.
        self._logger.log_metadata(metadata)  # type: ignore[arg-type]

    def log_iteration(self, iteration: Any, /) -> None:
        # Legacy logger uses `.log(iteration)` (not `.log_iteration`).
        self._logger.log(iteration)  # type: ignore[arg-type]
