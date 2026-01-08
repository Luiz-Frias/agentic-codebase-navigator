from __future__ import annotations

from rlm._legacy.logger.rlm_logger import RLMLogger
from rlm.adapters.base import BaseLoggerAdapter
from rlm.domain.models import Iteration, RunMetadata


class LegacyLoggerAdapter(BaseLoggerAdapter):
    """Adapter: legacy `RLMLogger` -> domain `LoggerPort`."""

    def __init__(self, logger: RLMLogger):
        self._logger = logger

    def log_metadata(self, metadata: RunMetadata, /) -> None:
        # Legacy logger only requires `.to_dict()` shape; domain RunMetadata matches it.
        self._logger.log_metadata(metadata)  # type: ignore[arg-type]

    def log_iteration(self, iteration: Iteration, /) -> None:
        # Legacy logger uses `.log(iteration)` (not `.log_iteration`).
        self._logger.log(iteration)  # type: ignore[arg-type]
