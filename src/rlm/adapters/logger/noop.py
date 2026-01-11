from __future__ import annotations

from rlm.adapters.base import BaseLoggerAdapter
from rlm.domain.models import Iteration, RunMetadata


class NoopLoggerAdapter(BaseLoggerAdapter):
    """Logger adapter that discards all events."""

    __slots__ = ()

    def log_metadata(self, metadata: RunMetadata, /) -> None:  # noqa: ARG002 - intentional
        return

    def log_iteration(self, iteration: Iteration, /) -> None:  # noqa: ARG002 - intentional
        return
