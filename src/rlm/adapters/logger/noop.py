from __future__ import annotations

from typing import TYPE_CHECKING

from rlm.adapters.base import BaseLoggerAdapter

if TYPE_CHECKING:
    from rlm.domain.models import Iteration, RunMetadata


class NoopLoggerAdapter(BaseLoggerAdapter):
    """Logger adapter that discards all events."""

    __slots__ = ()

    def log_metadata(self, metadata: RunMetadata, /) -> None:
        return

    def log_iteration(self, iteration: Iteration, /) -> None:
        return
