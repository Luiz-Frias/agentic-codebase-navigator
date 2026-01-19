"""
Infrastructure layer (hexagonal).

Cross-cutting technical utilities (protocols, filesystem/time helpers, etc.).
Must not import adapters.
"""

from __future__ import annotations

from rlm.infrastructure.logging import warn_cleanup_failure

__all__ = [
    "warn_cleanup_failure",
]
