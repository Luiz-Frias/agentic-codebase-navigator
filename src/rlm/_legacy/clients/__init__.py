"""
Legacy LLM client implementations (upstream mirror).

Provider implementations are intentionally *not* imported by default to keep the
base installation lightweight. During the Phase 1 legacy port we only need a
minimal router (`get_client`) for the legacy orchestrator; tests can patch this
function to inject a `MockLM`.
"""

from __future__ import annotations

from typing import Any

from rlm._legacy.clients.base_lm import BaseLM
from rlm._legacy.core.types import ClientBackend

__all__ = ["BaseLM", "get_client"]


def get_client(
    backend: ClientBackend,
    backend_kwargs: dict[str, Any],
) -> BaseLM:
    """
    Legacy client router.

    Upstream ships concrete clients for many providers, but in this refactor we
    keep provider integrations behind optional extras and will reintroduce them
    later via adapters. For now, this is intentionally not implemented.
    """

    raise NotImplementedError(
        "Legacy provider clients are not ported in Phase 1. "
        "Use a test MockLM (monkeypatch `rlm._legacy.clients.get_client`) or "
        "wait for provider adapters in later phases."
    )
