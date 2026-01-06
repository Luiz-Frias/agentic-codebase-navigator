"""
Public API layer (hexagonal entrypoints).

This will expose the stable user-facing surface (Python API, optional CLI).
"""

from __future__ import annotations

from rlm.api.factory import create_rlm, create_rlm_from_config
from rlm.api.rlm import RLM

__all__ = ["RLM", "create_rlm", "create_rlm_from_config"]
