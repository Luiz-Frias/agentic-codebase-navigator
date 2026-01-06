"""
rlm

This repository is migrating an upstream snapshot in `references/rlm/**` into a
src-layout Python package (`src/rlm/**`) and refactoring toward a hexagonal
modular monolith.
"""

from __future__ import annotations

from rlm._meta import __version__
from rlm.api.rlm import RLM

__all__ = ["RLM", "__version__"]
