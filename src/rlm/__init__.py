"""
rlm

This repository is migrating an upstream snapshot in `references/rlm/**` into a
src-layout Python package (`src/rlm/**`) and refactoring toward a hexagonal
modular monolith.
"""

from __future__ import annotations

from rlm._meta import __version__

__all__ = ["__version__"]
