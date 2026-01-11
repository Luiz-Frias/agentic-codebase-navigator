from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

DIST_NAME = "agentic-codebase-navigator"

try:
    __version__ = version(DIST_NAME)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
