"""
Environment adapters (hexagonal).

Native adapters live here. Legacy has been fully removed.
"""

from __future__ import annotations

from rlm.adapters.environments.docker import DockerEnvironmentAdapter
from rlm.adapters.environments.local import LocalEnvironmentAdapter

__all__ = ["DockerEnvironmentAdapter", "LocalEnvironmentAdapter"]
