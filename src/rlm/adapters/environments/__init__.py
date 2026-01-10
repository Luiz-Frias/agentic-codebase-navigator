"""
Environment adapters (hexagonal).

Phase 1 note:
- Native adapters live here.
- Legacy environment implementations live under `rlm._legacy` and are used only
  as a migration bridge during the refactor (until Phase 05+ is complete).
"""

from __future__ import annotations

from rlm.adapters.environments.docker import DockerEnvironmentAdapter
from rlm.adapters.environments.local import LocalEnvironmentAdapter

__all__ = ["DockerEnvironmentAdapter", "LocalEnvironmentAdapter"]
