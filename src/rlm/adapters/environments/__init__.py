"""
Environment adapters (hexagonal).

Phase 1 note:
- These adapters are thin wrappers around the legacy environment implementations.
- They exist to establish the target seams; later phases will replace the legacy
  envs with native implementations.
"""

from __future__ import annotations

from rlm.adapters.environments.docker import LegacyDockerEnvironmentAdapter

__all__ = ["LegacyDockerEnvironmentAdapter"]
