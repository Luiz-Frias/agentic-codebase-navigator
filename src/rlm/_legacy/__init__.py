"""
Upstream snapshot port (temporary).

During Phase 1 we will mirror the upstream `references/rlm/rlm/**` package into
`src/rlm/_legacy/**`, adjusting only imports and Python 3.12 compatibility.

Runtime code should migrate toward the hexagonal layers in `src/rlm/*` and stop
depending on `_legacy` over time.
"""

from __future__ import annotations

from rlm._legacy.core.rlm import RLM

__all__ = ["RLM"]
