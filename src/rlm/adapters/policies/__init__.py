"""
Default policy implementations for RLM extension protocols.

These adapters provide sensible defaults that preserve backward compatibility
while enabling external apps to inject custom behavior.

Exports:
    DefaultStoppingPolicy: Max-iterations check only (default behavior).
    NoOpContextCompressor: Passthrough, no compression (default behavior).
    SimpleNestedCallPolicy: Always simple LLM call, no orchestration (default).
"""

from __future__ import annotations

from rlm.adapters.policies.compression import NoOpContextCompressor
from rlm.adapters.policies.nested import SimpleNestedCallPolicy
from rlm.adapters.policies.stopping import DefaultStoppingPolicy

__all__ = [
    "DefaultStoppingPolicy",
    "NoOpContextCompressor",
    "SimpleNestedCallPolicy",
]
