from __future__ import annotations

from collections.abc import Mapping, Sequence

# -----------------------------------------------------------------------------
# Core domain type aliases (type-driven boundary pattern)
# -----------------------------------------------------------------------------

# Prompt payloads can be:
# - a raw string
# - a dict payload (legacy) - uses `object` to force narrowing before use
# - an OpenAI-style sequence of message dicts (common)
#
# Note: Using `object` instead of `Any` enforces type-driven boundary pattern:
# callers must narrow (isinstance) before accessing nested values.
# Using Sequence/Mapping (covariant) instead of list/dict for flexibility.
type Prompt = str | Mapping[str, object] | Sequence[Mapping[str, object]]

# Context passed into environments. This mirrors the upstream snapshot and is
# intentionally broad - uses `object` to enforce validation at boundaries.
type ContextPayload = Mapping[str, object] | Sequence[object] | str
