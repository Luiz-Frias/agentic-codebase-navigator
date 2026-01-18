from __future__ import annotations

from typing import Any, TypeAlias

# -----------------------------------------------------------------------------
# Core domain type aliases (stable-ish during Phase 2)
# -----------------------------------------------------------------------------

# Prompt payloads can be:
# - a raw string
# - a dict payload (legacy)
# - an OpenAI-style list of message dicts (common)
Prompt: TypeAlias = str | dict[str, Any] | list[dict[str, Any]]

# Context passed into environments. This mirrors the upstream snapshot and is
# intentionally broad early in the refactor.
ContextPayload: TypeAlias = dict[str, Any] | list[Any] | str
