"""
Domain models (hexagonal core).

These are pure, dependency-free dataclasses that represent the core entities and
results flowing through the system (prompts, completions, execution results,
usage accounting, etc.).
"""

from __future__ import annotations

from rlm.domain.models.completion import ChatCompletion
from rlm.domain.models.iteration import CodeBlock, Iteration
from rlm.domain.models.repl import ReplResult
from rlm.domain.models.usage import ModelUsageSummary, UsageSummary

# NOTE: QueryMetadata will be added in phase-two

__all__ = [
    "ChatCompletion",
    "CodeBlock",
    "Iteration",
    "ModelUsageSummary",
    "ReplResult",
    "UsageSummary",
]
