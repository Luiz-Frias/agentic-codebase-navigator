"""
Domain models (hexagonal core).

These are pure, dependency-free dataclasses that represent the core entities and
results flowing through the system (prompts, completions, execution results,
usage accounting, etc.).
"""

from __future__ import annotations

from rlm.domain.models.completion import ChatCompletion
from rlm.domain.models.iteration import CodeBlock, Iteration
from rlm.domain.models.llm_request import BatchedLLMRequest, LLMRequest
from rlm.domain.models.model_spec import (
    ModelRoutingRules,
    ModelSpec,
    build_routing_rules,
)
from rlm.domain.models.query_metadata import QueryMetadata
from rlm.domain.models.repl import ReplResult
from rlm.domain.models.run_metadata import RunMetadata
from rlm.domain.models.usage import ModelUsageSummary, UsageSummary

__all__ = [
    "BatchedLLMRequest",
    "ChatCompletion",
    "CodeBlock",
    "Iteration",
    "LLMRequest",
    "ModelRoutingRules",
    "ModelSpec",
    "ModelUsageSummary",
    "QueryMetadata",
    "ReplResult",
    "RunMetadata",
    "UsageSummary",
    "build_routing_rules",
]
