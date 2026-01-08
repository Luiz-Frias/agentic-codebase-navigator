from __future__ import annotations

from dataclasses import dataclass

from rlm.domain.types import Prompt


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """
    A typed LLM request.

    This is a Phase 2 bridge model used by ports/adapters and the new orchestrator.
    """

    prompt: Prompt
    model: str | None = None


@dataclass(frozen=True, slots=True)
class BatchedLLMRequest:
    """A typed batched LLM request (ordered)."""

    prompts: list[Prompt]
    model: str | None = None
