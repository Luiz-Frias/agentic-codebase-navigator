from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from rlm.domain.types import ContextPayload

ContextType = Literal["str", "dict", "list"]


@dataclass(frozen=True, slots=True)
class QueryMetadata:
    """
    Metadata about a query/context payload.

    This is the domain-owned counterpart to the legacy `QueryMetadata` used to
    build the initial system prompt. It is intentionally dependency-free and
    computes:
    - a per-chunk length breakdown
    - the total length
    - a coarse-grained context type
    """

    context_lengths: list[int]
    context_total_length: int
    context_type: ContextType

    @classmethod
    def from_context(cls, context: ContextPayload, /) -> QueryMetadata:
        """
        Compute metadata for a context payload.

        Semantics are intentionally aligned with the upstream/legacy computation
        so prompt behavior remains stable during the migration.
        """
        if isinstance(context, str):
            lengths = [len(context)]
            ctx_type: ContextType = "str"
            return cls(
                context_lengths=lengths,
                context_total_length=sum(lengths),
                context_type=ctx_type,
            )

        if isinstance(context, dict):
            lengths: list[int] = []
            for chunk in context.values():
                if isinstance(chunk, str):
                    lengths.append(len(chunk))
                    continue
                try:
                    lengths.append(len(json.dumps(chunk, default=str)))
                except Exception:
                    lengths.append(len(repr(chunk)))
            ctx_type = "dict"
            return cls(
                context_lengths=lengths,
                context_total_length=sum(lengths),
                context_type=ctx_type,
            )

        if isinstance(context, list):
            ctx_type = "list"
            if len(context) == 0:
                lengths = [0]
                return cls(
                    context_lengths=lengths,
                    context_total_length=0,
                    context_type=ctx_type,
                )

            first = context[0]
            if isinstance(first, dict):
                # Chat-style message list (OpenAI-ish).
                if "content" in first:
                    lengths = [
                        len(str(chunk.get("content", "")))  # type: ignore[union-attr]
                        for chunk in context
                    ]
                else:
                    lengths = []
                    for chunk in context:
                        try:
                            lengths.append(len(json.dumps(chunk, default=str)))
                        except Exception:
                            lengths.append(len(repr(chunk)))
            else:
                # Treat as list[str]-like.
                lengths = [len(chunk) for chunk in context]  # type: ignore[arg-type]

            return cls(
                context_lengths=lengths,
                context_total_length=sum(lengths),
                context_type=ctx_type,
            )

        raise ValueError(f"Invalid context type: {type(context)}")
