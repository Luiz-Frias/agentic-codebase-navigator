"""
Query metadata computation using type-safe dispatch.

This module computes metadata about context payloads using the TypeMapper pattern
for clean type dispatch and the Result pattern for safe serialization fallbacks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from rlm.domain.models.result import Result, try_call
from rlm.domain.models.type_mapping import TypeMapper

if TYPE_CHECKING:
    from rlm.domain.types import ContextPayload

type ContextType = Literal["str", "dict", "list"]


def _safe_serialize_length(value: object) -> int:
    """
    Compute serialized length of a value, with fallback to repr.

    Uses Result pattern to safely attempt json.dumps, falling back to repr()
    if serialization fails.
    """
    result: Result[str, Exception] = try_call(lambda: json.dumps(value, default=str))
    return len(result.unwrap_or(repr(value)))


# ============================================================================
# TypeMapper-compatible handlers (must accept object, narrow internally)
# ============================================================================


def _compute_str_metadata(value: object) -> tuple[list[int], int, ContextType]:
    """Handle string context - simplest case."""
    # TypeMapper guarantees this is str via isinstance check before dispatch
    assert isinstance(value, str)  # noqa: S101
    lengths = [len(value)]
    return lengths, sum(lengths), "str"


def _compute_dict_metadata(value: object) -> tuple[list[int], int, ContextType]:
    """Handle dict context - compute length per value."""
    # TypeMapper guarantees this is dict via isinstance check before dispatch
    assert isinstance(value, dict)  # noqa: S101
    # Explicitly iterate values as objects
    lengths: list[int] = []
    for chunk in value.values():
        chunk_obj: object = chunk
        if isinstance(chunk_obj, str):
            lengths.append(len(chunk_obj))
        else:
            lengths.append(_safe_serialize_length(chunk_obj))
    return lengths, sum(lengths), "dict"


def _compute_list_metadata(value: object) -> tuple[list[int], int, ContextType]:
    """Handle list context - chat messages or string list."""
    # TypeMapper guarantees this is list via isinstance check before dispatch
    assert isinstance(value, list)  # noqa: S101

    if len(value) == 0:
        return [0], 0, "list"

    first: object = value[0]
    lengths: list[int] = []

    if isinstance(first, dict) and "content" in first:
        # Chat-style message list (OpenAI-ish)
        for item in value:
            item_obj: object = item
            if isinstance(item_obj, dict):
                raw_content = item_obj.get("content", "")
                lengths.append(len(str(raw_content)))
            else:
                lengths.append(len(str(item_obj)))
    elif isinstance(first, dict):
        # Generic dict list - serialize each
        lengths.extend(_safe_serialize_length(item) for item in value)
    else:
        # Treat as list[str]-like
        lengths.extend(len(str(item)) for item in value)

    return lengths, sum(lengths), "list"


# Build the context type mapper once at module load
_context_mapper: TypeMapper[object, tuple[list[int], int, ContextType]] = (
    TypeMapper[object, tuple[list[int], int, ContextType]]()
    .register(str, _compute_str_metadata)
    .register(dict, _compute_dict_metadata)
    .register(list, _compute_list_metadata)
)


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

        Uses TypeMapper for clean dispatch based on context type (str, dict, list).
        """
        if not _context_mapper.can_handle(context):
            raise ValueError(f"Invalid context type: {type(context)}")

        lengths, total, ctx_type = _context_mapper.map(context)
        return cls(
            context_lengths=lengths,
            context_total_length=total,
            context_type=ctx_type,
        )
