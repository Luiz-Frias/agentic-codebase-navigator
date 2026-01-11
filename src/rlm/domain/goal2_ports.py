"""
Goal2 placeholder ports (Phase 06).

These are intentionally *not* used by the Phase 05–06 completion/runtime path.
They exist only to reserve stable extension points for future codebase-analysis
capabilities, while keeping the Goal1 core independent (DIP).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict


class CodeChunk(TypedDict):
    file_path: str
    start_line: int
    end_line: int
    content: str


class CodebaseIndex(TypedDict):
    root: str
    files: list[str]


class CodebaseIndexerPort(Protocol):
    """Build a lightweight index over a codebase (placeholder)."""

    def build_index(self, repo_root: Path, /) -> CodebaseIndex: ...


class SemanticChunkerPort(Protocol):
    """Split file content into semantically meaningful chunks (placeholder)."""

    def chunk(self, text: str, /, *, file_path: str | None = None) -> list[CodeChunk]: ...


class ContextManagerPort(Protocol):
    """
    Select a relevant subset of a codebase index/chunks for a query (placeholder).

    Returns an opaque context payload for now; future phases will introduce typed
    result models once the public surface is finalized.
    """

    def select(
        self,
        query: str,
        index: CodebaseIndex,
        chunks: Sequence[CodeChunk] | None,
        /,
        *,
        strategy: Literal["sequential", "parallel", "hierarchical"] = "hierarchical",
        max_tokens: int = 100_000,
        depth: int = 0,
    ) -> dict[str, Any]: ...
