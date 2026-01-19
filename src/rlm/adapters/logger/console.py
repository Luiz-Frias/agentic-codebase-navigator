from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rlm.adapters.base import BaseLoggerAdapter

if TYPE_CHECKING:
    from rlm.domain.models import CodeBlock, Iteration, RunMetadata


class ConsoleLoggerAdapter(BaseLoggerAdapter):
    """Minimal stdout logger (no external dependencies)."""

    __slots__ = ("enabled",)

    def __init__(self, *, enabled: bool = True) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("ConsoleLoggerAdapter requires enabled to be a bool")
        self.enabled = enabled

    def log_metadata(self, metadata: RunMetadata, /) -> None:
        if not self.enabled:
            return
        cid = metadata.correlation_id or "-"
        print(
            "[RLM] "
            f"cid={cid} "
            f"backend={metadata.backend} "
            f"root_model={metadata.root_model} "
            f"env={metadata.environment_type} "
            f"max_depth={metadata.max_depth} "
            f"max_iterations={metadata.max_iterations}",
        )

    def log_iteration(self, iteration: Iteration, /) -> None:
        if not self.enabled:
            return

        cid = iteration.correlation_id or "-"
        code_blocks = cast("list[CodeBlock]", iteration.code_blocks or [])
        subcalls = sum(len(cb.result.llm_calls) for cb in code_blocks)
        has_final = iteration.final_answer is not None
        print(
            "[RLM] "
            f"cid={cid} "
            f"iteration_time={iteration.iteration_time:.3f}s "
            f"code_blocks={len(code_blocks)} "
            f"subcalls={subcalls} "
            f"final={has_final}",
        )
