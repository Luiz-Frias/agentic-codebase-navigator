"""
Minimal verbose output for the legacy port.

Upstream used `rich` for pretty printing. We keep the legacy orchestrator
importable without pulling in optional UI dependencies by providing a small,
print-based implementation here.
"""

from __future__ import annotations

from typing import Any

from rlm._legacy.core.types import RLMIteration, RLMMetadata


class VerbosePrinter:
    """Lightweight verbose printer (stdout-based)."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def print_metadata(self, metadata: RLMMetadata) -> None:
        if not self.enabled:
            return
        model = metadata.backend_kwargs.get("model_name", "unknown")
        print(
            "[RLM] "
            f"backend={metadata.backend} "
            f"model={model} "
            f"env={metadata.environment_type} "
            f"max_depth={metadata.max_depth} "
            f"max_iterations={metadata.max_iterations}"
        )

    def print_iteration(self, iteration: RLMIteration, iteration_num: int) -> None:
        if not self.enabled:
            return
        code_blocks = iteration.code_blocks or []
        subcalls = sum(len(cb.result.rlm_calls) for cb in code_blocks)
        print(
            f"[RLM] iter={iteration_num} time={iteration.iteration_time:.3f}s "
            if iteration.iteration_time
            else f"[RLM] iter={iteration_num} "
        )
        print(f"[RLM] code_blocks={len(code_blocks)} subcalls={subcalls}")

    def print_final_answer(self, answer: Any) -> None:
        if not self.enabled:
            return
        print("[RLM] FINAL:", answer)

    def print_summary(
        self,
        total_iterations: int,
        total_time: float,
        usage_summary: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        print(f"[RLM] done iterations={total_iterations} total_time={total_time:.2f}s")
        if usage_summary:
            print(f"[RLM] usage={usage_summary}")
