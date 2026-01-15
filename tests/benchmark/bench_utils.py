from __future__ import annotations

from collections.abc import Callable
from typing import Any

PEDANTIC_KWARGS = {"iterations": 1, "rounds": 1, "warmup_rounds": 0}


def run_pedantic_once(benchmark: Any, func: Callable[[], Any]) -> Any:
    return benchmark.pedantic(func, **PEDANTIC_KWARGS)
