from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.api import create_rlm


@pytest.mark.e2e
def test_parallel_completions_do_not_conflict() -> None:
    """Boundary: two parallel runs should not conflict (broker ports/threads/env state).

    This is intentionally minimal and hermetic: it doesn't require docker or network.
    """

    def _run(i: int) -> str:
        rlm = create_rlm(
            MockLLMAdapter(model=f"m{i}", script=[f"FINAL(ok{i})"]),
            environment="local",
            max_iterations=2,
            verbose=False,
        )
        return rlm.completion("hello").response

    with ThreadPoolExecutor(max_workers=2) as ex:
        a = ex.submit(_run, 1)
        b = ex.submit(_run, 2)
        assert sorted([a.result(), b.result()]) == ["ok1", "ok2"]
