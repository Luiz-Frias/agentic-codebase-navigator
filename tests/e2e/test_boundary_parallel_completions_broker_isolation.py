from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.api import create_rlm
from tests.live_llm import LiveLLMSettings


@pytest.mark.e2e
def test_parallel_completions_do_not_conflict(
    live_llm_settings: LiveLLMSettings | None,
) -> None:
    """
    Boundary: two parallel runs should not conflict (broker ports/threads/env state).

    This is intentionally minimal and hermetic: it doesn't require docker or network.
    """

    def _run(i: int) -> str:
        def echo(value: str) -> str:
            return value

        if live_llm_settings is not None:
            llm = live_llm_settings.build_openai_adapter(
                request_kwargs={"temperature": 0, "max_tokens": 32}
            )
            rlm = create_rlm(
                llm,
                environment="local",
                max_iterations=2,
                verbose=False,
                tools=[echo],
                agent_mode="tools",
            )
            return rlm.completion(f"Return a short response for run {i}.").response

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
        results = [a.result(), b.result()]
        assert all(isinstance(r, str) and r.strip() for r in results)
