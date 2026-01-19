from __future__ import annotations

import pytest

from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.api import create_rlm


@pytest.mark.e2e
def test_orchestrator_propagates_usage_summary_from_llm() -> None:
    """Boundary test: LLM usage accounting should propagate to the final completion."""
    llm = MockLLMAdapter(
        model="mock-model",
        script=[
            # Iteration 0: execute code but don't finalize.
            "```repl\nx = 1\n```\n",
            # Iteration 1: finalize.
            "FINAL(done)",
        ],
    )
    rlm = create_rlm(llm, environment="local", max_iterations=2, verbose=False)
    cc = rlm.completion("hello")

    assert cc.response == "done"
    mus = cc.usage_summary.model_usage_summaries["mock-model"]
    assert mus.total_calls == 2
