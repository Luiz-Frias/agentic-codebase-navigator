from __future__ import annotations

import pytest

from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.api import create_rlm


@pytest.mark.integration
def test_facade_multi_backend_routes_subcalls_by_model_and_merges_usage() -> None:
    """
    Boundary: public facade -> application use case -> broker -> local env -> subcall routing.

    The root model emits a repl code block that calls `llm_query(..., model='sub')`.
    """
    root_script = "```repl\nresp = llm_query('ping', model='sub')\n```\nFINAL_VAR('resp')"

    rlm = create_rlm(
        MockLLMAdapter(model="root", script=[root_script]),
        other_llms=[MockLLMAdapter(model="sub", script=["pong"])],
        environment="local",
        max_iterations=2,
        verbose=False,
    )

    cc = rlm.completion("hello")
    assert cc.root_model == "root"
    assert cc.response == "pong"

    # Usage should include both the root LM call and the nested subcall.
    assert cc.usage_summary.model_usage_summaries["root"].total_calls == 1
    assert cc.usage_summary.model_usage_summaries["sub"].total_calls == 1
