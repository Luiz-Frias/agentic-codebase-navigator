from __future__ import annotations

import pytest

from rlm.api.factory import create_rlm
from tests.fakes_ports import CollectingLogger, QueueLLM


@pytest.mark.e2e
def test_local_env_code_can_call_llm_query_and_result_is_returned_via_final_var() -> None:
    """
    Integration: domain orchestrator + local environment + broker.

    The LLM produces a code block that calls `llm_query()`, and then returns the
    nested call result via `FINAL_VAR(...)`.
    """

    llm = QueueLLM(
        model_name="dummy",
        responses=[
            "```repl\nresp = llm_query('ping')\n```\nFINAL_VAR('resp')",
            "pong",
        ],
    )
    logger = CollectingLogger()
    rlm = create_rlm(llm, environment="local", max_iterations=3, verbose=False, logger=logger)

    cc = rlm.completion("hello")
    assert cc.response == "pong"

    # Assert we recorded the nested llm_query call in the executed code block.
    assert len(logger.iterations) == 1
    iter0 = logger.iterations[0]
    assert len(iter0.code_blocks) == 1
    repl_result = iter0.code_blocks[0].result
    assert [c.response for c in repl_result.llm_calls] == ["pong"]
