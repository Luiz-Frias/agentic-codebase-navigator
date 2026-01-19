from __future__ import annotations

import pytest

from rlm.api.factory import create_rlm
from rlm.api.registries import ensure_docker_available
from rlm.domain.errors import ExecutionError
from tests.fakes_ports import CollectingLogger, QueueLLM


@pytest.mark.e2e
@pytest.mark.docker
def test_docker_env_code_can_call_llm_query_and_result_is_returned_via_final_var() -> None:
    """
    Integration: domain orchestrator + docker environment + broker.

    This is best-effort and should skip cleanly if Docker isn't available or
    container startup/pulls are blocked in the environment running tests.
    """
    try:
        ensure_docker_available(timeout_s=0.5)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    llm = QueueLLM(
        model_name="dummy",
        responses=[
            "```repl\nresp = llm_query('ping')\n```\nFINAL_VAR('resp')",
            "pong",
        ],
    )
    logger = CollectingLogger()
    rlm = create_rlm(llm, environment="docker", max_iterations=3, verbose=False, logger=logger)

    try:
        cc = rlm.completion("hello")
    except ExecutionError as exc:
        cause = exc.__cause__
        if cause is not None and "Failed to start container" in str(cause):
            pytest.skip(str(cause))
        raise
    except RuntimeError as exc:
        if "Failed to start container" in str(exc):
            pytest.skip(str(exc))
        raise

    assert cc.response == "pong"

    assert len(logger.iterations) == 1
    iter0 = logger.iterations[0]
    assert len(iter0.code_blocks) == 1
    repl_result = iter0.code_blocks[0].result
    assert [c.response for c in repl_result.llm_calls] == ["pong"]
