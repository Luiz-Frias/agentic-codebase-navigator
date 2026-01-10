from __future__ import annotations

import pytest

from rlm.api.factory import create_rlm
from rlm.api.registries import ensure_docker_available
from rlm.domain.errors import ExecutionError
from tests.fakes_ports import CollectingLogger, QueueLLM


@pytest.mark.e2e
@pytest.mark.docker
def test_docker_env_state_persists_across_execute_code_calls() -> None:
    """
    Integration: docker env must persist state across multiple `execute_code` calls.

    We validate this by:
    - Iteration 0: set `x = 1` in a code block (no FINAL)
    - Iteration 1: return `FINAL_VAR('x')` without redefining `x`
    """

    try:
        ensure_docker_available(timeout_s=0.5)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    llm = QueueLLM(
        model_name="dummy",
        responses=[
            "```repl\nx = 1\n```",
            "FINAL_VAR('x')",
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

    assert cc.response == "1"
    assert len(logger.iterations) >= 2
    it0 = logger.iterations[0]
    assert it0.code_blocks
    assert it0.code_blocks[0].result.locals.get("x") == "1"
