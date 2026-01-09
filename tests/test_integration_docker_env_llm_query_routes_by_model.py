from __future__ import annotations

import pytest

from rlm.api.factory import create_rlm
from rlm.api.registries import ensure_docker_available
from rlm.domain.errors import ExecutionError
from tests.fakes_ports import CollectingLogger, QueueLLM


@pytest.mark.integration
@pytest.mark.docker
def test_docker_env_llm_query_can_route_to_other_backend_by_model_name() -> None:
    """
    Integration: orchestrator + docker env + broker, routing subcalls by `model=...`.

    Best-effort; should skip cleanly if Docker isn't available.
    """
    try:
        ensure_docker_available(timeout_s=0.5)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    root_script = "```repl\nresp = llm_query('ping', model='sub')\n```\nFINAL_VAR('resp')"
    root_llm = QueueLLM(model_name="root", responses=[root_script])
    sub_llm = QueueLLM(model_name="sub", responses=["pong"])

    logger = CollectingLogger()
    rlm = create_rlm(
        root_llm,
        other_llms=[sub_llm],
        environment="docker",
        max_iterations=3,
        verbose=False,
        logger=logger,
    )

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
