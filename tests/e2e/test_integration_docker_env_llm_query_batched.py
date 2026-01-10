from __future__ import annotations

import pytest

from rlm.api.factory import create_rlm
from rlm.api.registries import ensure_docker_available
from rlm.domain.errors import ExecutionError
from tests.fakes_ports import CollectingLogger, QueueLLM


@pytest.mark.e2e
@pytest.mark.docker
def test_docker_env_llm_query_batched_preserves_order() -> None:
    """
    Integration: docker env nested llm_query_batched should preserve prompt ordering.
    """

    try:
        ensure_docker_available(timeout_s=0.5)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    llm = QueueLLM(
        model_name="dummy",
        responses=[
            "```repl\njoined = '|'.join(llm_query_batched(['a','b','c']))\n```\nFINAL_VAR('joined')",
            "r1",
            "r2",
            "r3",
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

    assert cc.response == "r1|r2|r3"

    assert len(logger.iterations) == 1
    iter0 = logger.iterations[0]
    assert len(iter0.code_blocks) == 1
    repl_result = iter0.code_blocks[0].result
    assert [c.response for c in repl_result.llm_calls] == ["r1", "r2", "r3"]
