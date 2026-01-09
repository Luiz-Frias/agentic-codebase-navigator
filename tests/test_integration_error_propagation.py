from __future__ import annotations

import pytest

from rlm.api.factory import create_rlm
from rlm.api.registries import ensure_docker_available
from rlm.domain.errors import ExecutionError
from tests.fakes_ports import CollectingLogger, QueueLLM


@pytest.mark.integration
def test_local_env_llm_query_errors_propagate_as_error_string_without_hanging() -> None:
    llm = QueueLLM(
        model_name="dummy",
        responses=[
            "```repl\nresp = llm_query('ping')\n```\nFINAL_VAR('resp')",
            RuntimeError("boom"),
        ],
    )
    logger = CollectingLogger()
    rlm = create_rlm(llm, environment="local", max_iterations=3, verbose=False, logger=logger)

    cc = rlm.completion("hello")
    assert "boom" in cc.response

    # The nested call should not be recorded as a successful llm_call.
    iter0 = logger.iterations[0]
    assert iter0.code_blocks[0].result.llm_calls == []


@pytest.mark.integration
@pytest.mark.docker
def test_docker_env_llm_query_errors_propagate_as_error_string_without_hanging() -> None:
    try:
        ensure_docker_available(timeout_s=0.5)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    llm = QueueLLM(
        model_name="dummy",
        responses=[
            "```repl\nresp = llm_query('ping')\n```\nFINAL_VAR('resp')",
            RuntimeError("boom"),
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

    assert cc.response == "Error: boom"
