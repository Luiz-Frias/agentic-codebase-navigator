from __future__ import annotations

import pytest

from rlm.api.factory import create_rlm
from rlm.api.registries import ensure_docker_available
from rlm.domain.errors import ExecutionError
from tests.fakes_ports import CollectingLogger, QueueLLM


@pytest.mark.integration
def test_local_env_load_context_sets_context_variable_for_str_prompt() -> None:
    llm = QueueLLM(
        responses=[
            "```repl\nprint(context)\n```\nFINAL(ok)",
        ]
    )
    logger = CollectingLogger()
    rlm = create_rlm(llm, environment="local", max_iterations=1, verbose=False, logger=logger)

    cc = rlm.completion("hello")
    assert cc.response == "ok"

    assert len(logger.iterations) == 1
    repl_result = logger.iterations[0].code_blocks[0].result
    assert repl_result.stdout.strip() == "hello"


@pytest.mark.integration
def test_local_env_load_context_sets_context_variable_for_dict_prompt() -> None:
    llm = QueueLLM(
        responses=[
            "```repl\nprint(context['x'])\n```\nFINAL(ok)",
        ]
    )
    logger = CollectingLogger()
    rlm = create_rlm(llm, environment="local", max_iterations=1, verbose=False, logger=logger)

    cc = rlm.completion({"x": "hello"})  # type: ignore[arg-type]
    assert cc.response == "ok"

    repl_result = logger.iterations[0].code_blocks[0].result
    assert repl_result.stdout.strip() == "hello"


@pytest.mark.integration
@pytest.mark.docker
def test_docker_env_load_context_sets_context_variable_for_str_prompt() -> None:
    try:
        ensure_docker_available(timeout_s=0.5)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    llm = QueueLLM(responses=["```repl\nprint(context)\n```\nFINAL(ok)"])
    logger = CollectingLogger()
    rlm = create_rlm(llm, environment="docker", max_iterations=1, verbose=False, logger=logger)

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

    assert cc.response == "ok"
    repl_result = logger.iterations[0].code_blocks[0].result
    assert repl_result.stdout.strip() == "hello"
