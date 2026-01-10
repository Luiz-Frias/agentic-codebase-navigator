from __future__ import annotations

import pytest

from rlm.api.registries import ensure_docker_available


@pytest.mark.integration
@pytest.mark.docker
def test_dockerrepl_hanging_exec_times_out_and_cleans_up() -> None:
    """
    Integration: if docker exec hangs past subprocess timeout, DockerREPL returns a safe
    timeout error and cleans up its proxy + container.
    """

    try:
        ensure_docker_available(timeout_s=0.5)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    from rlm._legacy.environments.docker_repl import DockerREPL

    env: DockerREPL | None = None
    try:
        env = DockerREPL(image="python:3.12-slim", subprocess_timeout_s=0.5)
    except Exception as exc:
        # Common in constrained CI environments (no image pull / no daemon permissions).
        if "Failed to start container" in str(exc):
            pytest.skip(str(exc))
        raise

    try:
        res = env.execute_code("import time\ntime.sleep(10)\n")
        assert "TimeoutExpired" in res.stderr
        assert "docker exec exceeded" in res.stderr

        # Timeout path triggers cleanup.
        assert env.container_id is None
        assert env.proxy_server is None
        assert env.proxy_thread is None
    finally:
        env.cleanup()
