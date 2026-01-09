from __future__ import annotations

import pytest


@pytest.mark.unit
def test_localrepl_cleanup_is_idempotent() -> None:
    from rlm._legacy.environments.local_repl import LocalREPL

    env = LocalREPL()
    env.cleanup()
    env.cleanup()


@pytest.mark.unit
def test_dockerrepl_cleanup_is_idempotent_and_stops_proxy_and_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlm._legacy.environments.docker_repl as docker_mod

    calls: list[list[str]] = []

    def _fake_run(cmd, *args, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(docker_mod.subprocess, "run", _fake_run)

    class _DummyServer:
        def __init__(self) -> None:
            self.shutdown_called = False
            self.close_called = False

        def shutdown(self) -> None:
            self.shutdown_called = True

        def server_close(self) -> None:
            self.close_called = True

    class _DummyThread:
        def __init__(self) -> None:
            self.join_called = False

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            self.join_called = True

    server = _DummyServer()
    thread = _DummyThread()

    def _fake_setup(self) -> None:
        self.container_id = "cid"
        self.proxy_server = server
        self.proxy_thread = thread
        self.proxy_port = 1234

    monkeypatch.setattr(docker_mod.DockerREPL, "setup", _fake_setup)

    env = docker_mod.DockerREPL(image="python:3.12-slim", lm_handler_address=("127.0.0.1", 0))
    env.cleanup()

    # Proxy cleaned up.
    assert server.shutdown_called is True
    assert server.close_called is True
    assert thread.join_called is True

    # Container stop attempted once.
    assert any(cmd[:2] == ["docker", "stop"] for cmd in calls)

    # Idempotent.
    env.cleanup()


@pytest.mark.unit
def test_dockerrepl_cleanup_tolerates_partial_initialization() -> None:
    import rlm._legacy.environments.docker_repl as docker_mod

    env = object.__new__(docker_mod.DockerREPL)
    docker_mod.DockerREPL.cleanup(env)  # should not raise
