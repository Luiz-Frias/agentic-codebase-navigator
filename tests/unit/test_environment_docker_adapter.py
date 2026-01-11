from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm.adapters.environments.docker import DockerEnvironmentAdapter
from rlm.domain.models import ChatCompletion
from rlm.domain.models.usage import UsageSummary


def _make_env(tmp_path: Path) -> DockerEnvironmentAdapter:
    # Avoid __init__ (would try to start real docker + proxy). We only need the methods.
    env = DockerEnvironmentAdapter.__new__(DockerEnvironmentAdapter)
    env.image = "python:3.12-slim"
    env._host_workspace = str(tmp_path)
    env._subprocess_timeout_s = 1.0
    env._proxy_http_timeout_s = 1.0
    env._cleanup_subprocess_timeout_s = 0.1
    env._thread_join_timeout_s = 0.1
    env._stop_grace_s = 0
    env._correlation_id = "cid"
    env._proxy_port = 9999
    env._calls_lock = threading.Lock()
    env._pending_calls = []
    env._container_id = "container"
    env._tmp = SimpleNamespace(cleanup=lambda: None)
    env._proxy_server = None
    env._proxy_thread = None
    return env


@pytest.mark.unit
def test_docker_environment_start_container_sets_container_id_and_raises_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _make_env(tmp_path)

    def _ok_run(cmd, **_kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, stdout="abc\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _ok_run)
    env._start_container()
    assert env._container_id == "abc"

    def _bad_run(cmd, **_kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", _bad_run)
    with pytest.raises(RuntimeError, match="Failed to start container"):
        env._start_container()


@pytest.mark.unit
def test_docker_environment_load_context_writes_files_and_calls_execute_code(
    tmp_path: Path,
) -> None:
    env = _make_env(tmp_path)

    seen: list[str] = []
    env.execute_code = lambda code: seen.append(code) or SimpleNamespace()  # type: ignore[method-assign]

    env.load_context("O'Brien")
    assert (tmp_path / "context.txt").read_text(encoding="utf-8") == "O'Brien"
    assert any("context = f.read()" in c for c in seen)

    seen.clear()
    env.load_context({"x": 1})
    data = json.loads((tmp_path / "context.json").read_text(encoding="utf-8"))
    assert data == {"x": 1}
    assert any("context = json.load(f)" in c for c in seen)


@pytest.mark.unit
def test_docker_environment_execute_code_requires_container_id(tmp_path: Path) -> None:
    env = _make_env(tmp_path)
    env._container_id = None
    with pytest.raises(RuntimeError, match="container_id is missing"):
        env.execute_code("print('x')")


@pytest.mark.unit
def test_docker_environment_execute_code_timeout_returns_repl_result_and_calls_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _make_env(tmp_path)

    cleaned: list[bool] = []
    env.cleanup = lambda: cleaned.append(True)  # type: ignore[method-assign]

    def _timeout_run(cmd, **_kwargs):  # noqa: ANN001
        # Simulate the in-flight proxy recording a nested call before the subprocess times out.
        with env._calls_lock:
            env._pending_calls.append(
                ChatCompletion(
                    root_model="m",
                    prompt="p",
                    response="r",
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=0.0,
                )
            )
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1.0, output="OUT", stderr="ERR")

    monkeypatch.setattr(subprocess, "run", _timeout_run)

    rr = env.execute_code("print('x')")
    assert rr.stdout == "OUT"
    assert "ERR" in rr.stderr
    assert "TimeoutExpired: docker exec exceeded" in rr.stderr
    assert len(rr.llm_calls) == 1
    assert cleaned == [True]


@pytest.mark.unit
def test_docker_environment_execute_code_parses_last_json_line_and_parse_error_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _make_env(tmp_path)

    ok_payload = {"stdout": "SO", "stderr": "SE1", "locals": {"x": "1"}}
    ok_out = "noise\n" + json.dumps(ok_payload)

    def _ok_run(cmd, **_kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, stdout=ok_out, stderr="SE2")

    monkeypatch.setattr(subprocess, "run", _ok_run)
    rr = env.execute_code("print('x')")
    assert rr.stdout == "SO"
    assert rr.stderr == "SE1SE2"
    assert rr.locals == {"x": "1"}

    def _bad_json(cmd, **_kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, stdout="not-json", stderr="")

    monkeypatch.setattr(subprocess, "run", _bad_json)
    rr2 = env.execute_code("print('x')")
    assert rr2.stdout == "not-json"
    assert rr2.stderr == "Parse error"


@pytest.mark.unit
def test_docker_environment_cleanup_is_idempotent_and_best_effort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _make_env(tmp_path)

    # Make cleanup exercise error-tolerant paths.
    env._container_id = "cid"
    env._proxy_server = SimpleNamespace(
        shutdown=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        server_close=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    env._proxy_thread = SimpleNamespace(is_alive=lambda: True, join=lambda **_k: None)
    env._tmp = SimpleNamespace(cleanup=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with env._calls_lock:
        env._pending_calls.extend(
            [
                ChatCompletion(
                    root_model="m",
                    prompt="p",
                    response="r",
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=0.0,
                )
            ]
        )

    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    env.cleanup()  # must not raise
    assert env._container_id is None
    assert env._proxy_server is None
    assert env._proxy_thread is None
    assert env._pending_calls == []


@pytest.mark.unit
def test_docker_environment_init_calls_cleanup_on_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleaned: list[bool] = []

    def _boom(self) -> None:  # noqa: ANN001
        raise RuntimeError("boom")

    def _cleanup(self) -> None:  # noqa: ANN001
        cleaned.append(True)

    monkeypatch.setattr(DockerEnvironmentAdapter, "_start_proxy", _boom)
    monkeypatch.setattr(DockerEnvironmentAdapter, "_start_container", lambda *_a, **_k: None)
    monkeypatch.setattr(DockerEnvironmentAdapter, "cleanup", _cleanup)

    with pytest.raises(RuntimeError, match="boom"):
        DockerEnvironmentAdapter()
    assert cleaned == [True]
