from __future__ import annotations

import os

import pytest

from rlm._legacy.environments.local_repl import LocalREPL


@pytest.mark.unit
def test_local_repl_state_persists_across_execute_code_calls() -> None:
    with LocalREPL() as env:
        r1 = env.execute_code("x = 1")
        assert r1.locals["x"] == 1

        r2 = env.execute_code("print(x)")
        assert r2.stdout == "1\n"
        assert r2.stderr == ""
        assert r2.locals["x"] == 1


@pytest.mark.unit
def test_local_repl_captures_stdout_and_stderr() -> None:
    with LocalREPL() as env:
        r = env.execute_code('print("hi")')
        assert r.stdout == "hi\n"
        assert r.stderr == ""


@pytest.mark.unit
def test_local_repl_captures_exceptions_in_stderr() -> None:
    with LocalREPL() as env:
        r = env.execute_code("1/0")
        assert r.stdout == ""
        assert "ZeroDivisionError" in r.stderr


@pytest.mark.unit
def test_local_repl_load_context_json_handles_single_quotes_in_payload() -> None:
    payload = {"name": "O'Brien"}
    with LocalREPL(context_payload=payload) as env:
        r = env.execute_code('print(context["name"])')
        assert r.stdout == "O'Brien\n"
        assert r.stderr == ""


@pytest.mark.unit
def test_local_repl_exec_runs_in_temp_cwd() -> None:
    with LocalREPL() as env:
        r = env.execute_code("import os\nprint(os.getcwd())")
        assert os.path.realpath(r.stdout.strip()) == os.path.realpath(env.temp_dir)
        assert os.path.isdir(env.temp_dir)
