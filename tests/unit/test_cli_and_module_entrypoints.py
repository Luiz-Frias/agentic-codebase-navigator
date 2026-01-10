from __future__ import annotations

import json
import runpy
import sys

import pytest

from rlm import __version__
from rlm.cli import main


@pytest.mark.unit
def test_cli_version_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--version"])
    assert code == 0
    out = capsys.readouterr().out.strip()
    assert out == __version__


@pytest.mark.unit
def test_cli_completion_mock_prints_final_answer(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "completion",
            "hello",
            "--backend",
            "mock",
            "--model-name",
            "test-model",
            "--final",
            "ok",
            "--environment",
            "local",
            "--max-iterations",
            "2",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out.strip()
    assert out == "ok"


@pytest.mark.unit
def test_cli_completion_mock_json_outputs_chat_completion_dict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "completion",
            "hello",
            "--backend",
            "mock",
            "--model-name",
            "test-model",
            "--final",
            "ok",
            "--environment",
            "local",
            "--max-iterations",
            "2",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["root_model"] == "test-model"
    assert payload["response"] == "ok"


@pytest.mark.unit
def test_cli_completion_mock_jsonl_log_dir_writes_log(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    code = main(
        [
            "completion",
            "hello",
            "--backend",
            "mock",
            "--model-name",
            "test-model",
            "--final",
            "ok",
            "--environment",
            "local",
            "--max-iterations",
            "2",
            "--jsonl-log-dir",
            str(log_dir),
        ]
    )
    assert code == 0
    assert capsys.readouterr().out.strip() == "ok"

    files = sorted(log_dir.glob("rlm_*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2


@pytest.mark.unit
def test_cli_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([])
    assert code == 0
    out = capsys.readouterr().out
    assert "usage:" in out.lower()


@pytest.mark.unit
def test_python_m_rlm_executes_main_module_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["rlm", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("rlm", run_name="__main__")
    assert exc.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out == __version__
