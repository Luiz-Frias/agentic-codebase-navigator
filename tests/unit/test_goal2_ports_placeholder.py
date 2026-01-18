from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.unit
def test_goal2_ports_module_imports_and_exposes_types() -> None:
    # The placeholder ports must be importable and dependency-free.
    from rlm.domain import goal2_ports as g2

    chunk: g2.CodeChunk = {
        "file_path": "x.py",
        "start_line": 1,
        "end_line": 2,
        "content": "print(1)",
    }
    index: g2.CodebaseIndex = {"root": "/repo", "files": ["x.py"]}
    assert chunk["file_path"] == "x.py"
    assert index["files"] == ["x.py"]


@pytest.mark.unit
def test_goal2_ports_not_imported_by_default_completion_paths() -> None:
    """Guard: Goal 1 runtime paths must not import Goal 2 placeholder modules.

    We run this in a subprocess to avoid pytest import ordering interactions.
    """
    code = textwrap.dedent(
        """
        from rlm.api import create_rlm
        from rlm.adapters.llm.mock import MockLLMAdapter

        rlm = create_rlm(
            MockLLMAdapter(model="m", script=["FINAL(ok)"]),
            environment="local",
            max_iterations=2,
            verbose=False,
        )
        cc = rlm.completion("hello")
        assert cc.response == "ok"

        import sys

        print("rlm.domain.goal2_ports" in sys.modules)
        """,
    ).strip()

    cp = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cp.returncode == 0, f"subprocess failed:\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
    assert cp.stdout.strip() == "False"
