from __future__ import annotations

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
