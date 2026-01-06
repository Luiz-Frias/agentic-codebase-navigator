from __future__ import annotations

import ast
from pathlib import Path

import pytest


@pytest.mark.unit
def test_src_code_does_not_import_references_snapshot() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"

    offenders: list[str] = []
    for py_file in src_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            match node:
                case ast.Import(names=names):
                    for alias in names:
                        if alias.name.startswith("references"):
                            offenders.append(f"{py_file}: import {alias.name}")
                case ast.ImportFrom(module=module) if module is not None:
                    if module.startswith("references"):
                        offenders.append(f"{py_file}: from {module} import ...")

    assert not offenders, "Found imports from references snapshot:\n" + "\n".join(offenders)
