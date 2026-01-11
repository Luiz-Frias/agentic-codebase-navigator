from __future__ import annotations

from pathlib import Path

import pytest

from rlm.infrastructure.execution_namespace_policy import ExecutionNamespacePolicy
from tests.fakes_ports import LocalEnvironmentAdapter


@pytest.mark.unit
def test_execution_namespace_policy_import_and_open_validations(tmp_path: Path) -> None:
    policy = ExecutionNamespacePolicy()
    builtins_dict = policy.build_builtins(session_dir=tmp_path)

    controlled_import = builtins_dict["__import__"]
    with pytest.raises(ImportError, match="Relative imports are not allowed"):
        controlled_import("math", level=1)

    with pytest.raises(ImportError, match="Invalid import name"):
        controlled_import(123)  # type: ignore[arg-type]

    restricted_open = builtins_dict["open"]
    with pytest.raises(PermissionError, match="file descriptor"):
        restricted_open(0)

    with pytest.raises(PermissionError, match="path-like"):
        restricted_open(object())


@pytest.mark.unit
def test_namespace_policy_allows_whitelisted_imports() -> None:
    env = LocalEnvironmentAdapter()
    try:
        r = env.execute_code("import math\nprint(math.sqrt(9))")
        assert r.stdout.strip() == "3.0"
        assert r.stderr == ""
    finally:
        env.cleanup()


@pytest.mark.unit
def test_namespace_policy_blocks_non_whitelisted_imports() -> None:
    env = LocalEnvironmentAdapter()
    try:
        r = env.execute_code("import os\nprint('hi')")
        assert r.stdout == ""
        assert "ImportError" in r.stderr
    finally:
        env.cleanup()


@pytest.mark.unit
def test_namespace_policy_restricts_open_to_session_dir(tmp_path: Path) -> None:
    env = LocalEnvironmentAdapter()
    try:
        outside = tmp_path / "outside.txt"
        r = env.execute_code(f"with open({str(outside)!r}, 'w') as f:\n    f.write('nope')")
        assert r.stdout == ""
        assert "PermissionError" in r.stderr
    finally:
        env.cleanup()


@pytest.mark.unit
def test_namespace_policy_allows_open_within_session_dir() -> None:
    env = LocalEnvironmentAdapter()
    try:
        r = env.execute_code(
            "with open('ok.txt', 'w') as f:\n"
            "    f.write('hi')\n"
            "with open('ok.txt', 'r') as f:\n"
            "    print(f.read())"
        )
        assert r.stdout == "hi\n"
        assert r.stderr == ""
    finally:
        env.cleanup()
