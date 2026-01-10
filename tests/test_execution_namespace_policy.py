from __future__ import annotations

from pathlib import Path

import pytest

from rlm.adapters.environments.local import LocalEnvironmentAdapter


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
