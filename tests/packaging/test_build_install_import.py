from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


def _venv_python(venv_dir: Path) -> Path:
    candidates = [
        venv_dir / "bin" / "python",
        venv_dir / "Scripts" / "python.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise RuntimeError(f"Could not find venv python under {venv_dir}")


def _run(
    cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.packaging
def test_build_wheel_and_sdist_excludes_references_and_installs_cleanly(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    # Build artifacts to an isolated output dir so the repo stays clean.
    cp = _run(["uv", "build", "--wheel", "--sdist", "--out-dir", str(dist_dir)], cwd=repo_root)
    assert cp.returncode == 0, f"uv build failed:\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"

    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    assert wheels, f"expected at least one wheel in {dist_dir}"
    assert sdists, f"expected at least one sdist in {dist_dir}"

    wheel = wheels[0]
    sdist = sdists[0]

    # Wheel should never include the upstream snapshot.
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        assert not any("references/" in n for n in names)

        metadata_files = [n for n in names if n.endswith(".dist-info/METADATA")]
        assert metadata_files, "expected METADATA in wheel"
        metadata = zf.read(metadata_files[0]).decode("utf-8", errors="replace")

        # Packaging metadata contract: extras are present in wheel metadata.
        for extra in (
            "llm-openai",
            "llm-anthropic",
            "llm-gemini",
            "llm-portkey",
            "llm-litellm",
            "llm-azure-openai",
            "env-modal",
            "env-docker",
        ):
            assert f"Provides-Extra: {extra}" in metadata

        assert "Requires-Python: >=3.12" in metadata

    # Sdist should also prune the snapshot to keep the artifact small.
    with tarfile.open(sdist, mode="r:gz") as tf:
        members = tf.getnames()
        assert not any("/references/" in m for m in members)

    # Install the wheel into a fresh venv and run a tiny no-network scenario.
    venv_dir = tmp_path / "venv"
    env = os.environ.copy()
    cp = _run([sys.executable, "-m", "venv", str(venv_dir)], cwd=repo_root, env=env)
    assert cp.returncode == 0, f"venv create failed:\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
    py = _venv_python(venv_dir)

    # Ensure pip exists (some python builds may omit it from venv by default).
    cp = _run([str(py), "-m", "pip", "--version"], cwd=repo_root)
    if cp.returncode != 0:
        cp2 = _run([str(py), "-m", "ensurepip", "--upgrade"], cwd=repo_root)
        assert cp2.returncode == 0, (
            f"ensurepip failed:\nSTDOUT:\n{cp2.stdout}\nSTDERR:\n{cp2.stderr}"
        )

    # Install the wheel with dependencies. The "no-network" aspect of this test
    # refers to using the mock LLM backend (no external API calls), not pip install.
    # Core dependencies like loguru must be installed for the package to function.
    cp = _run([str(py), "-m", "pip", "install", str(wheel)], cwd=repo_root)
    assert cp.returncode == 0, f"pip install failed:\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"

    smoke_code = r"""
import rlm
from rlm.api import EnvironmentConfig, LLMConfig, RLMConfig, create_rlm_from_config

cfg = RLMConfig(
    llm=LLMConfig(backend="mock", model_name="mock-model", backend_kwargs={"script": ["FINAL(ok)"]}),
    env=EnvironmentConfig(environment="local"),
    max_iterations=2,
    verbose=False,
)
cc = create_rlm_from_config(cfg).completion("hello")
assert cc.response == "ok"
"""
    cp = _run([str(py), "-c", smoke_code], cwd=repo_root)
    assert cp.returncode == 0, (
        f"smoke import/run failed:\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
    )

    # CLI smoke: module runner.
    cp = _run([str(py), "-m", "rlm", "--version"], cwd=repo_root)
    assert cp.returncode == 0, f"cli --version failed:\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
    cp = _run(
        [
            str(py),
            "-m",
            "rlm",
            "completion",
            "hello",
            "--backend",
            "mock",
            "--final",
            "ok",
        ],
        cwd=repo_root,
    )
    assert cp.returncode == 0, f"cli completion failed:\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
    assert cp.stdout.strip() == "ok"
