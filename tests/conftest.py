from __future__ import annotations

import subprocess
from functools import lru_cache
from shutil import which

import pytest


@lru_cache(maxsize=1)
def _docker_available() -> bool:
    if which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return False
    return True


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unit: unit tests (fast, hermetic)")
    config.addinivalue_line("markers", "integration: integration tests (may use Docker)")
    config.addinivalue_line("markers", "docker: requires a working local Docker daemon")
    # TODO(phase4/phase5): Add a `live_llm` marker + opt-in env gate (no CI by default)
    # for integration tests that exercise real provider adapters (e.g. OpenAIAdapter)
    # against either a local OpenAI-compatible endpoint or real APIs via env vars.


@pytest.fixture(scope="session")
def docker_is_available() -> bool:
    return _docker_available()


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "docker" in item.keywords and not _docker_available():
        pytest.skip("Docker not available (no docker binary or daemon not reachable)")
