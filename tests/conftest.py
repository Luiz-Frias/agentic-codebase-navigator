from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
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


def _live_llm_enabled() -> bool:
    """
    Opt-in gate for tests that hit real provider APIs.

    These tests are skipped by default to keep CI hermetic and avoid accidental
    spend. Enable with:
      - RLM_RUN_LIVE_LLM_TESTS=1
    """

    raw = (os.environ.get("RLM_RUN_LIVE_LLM_TESTS") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unit: unit tests (fast, hermetic)")
    config.addinivalue_line("markers", "integration: integration tests (multi-component)")
    config.addinivalue_line("markers", "e2e: end-to-end tests (public API full flow)")
    config.addinivalue_line("markers", "packaging: packaging smoke tests (build/install/import)")
    config.addinivalue_line("markers", "chaos: chaos or resilience tests")
    config.addinivalue_line("markers", "performance: performance or load tests")
    config.addinivalue_line("markers", "docker: requires a working local Docker daemon")
    config.addinivalue_line(
        "markers",
        "live_llm: opt-in tests that call real provider APIs (requires RLM_RUN_LIVE_LLM_TESTS=1)",
    )


@pytest.fixture(scope="session")
def docker_is_available() -> bool:
    return _docker_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    category_markers = {
        "unit",
        "integration",
        "e2e",
        "packaging",
        "chaos",
        "performance",
    }
    tests_root = Path(__file__).resolve().parent
    errors: list[str] = []

    for item in items:
        item_path = Path(str(item.fspath))
        try:
            rel = item_path.resolve().relative_to(tests_root)
        except ValueError:
            errors.append(f"{item_path}: test path is outside tests/")
            continue

        category = rel.parts[0] if rel.parts else None
        markers = {m.name for m in item.iter_markers() if m.name in category_markers}

        if category not in category_markers:
            errors.append(f"{item_path}: tests must live under tests/<category>/")
            continue
        if len(markers) != 1:
            errors.append(
                f"{item_path}: expected exactly one category marker "
                f"{sorted(category_markers)}; got {sorted(markers)}"
            )
            continue
        marker = next(iter(markers))
        if marker != category:
            errors.append(f"{item_path}: marker '{marker}' does not match directory '{category}'")

    if errors:
        raise pytest.UsageError("Test marker/category mismatch:\n" + "\n".join(errors))


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "docker" in item.keywords and not _docker_available():
        pytest.skip("Docker not available (no docker binary or daemon not reachable)")
    if "live_llm" in item.keywords and not _live_llm_enabled():
        pytest.skip("Live LLM tests disabled (set RLM_RUN_LIVE_LLM_TESTS=1 to enable)")
