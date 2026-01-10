from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import pytest


@pytest.mark.unit
def test_meta_version_matches_dist_if_installed() -> None:
    import rlm
    from rlm._meta import DIST_NAME

    try:
        dist_version = version(DIST_NAME)
    except PackageNotFoundError:
        assert rlm.__version__ == "0.0.0"
    else:
        assert rlm.__version__ == dist_version


@pytest.mark.unit
def test_pytest_markers_registered(pytestconfig: pytest.Config) -> None:
    markers = pytestconfig.getini("markers")
    assert any(m.startswith("unit:") for m in markers)
    assert any(m.startswith("integration:") for m in markers)
