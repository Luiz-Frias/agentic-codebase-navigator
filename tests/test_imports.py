from __future__ import annotations

import pytest


@pytest.mark.unit
def test_import_rlm() -> None:
    import rlm  # noqa: F401
