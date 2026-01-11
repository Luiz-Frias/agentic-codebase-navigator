from __future__ import annotations

import pytest

from rlm.domain.models.query_metadata import QueryMetadata
from rlm.domain.services.prompts import build_rlm_system_prompt


@pytest.mark.unit
def test_build_rlm_system_prompt_truncates_more_than_100_chunks() -> None:
    md = QueryMetadata.from_context(["x"] * 101)
    messages = build_rlm_system_prompt("sys", md)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "assistant"
    assert "... [1 others]" in messages[1]["content"]
