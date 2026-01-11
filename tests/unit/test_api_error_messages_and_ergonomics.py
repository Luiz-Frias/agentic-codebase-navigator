from __future__ import annotations

import pytest

from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.api import create_rlm
from rlm.domain.errors import ValidationError


@pytest.mark.unit
def test_facade_rejects_duplicate_other_llm_model_names() -> None:
    rlm = create_rlm(
        MockLLMAdapter(model="dup", script=["FINAL(ok)"]),
        other_llms=[MockLLMAdapter(model="dup", script=["FINAL(nope)"])],
        environment="local",
        max_iterations=2,
        verbose=False,
    )
    with pytest.raises(ValidationError, match="Duplicate model registered"):
        rlm.completion("hello")
