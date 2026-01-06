from __future__ import annotations

import pytest

from rlm._legacy.core.types import ModelUsageSummary


@pytest.mark.unit
def test_model_usage_summary_from_dict_defaults_missing_keys_to_zero() -> None:
    s = ModelUsageSummary.from_dict({})
    assert s.total_calls == 0
    assert s.total_input_tokens == 0
    assert s.total_output_tokens == 0


@pytest.mark.unit
def test_model_usage_summary_from_dict_treats_none_as_zero() -> None:
    s = ModelUsageSummary.from_dict(
        {"total_calls": None, "total_input_tokens": None, "total_output_tokens": None}
    )
    assert s.total_calls == 0
    assert s.total_input_tokens == 0
    assert s.total_output_tokens == 0


@pytest.mark.unit
def test_model_usage_summary_from_dict_casts_string_ints() -> None:
    s = ModelUsageSummary.from_dict(
        {"total_calls": "1", "total_input_tokens": "2", "total_output_tokens": "3"}
    )
    assert (s.total_calls, s.total_input_tokens, s.total_output_tokens) == (1, 2, 3)
