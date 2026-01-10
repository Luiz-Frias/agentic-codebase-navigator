from __future__ import annotations

import pytest

from rlm.domain.errors import ValidationError
from rlm.domain.models.usage import ModelUsageSummary, UsageSummary


@pytest.mark.unit
def test_model_usage_summary_rejects_negative_values() -> None:
    with pytest.raises(ValidationError, match="total_calls"):
        ModelUsageSummary(total_calls=-1)
    with pytest.raises(ValidationError, match="total_input_tokens"):
        ModelUsageSummary(total_input_tokens=-1)
    with pytest.raises(ValidationError, match="total_output_tokens"):
        ModelUsageSummary(total_output_tokens=-1)


@pytest.mark.unit
def test_model_usage_summary_from_dict_casts_and_validates() -> None:
    s = ModelUsageSummary.from_dict(
        {"total_calls": "2", "total_input_tokens": None, "total_output_tokens": 3}
    )
    assert (s.total_calls, s.total_input_tokens, s.total_output_tokens) == (2, 0, 3)

    with pytest.raises(ValidationError, match="total_calls"):
        ModelUsageSummary.from_dict({"total_calls": -1})


@pytest.mark.unit
def test_usage_summary_validates_shape_and_keys() -> None:
    with pytest.raises(ValidationError, match="must be a dict"):
        UsageSummary(model_usage_summaries="not_a_dict")  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="keys must be non-empty"):
        UsageSummary(model_usage_summaries={"": ModelUsageSummary()})

    with pytest.raises(ValidationError, match="values must be ModelUsageSummary"):
        UsageSummary(model_usage_summaries={"m": "bad"})  # type: ignore[dict-item]


@pytest.mark.unit
def test_usage_summary_from_dict_rejects_non_dict_model_usage_summaries() -> None:
    with pytest.raises(TypeError, match="model_usage_summaries"):
        UsageSummary.from_dict({"model_usage_summaries": ["not", "a", "dict"]})  # type: ignore[arg-type]


@pytest.mark.unit
def test_usage_summary_total_properties_sum_across_models() -> None:
    s = UsageSummary(
        model_usage_summaries={
            "a": ModelUsageSummary(total_calls=1, total_input_tokens=2, total_output_tokens=3),
            "b": ModelUsageSummary(total_calls=4, total_input_tokens=5, total_output_tokens=6),
        }
    )
    assert s.total_calls == 5
    assert s.total_input_tokens == 7
    assert s.total_output_tokens == 9
