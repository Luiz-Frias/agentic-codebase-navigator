from __future__ import annotations

import pytest

from rlm._legacy.core.types import RLMChatCompletion


@pytest.mark.unit
def test_rlm_chat_completion_from_dict_allows_missing_usage_summary() -> None:
    # Malformed/older payloads may omit `usage_summary`. This should not crash
    # with AttributeError from UsageSummary.from_dict(None).
    cc = RLMChatCompletion.from_dict(
        {
            "root_model": "dummy",
            "prompt": "p",
            "response": "r",
            "execution_time": 0.1,
        }
    )

    assert cc.usage_summary.model_usage_summaries == {}


@pytest.mark.unit
def test_rlm_chat_completion_from_dict_rejects_non_dict_usage_summary() -> None:
    with pytest.raises(TypeError, match="usage_summary must be a dict"):
        RLMChatCompletion.from_dict(
            {
                "root_model": "dummy",
                "prompt": "p",
                "response": "r",
                "usage_summary": "not_a_dict",
                "execution_time": 0.1,
            }
        )
