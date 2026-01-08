from __future__ import annotations

import pytest

from rlm._legacy.core.types import ModelUsageSummary as LegacyModelUsageSummary
from rlm._legacy.core.types import RLMChatCompletion as LegacyChatCompletion
from rlm._legacy.core.types import UsageSummary as LegacyUsageSummary
from rlm.adapters.legacy.mappers import (
    legacy_chat_completion_to_domain,
    legacy_model_usage_summary_to_domain,
    legacy_usage_summary_to_domain,
)


@pytest.mark.unit
def test_legacy_model_usage_summary_to_domain() -> None:
    legacy = LegacyModelUsageSummary(total_calls=1, total_input_tokens=2, total_output_tokens=3)
    dom = legacy_model_usage_summary_to_domain(legacy)
    assert (dom.total_calls, dom.total_input_tokens, dom.total_output_tokens) == (1, 2, 3)


@pytest.mark.unit
def test_legacy_usage_summary_to_domain_preserves_per_model_keys() -> None:
    legacy = LegacyUsageSummary(
        model_usage_summaries={
            "a": LegacyModelUsageSummary(
                total_calls=1, total_input_tokens=10, total_output_tokens=20
            ),
            "b": LegacyModelUsageSummary(
                total_calls=2, total_input_tokens=11, total_output_tokens=21
            ),
        }
    )
    dom = legacy_usage_summary_to_domain(legacy)
    assert set(dom.model_usage_summaries.keys()) == {"a", "b"}
    assert dom.model_usage_summaries["a"].total_calls == 1
    assert dom.model_usage_summaries["b"].total_output_tokens == 21


@pytest.mark.unit
def test_legacy_chat_completion_to_domain_preserves_core_fields() -> None:
    legacy_usage = LegacyUsageSummary(
        model_usage_summaries={
            "m": LegacyModelUsageSummary(total_calls=1, total_input_tokens=0, total_output_tokens=0)
        }
    )
    legacy = LegacyChatCompletion(
        root_model="m",
        prompt=[{"role": "user", "content": "hi"}],
        response="answer",
        usage_summary=legacy_usage,
        execution_time=0.123,
    )
    dom = legacy_chat_completion_to_domain(legacy)
    assert dom.root_model == "m"
    assert dom.prompt == [{"role": "user", "content": "hi"}]
    assert dom.response == "answer"
    assert dom.execution_time == 0.123
    assert dom.usage_summary.model_usage_summaries["m"].total_calls == 1
