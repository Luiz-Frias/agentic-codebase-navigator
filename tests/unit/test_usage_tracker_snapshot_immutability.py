from __future__ import annotations

import pytest

from rlm.adapters.llm.provider_base import UsageTracker
from rlm.domain.models import ModelUsageSummary


@pytest.mark.unit
def test_usage_tracker_get_usage_summary_is_snapshot_not_mutated_by_later_record() -> None:
    tracker = UsageTracker()

    tracker.record("m", input_tokens=1, output_tokens=2)
    snap1 = tracker.get_usage_summary()
    snap1_m = snap1.model_usage_summaries["m"]
    assert isinstance(snap1_m, ModelUsageSummary)
    assert snap1_m.total_calls == 1

    tracker.record("m", input_tokens=3, output_tokens=4)

    # Snapshot must not change after later records.
    assert snap1.model_usage_summaries["m"].total_calls == 1
    assert snap1.model_usage_summaries["m"].total_input_tokens == 1
    assert snap1.model_usage_summaries["m"].total_output_tokens == 2

    snap2 = tracker.get_usage_summary()
    assert snap2.model_usage_summaries["m"].total_calls == 2
    assert snap2.model_usage_summaries["m"].total_input_tokens == 4
    assert snap2.model_usage_summaries["m"].total_output_tokens == 6


@pytest.mark.unit
def test_usage_tracker_usage_summary_does_not_allow_mutating_internal_state() -> None:
    tracker = UsageTracker()

    tracker.record("m", input_tokens=1, output_tokens=2)
    snap = tracker.get_usage_summary()

    # If the returned snapshot shares internal objects, this will corrupt totals.
    snap.model_usage_summaries["m"].total_calls = 999

    snap2 = tracker.get_usage_summary()
    assert snap2.model_usage_summaries["m"].total_calls == 1
