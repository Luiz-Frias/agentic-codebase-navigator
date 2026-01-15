from __future__ import annotations

import pytest

from rlm.domain.models import ModelUsageSummary, UsageSummary
from rlm.domain.models.usage import merge_usage_summaries


@pytest.mark.unit
def test_merge_usage_summaries_sums_overlapping_models() -> None:
    a1 = UsageSummary(model_usage_summaries={"a": ModelUsageSummary(1, 2, 3)})
    a2 = UsageSummary(model_usage_summaries={"a": ModelUsageSummary(4, 5, 6)})

    merged = merge_usage_summaries([a1, a2])
    mus = merged.model_usage_summaries["a"]
    assert mus.total_calls == 5
    assert mus.total_input_tokens == 7
    assert mus.total_output_tokens == 9


@pytest.mark.unit
def test_merge_usage_summaries_merges_distinct_models_and_sorts_keys() -> None:
    # Intentionally insert keys out of order.
    s1 = UsageSummary(
        model_usage_summaries={
            "b": ModelUsageSummary(1, 0, 0),
            "a": ModelUsageSummary(2, 0, 0),
        }
    )
    s2 = UsageSummary(model_usage_summaries={"c": ModelUsageSummary(3, 0, 0)})

    merged = merge_usage_summaries([s1, s2])
    assert list(merged.model_usage_summaries.keys()) == ["a", "b", "c"]
    assert merged.model_usage_summaries["a"].total_calls == 2
    assert merged.model_usage_summaries["b"].total_calls == 1
    assert merged.model_usage_summaries["c"].total_calls == 3


@pytest.mark.unit
def test_merge_usage_summaries_returns_new_objects_without_aliasing_inputs() -> None:
    shared = ModelUsageSummary(1, 2, 3)
    s = UsageSummary(model_usage_summaries={"a": shared})

    merged = merge_usage_summaries([s])

    # Mutate the input summary after merge; output should not change.
    shared.total_calls = 999
    shared.total_input_tokens = 999
    shared.total_output_tokens = 999

    out = merged.model_usage_summaries["a"]
    assert out.total_calls == 1
    assert out.total_input_tokens == 2
    assert out.total_output_tokens == 3


@pytest.mark.unit
def test_merge_usage_summaries_empty_iterable_returns_empty_summary() -> None:
    merged = merge_usage_summaries([])
    assert merged.model_usage_summaries == {}


@pytest.mark.unit
def test_merge_usage_summaries_constructs_at_most_one_total_per_distinct_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Performance guardrail (algorithmic):

    `merge_usage_summaries()` should be linear in the number of distinct model keys
    (not in the number of input summaries).

    We assert this by counting how many `ModelUsageSummary` objects are constructed
    *inside* the merge function. A quadratic implementation (e.g. repeated pairwise
    merges with deep copies) would typically construct far more objects.
    """
    import rlm.domain.models.usage as usage_mod

    original = usage_mod.ModelUsageSummary
    constructed = 0

    class CountingModelUsageSummary(original):  # type: ignore[misc]
        def __post_init__(self) -> None:
            nonlocal constructed
            constructed += 1
            super().__post_init__()

    monkeypatch.setattr(usage_mod, "ModelUsageSummary", CountingModelUsageSummary)

    # Build inputs after patching, so UsageSummary's runtime isinstance checks
    # see the patched class.
    summaries = [
        usage_mod.UsageSummary(model_usage_summaries={"m": CountingModelUsageSummary(1, 0, 0)})
        for _ in range(200)
    ]

    constructed = 0  # count only constructions performed by merge
    merged = usage_mod.merge_usage_summaries(summaries)

    assert merged.model_usage_summaries["m"].total_calls == 200
    assert constructed == 1


@pytest.mark.unit
def test_merge_usage_summaries_construction_count_scales_with_distinct_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Performance guardrail (algorithmic):

    When all models are distinct, the merge should construct one total per model,
    not O(n^2) totals via repeated deep-copy merges.
    """
    import rlm.domain.models.usage as usage_mod

    original = usage_mod.ModelUsageSummary
    constructed = 0

    class CountingModelUsageSummary(original):  # type: ignore[misc]
        def __post_init__(self) -> None:
            nonlocal constructed
            constructed += 1
            super().__post_init__()

    monkeypatch.setattr(usage_mod, "ModelUsageSummary", CountingModelUsageSummary)

    n = 200
    summaries = [
        usage_mod.UsageSummary(model_usage_summaries={f"m{i}": CountingModelUsageSummary(1, 0, 0)})
        for i in range(n)
    ]

    constructed = 0  # count only constructions performed by merge
    merged = usage_mod.merge_usage_summaries(summaries)

    assert len(merged.model_usage_summaries) == n
    assert constructed == n
