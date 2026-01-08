from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rlm.domain.models import (
    ChatCompletion,
    ModelUsageSummary,
    QueryMetadata,
    ReplResult,
    UsageSummary,
)
from rlm.domain.result import Err, Ok


@pytest.mark.unit
def test_result_types_are_frozen_and_hashable() -> None:
    ok1 = Ok(123)
    ok2 = Ok(123)
    assert ok1 == ok2
    assert hash(ok1) == hash(ok2)
    with pytest.raises(FrozenInstanceError):
        ok1.value = 999  # type: ignore[misc]

    err1 = Err("nope")
    err2 = Err("nope")
    assert err1 == err2
    assert hash(err1) == hash(err2)
    with pytest.raises(FrozenInstanceError):
        err1.error = "changed"  # type: ignore[misc]


@pytest.mark.unit
def test_domain_models_are_eq_comparable_but_unhashable_by_default() -> None:
    usage1 = UsageSummary(model_usage_summaries={"m": ModelUsageSummary(1, 2, 3)})
    usage2 = UsageSummary(model_usage_summaries={"m": ModelUsageSummary(1, 2, 3)})
    assert usage1 == usage2
    with pytest.raises(TypeError):
        hash(usage1)

    cc1 = ChatCompletion(
        root_model="m",
        prompt={"role": "user", "content": "hi"},
        response="ok",
        usage_summary=usage1,
        execution_time=0.1,
    )
    cc2 = ChatCompletion(
        root_model="m",
        prompt={"role": "user", "content": "hi"},
        response="ok",
        usage_summary=usage2,
        execution_time=0.1,
    )
    assert cc1 == cc2
    with pytest.raises(TypeError):
        hash(cc1)

    repl1 = ReplResult(
        stdout="out", stderr="", locals={"x": 1}, llm_calls=[cc1], execution_time=0.2
    )
    repl2 = ReplResult(
        stdout="out", stderr="", locals={"x": 1}, llm_calls=[cc2], execution_time=0.2
    )
    assert repl1 == repl2
    with pytest.raises(TypeError):
        hash(repl1)


@pytest.mark.unit
def test_query_metadata_is_frozen_but_not_hashable() -> None:
    md1 = QueryMetadata.from_context("abc")
    md2 = QueryMetadata.from_context("abc")
    assert md1 == md2
    with pytest.raises(TypeError):
        hash(md1)
    with pytest.raises(FrozenInstanceError):
        md1.context_type = "dict"  # type: ignore[misc]
