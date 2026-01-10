from __future__ import annotations

import math

import pytest

from rlm._legacy.core.types import (
    ModelUsageSummary,
    QueryMetadata,
    REPLResult,
    RLMChatCompletion,
    UsageSummary,
)


@pytest.mark.unit
def test_replresult_llm_calls_is_canonical_and_rlm_calls_is_alias() -> None:
    usage = UsageSummary(model_usage_summaries={})
    call = RLMChatCompletion(
        root_model="dummy",
        prompt="p",
        response="r",
        usage_summary=usage,
        execution_time=0.01,
    )

    r = REPLResult(stdout="ok", stderr="", locals={}, llm_calls=[call])
    assert r.llm_calls == [call]
    assert r.rlm_calls == [call]

    # Setting compat alias should update canonical field
    r.rlm_calls = []
    assert r.llm_calls == []


@pytest.mark.unit
def test_replresult_rejects_both_llm_calls_and_rlm_calls() -> None:
    usage = UsageSummary(model_usage_summaries={})
    call = RLMChatCompletion(
        root_model="dummy",
        prompt="p",
        response="r",
        usage_summary=usage,
        execution_time=0.01,
    )

    with pytest.raises(ValueError, match="Pass only one of `llm_calls` or `rlm_calls`"):
        REPLResult(
            stdout="",
            stderr="",
            locals={},
            llm_calls=[call],
            rlm_calls=[call],
        )


@pytest.mark.unit
def test_replresult_to_dict_uses_upstream_key_rlm_calls() -> None:
    usage = UsageSummary(model_usage_summaries={})
    call = RLMChatCompletion(
        root_model="dummy",
        prompt="p",
        response="r",
        usage_summary=usage,
        execution_time=0.01,
    )

    r = REPLResult(stdout="", stderr="", locals={}, llm_calls=[call], execution_time=0.5)
    d = r.to_dict()

    assert "llm_calls" not in d
    assert d["rlm_calls"] == [call.to_dict()]


@pytest.mark.unit
def test_replresult_locals_are_serialized_safely() -> None:
    usage = UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(1, 2, 3)})
    call = RLMChatCompletion(
        root_model="dummy",
        prompt="p",
        response="r",
        usage_summary=usage,
        execution_time=0.01,
    )

    def my_fn() -> None:  # noqa: ANN001
        return None

    r = REPLResult(
        stdout="",
        stderr="",
        locals={"mod": math, "fn": my_fn, "n": 1},
        llm_calls=[call],
    )
    d = r.to_dict()
    assert d["locals"]["mod"] == "<module 'math'>"
    assert isinstance(d["locals"]["fn"], str)
    assert d["locals"]["n"] == 1


@pytest.mark.unit
def test_query_metadata_string_and_message_list() -> None:
    md = QueryMetadata("hello")
    assert md.context_type == "str"
    assert md.context_total_length == len("hello")

    md2 = QueryMetadata([{"role": "user", "content": "abc"}, {"role": "assistant", "content": "d"}])
    assert md2.context_type == "list"
    assert md2.context_lengths == [3, 1]
    assert md2.context_total_length == 4
