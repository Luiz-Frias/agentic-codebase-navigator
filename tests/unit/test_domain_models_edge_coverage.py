from __future__ import annotations

import pytest

from rlm.domain.errors import ValidationError
from rlm.domain.models import ChatCompletion, Iteration, ReplResult, RunMetadata
from rlm.domain.models.serialization import serialize_value
from rlm.domain.models.usage import ModelUsageSummary, UsageSummary


@pytest.mark.unit
def test_chat_completion_from_dict_back_compat_flat_token_counts() -> None:
    cc = ChatCompletion.from_dict(
        {
            "root_model": "m",
            "prompt": "p",
            "response": "r",
            "prompt_tokens": 7,
            "completion_tokens": 9,
            "execution_time": 0.1,
        },
    )
    mus = cc.usage_summary.model_usage_summaries["m"]
    assert mus.total_calls == 1
    assert mus.total_input_tokens == 7
    assert mus.total_output_tokens == 9


@pytest.mark.unit
def test_iteration_to_dict_includes_correlation_id_and_usage_summaries() -> None:
    iter_usage = UsageSummary(
        model_usage_summaries={
            "m": ModelUsageSummary(total_calls=1, total_input_tokens=2, total_output_tokens=3),
        },
    )
    cum_usage = UsageSummary(
        model_usage_summaries={
            "m": ModelUsageSummary(total_calls=2, total_input_tokens=4, total_output_tokens=6),
        },
    )
    it = Iteration(
        prompt="p",
        response="r",
        correlation_id="cid",
        iteration_usage_summary=iter_usage,
        cumulative_usage_summary=cum_usage,
    )
    d = it.to_dict()
    assert d["correlation_id"] == "cid"
    assert d["iteration_usage_summary"]["model_usage_summaries"]["m"]["total_calls"] == 1
    assert d["cumulative_usage_summary"]["model_usage_summaries"]["m"]["total_calls"] == 2


@pytest.mark.unit
def test_run_metadata_from_dict_parses_other_backends_and_drops_invalid() -> None:
    md = RunMetadata.from_dict(
        {"root_model": "m", "other_backends": ["a", 1], "correlation_id": "c"},
    )
    assert md.other_backends == ["a", "1"]
    assert md.correlation_id == "c"

    md0 = RunMetadata.from_dict({"root_model": "m"})
    assert md0.other_backends is None

    md2 = RunMetadata.from_dict({"root_model": "m", "other_backends": ("x", "y")})
    assert md2.other_backends == ["x", "y"]

    md3 = RunMetadata.from_dict({"root_model": "m", "other_backends": {"nope": 1}})
    assert md3.other_backends is None


@pytest.mark.unit
def test_repl_result_to_dict_includes_correlation_id_and_from_dict_accepts_llm_calls_key() -> None:
    rr = ReplResult(
        correlation_id="cid",
        stdout="out",
        stderr="err",
        llm_calls=[
            ChatCompletion(
                root_model="m",
                prompt="p",
                response="r",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=0.0,
            ),
        ],
        execution_time=0.1,
    )
    payload = rr.to_dict()
    assert payload["correlation_id"] == "cid"

    rr2 = ReplResult.from_dict(
        {
            "llm_calls": [
                {
                    "root_model": "m",
                    "prompt": "p",
                    "response": "r",
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "execution_time": 0.0,
                },
            ],
        },
    )
    assert len(rr2.llm_calls) == 1
    assert rr2.llm_calls[0].root_model == "m"


@pytest.mark.unit
def test_serialize_value_falls_back_when_repr_raises() -> None:
    class BadRepr:
        def __repr__(self) -> str:
            raise RuntimeError("nope")

    assert serialize_value(BadRepr()) == "<BadRepr>"


@pytest.mark.unit
def test_model_usage_summary_invalid_int_like_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match=r"Invalid total_calls: expected int-like value"):
        ModelUsageSummary(total_calls="oops")  # type: ignore[arg-type]
