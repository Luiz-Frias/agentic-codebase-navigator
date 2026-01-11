from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from rlm.domain.models import (
    ChatCompletion,
    CodeBlock,
    Iteration,
    ModelUsageSummary,
    ReplResult,
    UsageSummary,
)
from rlm.domain.result import Err, Ok, Result


@pytest.mark.unit
def test_domain_models_roundtrip_via_to_from_dict_is_stable() -> None:
    usage = UsageSummary(model_usage_summaries={"m": ModelUsageSummary(1, 2, 3)})
    cc = ChatCompletion(
        root_model="m",
        prompt={"role": "user", "content": "hi"},
        response="ok",
        usage_summary=usage,
        execution_time=0.123,
    )
    repl = ReplResult(
        stdout="out",
        stderr="err",
        locals={
            "a": 1,
            "b": [1, 2, 3],
            "c": {"x": "y"},
            "callable": math.sin,
            "module_like": json,  # module should serialize deterministically
            "unknown": SimpleNamespace(x=1),
        },
        llm_calls=[cc],
        execution_time=0.456,
    )
    it = Iteration(
        prompt="p",
        response="r",
        code_blocks=[CodeBlock(code="print(1)", result=repl)],
        final_answer="fa",
        iteration_time=0.789,
    )

    # Contract: dict → from_dict(dict) → to_dict() is stable for JSON logging.
    d = it.to_dict()
    it2 = Iteration.from_dict(d)
    assert it2.to_dict() == d


@pytest.mark.unit
def test_domain_model_usage_summary_from_dict_treats_none_as_zero() -> None:
    s = ModelUsageSummary.from_dict(
        {"total_calls": None, "total_input_tokens": None, "total_output_tokens": None}
    )
    assert (s.total_calls, s.total_input_tokens, s.total_output_tokens) == (0, 0, 0)


@pytest.mark.unit
def test_domain_repl_result_from_dict_accepts_legacy_rlm_calls_key() -> None:
    legacy_like = {
        "stdout": "out",
        "stderr": "",
        "locals": {},
        "execution_time": 0.2,
        # Legacy REPLResult.to_dict() uses "rlm_calls" for schema compatibility.
        "rlm_calls": [
            {
                "root_model": "dummy",
                "prompt": "p",
                "response": "FINAL(pong)",
                "usage_summary": {"model_usage_summaries": {"dummy": {"total_calls": 1}}},
                "execution_time": 0.01,
            }
        ],
    }

    repl = ReplResult.from_dict(legacy_like)
    assert [c.response for c in repl.llm_calls] == ["FINAL(pong)"]


@pytest.mark.unit
def test_domain_result_type_pattern_matching() -> None:
    def _f(flag: bool) -> Result[int]:
        if flag:
            return Ok(123)
        return Err("nope")

    match _f(True):
        case Ok(value=v):
            assert v == 123
        case _:
            raise AssertionError("expected Ok")
