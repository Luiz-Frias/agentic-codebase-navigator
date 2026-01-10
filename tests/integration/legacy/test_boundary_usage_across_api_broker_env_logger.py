from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.adapters.logger.jsonl import JsonlLoggerAdapter
from rlm.api import create_rlm
from rlm.domain.models.usage import UsageSummary


@pytest.mark.integration
def test_usage_is_correct_across_api_broker_env_and_logger(tmp_path: Path) -> None:
    """
    Boundary (Phase 4 usage):
    - Public API (`create_rlm`) drives the app use case
    - TCP broker routes subcalls by model name
    - Local environment executes code and calls `llm_query()`
    - JSONL logger persists usage fields
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLoggerAdapter(log_dir=log_dir, file_name="usage_boundary")

    root_script = "```repl\nresp = llm_query('ping', model='sub')\n```\nFINAL_VAR('resp')"
    rlm = create_rlm(
        MockLLMAdapter(model="root", script=[root_script]),
        other_llms=[MockLLMAdapter(model="sub", script=["pong"])],
        environment="local",
        max_iterations=2,
        verbose=False,
        logger=logger,
    )

    cc = rlm.completion("hello")
    assert cc.root_model == "root"
    assert cc.response == "pong"

    # Final merged usage must include both the root call and the nested subcall.
    assert cc.usage_summary.model_usage_summaries["root"].total_calls == 1
    assert cc.usage_summary.model_usage_summaries["sub"].total_calls == 1

    # Verify the JSONL output preserves usage subtrees end-to-end.
    path = logger.log_file_path
    assert path is not None
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) >= 2  # metadata + at least one iteration

    meta = json.loads(lines[0])
    assert meta["type"] == "metadata"
    assert meta["root_model"] == "root"
    assert "correlation_id" in meta

    entry = json.loads(lines[1])
    assert entry["type"] == "iteration"
    assert entry["iteration"] == 1
    assert entry["correlation_id"] == meta["correlation_id"]
    assert "iteration_usage_summary" in entry
    assert "cumulative_usage_summary" in entry

    iter_usage = UsageSummary.from_dict(entry["iteration_usage_summary"])
    cum_usage = UsageSummary.from_dict(entry["cumulative_usage_summary"])
    assert iter_usage.model_usage_summaries["root"].total_calls == 1
    assert iter_usage.model_usage_summaries["sub"].total_calls == 1
    assert cum_usage.model_usage_summaries["root"].total_calls == 1
    assert cum_usage.model_usage_summaries["sub"].total_calls == 1

    # Nested environment llm_calls should include the subcall usage.
    assert entry["code_blocks"]
    result = entry["code_blocks"][0]["result"]
    assert result["rlm_calls"]
    nested = result["rlm_calls"][0]
    assert nested["root_model"] == "sub"
    nested_usage = UsageSummary.from_dict(nested["usage_summary"])
    assert nested_usage.model_usage_summaries["sub"].total_calls == 1
