from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlm._legacy.logger.rlm_logger import RLMLogger
from rlm.adapters.legacy.logger import LegacyLoggerAdapter
from rlm.domain.models import (
    ChatCompletion,
    CodeBlock,
    Iteration,
    ModelUsageSummary,
    ReplResult,
    RunMetadata,
    UsageSummary,
)


@pytest.mark.integration
def test_legacy_jsonl_logger_roundtrips_usage_fields(tmp_path: Path) -> None:
    """
    Integration: legacy JSONL logger writes usage fields and they parse back into domain models.

    This is our Phase-4 contract: usage summary must be visible in logs without
    leaking provider internals and without relying on in-memory objects.
    """

    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    legacy = RLMLogger(log_dir=str(log_dir), file_name="usage_roundtrip")
    logger = LegacyLoggerAdapter(legacy)

    # metadata event
    logger.log_metadata(
        RunMetadata(
            root_model="root",
            max_depth=1,
            max_iterations=2,
            backend="mock",
            backend_kwargs={},
            environment_type="local",
            environment_kwargs={},
            other_backends=None,
            correlation_id="cid-123",
        )
    )

    # iteration event with nested llm_calls and both per-iteration + cumulative usage
    subcall_cc = ChatCompletion(
        root_model="sub",
        prompt="ping",
        response="pong",
        usage_summary=UsageSummary(
            model_usage_summaries={
                "sub": ModelUsageSummary(total_calls=1, total_input_tokens=2, total_output_tokens=3)
            }
        ),
        execution_time=0.01,
    )
    repl = ReplResult(
        stdout="out", stderr="", locals={}, llm_calls=[subcall_cc], execution_time=0.02
    )
    it = Iteration(
        prompt=[{"role": "user", "content": "hello"}],
        response="```repl\nresp = llm_query('ping', model='sub')\n```",
        code_blocks=[CodeBlock(code="resp = llm_query('ping', model='sub')", result=repl)],
        final_answer=None,
        iteration_time=0.1,
        iteration_usage_summary=UsageSummary(
            model_usage_summaries={
                "root": ModelUsageSummary(
                    total_calls=1, total_input_tokens=0, total_output_tokens=0
                ),
                "sub": ModelUsageSummary(
                    total_calls=1, total_input_tokens=2, total_output_tokens=3
                ),
            }
        ),
        cumulative_usage_summary=UsageSummary(
            model_usage_summaries={
                "root": ModelUsageSummary(
                    total_calls=1, total_input_tokens=0, total_output_tokens=0
                ),
                "sub": ModelUsageSummary(
                    total_calls=1, total_input_tokens=2, total_output_tokens=3
                ),
            }
        ),
        correlation_id="cid-123",
    )
    logger.log_iteration(it)

    # Read JSONL back.
    path = legacy.log_file_path
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 2

    meta = json.loads(lines[0])
    assert meta["type"] == "metadata"
    assert meta["root_model"] == "root"
    assert meta["correlation_id"] == "cid-123"

    entry = json.loads(lines[1])
    assert entry["type"] == "iteration"
    assert "iteration_usage_summary" in entry
    assert "cumulative_usage_summary" in entry

    # Roundtrip: dict -> Iteration -> dict preserves usage subtrees.
    parsed = Iteration.from_dict(entry)
    assert parsed.iteration_usage_summary is not None
    assert parsed.cumulative_usage_summary is not None
    assert parsed.iteration_usage_summary.model_usage_summaries["root"].total_calls == 1
    assert parsed.iteration_usage_summary.model_usage_summaries["sub"].total_output_tokens == 3
    assert parsed.cumulative_usage_summary.model_usage_summaries["sub"].total_input_tokens == 2

    # Nested llm_calls usage should also survive.
    assert parsed.code_blocks
    nested = parsed.code_blocks[0].result.llm_calls[0]
    assert nested.root_model == "sub"
    assert nested.usage_summary.model_usage_summaries["sub"].total_calls == 1
