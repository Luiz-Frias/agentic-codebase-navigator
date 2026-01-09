"""
Legacy → domain mapping helpers.

These functions live outside the domain layer by design: they depend on the
legacy snapshot types (`rlm._legacy.*`) and produce domain-owned models
(`rlm.domain.models.*`).
"""

from __future__ import annotations

from rlm._legacy.core import types as legacy_types
from rlm.domain import models as domain_models


def legacy_model_usage_summary_to_domain(
    legacy: legacy_types.ModelUsageSummary, /
) -> domain_models.ModelUsageSummary:
    return domain_models.ModelUsageSummary(
        total_calls=legacy.total_calls,
        total_input_tokens=legacy.total_input_tokens,
        total_output_tokens=legacy.total_output_tokens,
    )


def legacy_usage_summary_to_domain(
    legacy: legacy_types.UsageSummary, /
) -> domain_models.UsageSummary:
    return domain_models.UsageSummary(
        model_usage_summaries={
            model: legacy_model_usage_summary_to_domain(summary)
            for model, summary in legacy.model_usage_summaries.items()
        }
    )


def legacy_chat_completion_to_domain(
    legacy: legacy_types.RLMChatCompletion, /
) -> domain_models.ChatCompletion:
    return domain_models.ChatCompletion(
        root_model=legacy.root_model,
        prompt=legacy.prompt,
        response=legacy.response,
        usage_summary=legacy_usage_summary_to_domain(legacy.usage_summary),
        execution_time=legacy.execution_time,
    )


def legacy_repl_result_to_domain(legacy: legacy_types.REPLResult, /) -> domain_models.ReplResult:
    return domain_models.ReplResult(
        stdout=legacy.stdout,
        stderr=legacy.stderr,
        locals=dict(legacy.locals),
        llm_calls=[legacy_chat_completion_to_domain(c) for c in legacy.llm_calls],
        execution_time=legacy.execution_time or 0.0,
    )


def legacy_code_block_to_domain(legacy: legacy_types.CodeBlock, /) -> domain_models.CodeBlock:
    return domain_models.CodeBlock(
        code=legacy.code,
        result=legacy_repl_result_to_domain(legacy.result),
    )


def legacy_iteration_to_domain(legacy: legacy_types.RLMIteration, /) -> domain_models.Iteration:
    return domain_models.Iteration(
        prompt=legacy.prompt,
        response=legacy.response,
        code_blocks=[legacy_code_block_to_domain(b) for b in legacy.code_blocks],
        final_answer=legacy.final_answer,
        iteration_time=legacy.iteration_time or 0.0,
    )


def domain_model_usage_summary_to_legacy(
    domain: domain_models.ModelUsageSummary, /
) -> legacy_types.ModelUsageSummary:
    return legacy_types.ModelUsageSummary(
        total_calls=domain.total_calls,
        total_input_tokens=domain.total_input_tokens,
        total_output_tokens=domain.total_output_tokens,
    )


def domain_usage_summary_to_legacy(
    domain: domain_models.UsageSummary, /
) -> legacy_types.UsageSummary:
    return legacy_types.UsageSummary(
        model_usage_summaries={
            model: domain_model_usage_summary_to_legacy(summary)
            for model, summary in domain.model_usage_summaries.items()
        }
    )
