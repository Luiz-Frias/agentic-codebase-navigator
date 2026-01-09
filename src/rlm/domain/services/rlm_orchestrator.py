from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from rlm.domain.models.completion import ChatCompletion
from rlm.domain.models.iteration import CodeBlock, Iteration
from rlm.domain.models.llm_request import LLMRequest
from rlm.domain.models.query_metadata import QueryMetadata
from rlm.domain.models.usage import ModelUsageSummary, UsageSummary
from rlm.domain.ports import EnvironmentPort, LLMPort, LoggerPort
from rlm.domain.services.parsing import (
    afind_final_answer,
    find_code_blocks,
    find_final_answer,
    format_iteration,
)
from rlm.domain.services.prompts import (
    RLM_SYSTEM_PROMPT,
    build_rlm_system_prompt,
    build_user_prompt,
)
from rlm.domain.types import Prompt


def _add_usage_totals(
    totals: dict[str, ModelUsageSummary],
    summary: UsageSummary,
    /,  # noqa: D401 - internal helper
) -> None:
    """Add a usage summary into a running totals dict (mutating totals in-place)."""
    for model, mus in summary.model_usage_summaries.items():
        current = totals.get(model)
        if current is None:
            totals[model] = ModelUsageSummary(
                total_calls=mus.total_calls,
                total_input_tokens=mus.total_input_tokens,
                total_output_tokens=mus.total_output_tokens,
            )
        else:
            current.total_calls += mus.total_calls
            current.total_input_tokens += mus.total_input_tokens
            current.total_output_tokens += mus.total_output_tokens


def _clone_usage_totals(totals: dict[str, ModelUsageSummary], /) -> UsageSummary:
    """
    Snapshot totals into a standalone UsageSummary.

    Notes:
    - Clones ModelUsageSummary objects to avoid aliasing (callers may mutate).
    - Inserts keys in sorted order for deterministic behavior.
    """
    return UsageSummary(
        model_usage_summaries={
            model: ModelUsageSummary(
                total_calls=mus.total_calls,
                total_input_tokens=mus.total_input_tokens,
                total_output_tokens=mus.total_output_tokens,
            )
            for model, mus in ((m, totals[m]) for m in sorted(totals))
        }
    )


@dataclass(slots=True, frozen=True)
class RLMOrchestrator:
    """
    Pure domain orchestrator (Phase 2).

    This implements the legacy iteration loop semantics using only domain ports.
    Environment/broker lifecycle is handled outside (composition root).
    """

    llm: LLMPort
    environment: EnvironmentPort
    logger: LoggerPort | None = None
    system_prompt: str = RLM_SYSTEM_PROMPT

    def completion(
        self,
        prompt: Prompt,
        *,
        root_prompt: str | None = None,
        max_depth: int = 1,
        depth: int = 0,
        max_iterations: int = 30,
        correlation_id: str | None = None,
    ) -> ChatCompletion:
        time_start = time.perf_counter()
        # Accumulate orchestrator (root) usage incrementally to avoid repeatedly
        # re-merging a growing list (can be quadratic in pathological cases).
        root_usage_totals: dict[str, ModelUsageSummary] = {}

        # Only compute per-iteration and cumulative usage snapshots when a logger
        # is present (they are emitted only through logger events).
        cumulative_usage_totals: dict[str, ModelUsageSummary] | None = (
            {} if self.logger is not None else None
        )

        # Fallback: if we're at max depth, treat as a plain LM call.
        if depth >= max_depth:
            cc = self.llm.complete(LLMRequest(prompt=prompt))
            _add_usage_totals(root_usage_totals, cc.usage_summary)
            final_answer = find_final_answer(cc.response)
            response = final_answer if final_answer is not None else cc.response
            time_end = time.perf_counter()
            return ChatCompletion(
                root_model=cc.root_model,
                prompt=prompt,
                response=response,
                usage_summary=_clone_usage_totals(root_usage_totals),
                execution_time=time_end - time_start,
            )

        # Load the prompt as "context" into the environment (legacy-compatible semantics).
        self.environment.load_context(prompt)  # type: ignore[arg-type]

        # Build initial message history (system + metadata hint).
        query_metadata = QueryMetadata.from_context(prompt)
        message_history: list[dict[str, str]] = build_rlm_system_prompt(
            self.system_prompt, query_metadata
        )

        for i in range(max_iterations):
            iter_start = time.perf_counter()

            current_prompt: Prompt = message_history + [
                build_user_prompt(root_prompt=root_prompt, iteration=i)
            ]

            llm_cc = self.llm.complete(LLMRequest(prompt=current_prompt))
            _add_usage_totals(root_usage_totals, llm_cc.usage_summary)
            response = llm_cc.response

            code_block_strs = find_code_blocks(response)
            code_blocks: list[CodeBlock] = []
            for code in code_block_strs:
                repl_result = self.environment.execute_code(code)
                repl_result.correlation_id = correlation_id
                code_blocks.append(CodeBlock(code=code, result=repl_result))

            final_answer = find_final_answer(response, environment=self.environment)
            iteration_time = time.perf_counter() - iter_start

            iteration_usage: UsageSummary | None = None
            cumulative_usage: UsageSummary | None = None
            if cumulative_usage_totals is not None:
                # Usage for this iteration: orchestrator call + any nested `llm_query()`
                # calls recorded by the environment in REPL results.
                iteration_totals: dict[str, ModelUsageSummary] = {}
                _add_usage_totals(iteration_totals, llm_cc.usage_summary)
                _add_usage_totals(cumulative_usage_totals, llm_cc.usage_summary)

                for cb in code_blocks:
                    for sub_cc in cb.result.llm_calls:
                        _add_usage_totals(iteration_totals, sub_cc.usage_summary)
                        _add_usage_totals(cumulative_usage_totals, sub_cc.usage_summary)

                iteration_usage = _clone_usage_totals(iteration_totals)
                cumulative_usage = _clone_usage_totals(cumulative_usage_totals)

            iteration = Iteration(
                correlation_id=correlation_id,
                prompt=current_prompt,
                response=response,
                code_blocks=code_blocks,
                final_answer=final_answer,
                iteration_time=iteration_time,
                iteration_usage_summary=iteration_usage,
                cumulative_usage_summary=cumulative_usage,
            )

            if self.logger is not None:
                self.logger.log_iteration(iteration)

            if final_answer is not None:
                time_end = time.perf_counter()
                return ChatCompletion(
                    root_model=self.llm.model_name,
                    prompt=prompt,
                    response=final_answer,
                    usage_summary=_clone_usage_totals(root_usage_totals),
                    execution_time=time_end - time_start,
                )

            # Carry state into the next iteration.
            message_history.extend(format_iteration(iteration))

        # Out of iterations: ask one final time for an answer.
        final_prompt: Prompt = message_history + [
            {
                "role": "user",
                "content": "Please provide a final answer to the user's question based on the information provided.",
            }
        ]
        last_cc = self.llm.complete(LLMRequest(prompt=final_prompt))
        _add_usage_totals(root_usage_totals, last_cc.usage_summary)
        extracted = find_final_answer(last_cc.response)
        time_end = time.perf_counter()
        return ChatCompletion(
            root_model=self.llm.model_name,
            prompt=prompt,
            response=extracted if extracted is not None else last_cc.response,
            usage_summary=_clone_usage_totals(root_usage_totals),
            execution_time=time_end - time_start,
        )

    async def acompletion(
        self,
        prompt: Prompt,
        *,
        root_prompt: str | None = None,
        max_depth: int = 1,
        depth: int = 0,
        max_iterations: int = 30,
        correlation_id: str | None = None,
    ) -> ChatCompletion:
        """
        Async variant of `completion()`.

        Notes:
        - We still execute code blocks sequentially to preserve environment semantics.
        - We use `asyncio.TaskGroup` + `asyncio.to_thread` to avoid blocking the event loop
          while loading context / executing code.
        """
        time_start = time.perf_counter()
        root_usage_totals: dict[str, ModelUsageSummary] = {}
        cumulative_usage_totals: dict[str, ModelUsageSummary] | None = (
            {} if self.logger is not None else None
        )

        if depth >= max_depth:
            cc = await self.llm.acomplete(LLMRequest(prompt=prompt))
            _add_usage_totals(root_usage_totals, cc.usage_summary)
            final_answer = await afind_final_answer(cc.response)
            response = final_answer if final_answer is not None else cc.response
            time_end = time.perf_counter()
            return ChatCompletion(
                root_model=cc.root_model,
                prompt=prompt,
                response=response,
                usage_summary=_clone_usage_totals(root_usage_totals),
                execution_time=time_end - time_start,
            )

        query_metadata = QueryMetadata.from_context(prompt)
        message_history: list[dict[str, str]] = build_rlm_system_prompt(
            self.system_prompt, query_metadata
        )

        for i in range(max_iterations):
            iter_start = time.perf_counter()
            current_prompt: Prompt = message_history + [
                build_user_prompt(root_prompt=root_prompt, iteration=i)
            ]

            # On the first iteration, load context and run the LLM call concurrently.
            if i == 0:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(
                        asyncio.to_thread(self.environment.load_context, prompt)  # type: ignore[arg-type]
                    )
                    llm_task = tg.create_task(self.llm.acomplete(LLMRequest(prompt=current_prompt)))
                llm_cc = llm_task.result()
            else:
                llm_cc = await self.llm.acomplete(LLMRequest(prompt=current_prompt))

            _add_usage_totals(root_usage_totals, llm_cc.usage_summary)
            response = llm_cc.response
            code_block_strs = find_code_blocks(response)
            code_blocks: list[CodeBlock] = []

            for code in code_block_strs:
                repl_result = await asyncio.to_thread(self.environment.execute_code, code)
                repl_result.correlation_id = correlation_id
                code_blocks.append(CodeBlock(code=code, result=repl_result))

            final_answer = await afind_final_answer(response, environment=self.environment)
            iteration_time = time.perf_counter() - iter_start

            iteration_usage: UsageSummary | None = None
            cumulative_usage: UsageSummary | None = None
            if cumulative_usage_totals is not None:
                iteration_totals: dict[str, ModelUsageSummary] = {}
                _add_usage_totals(iteration_totals, llm_cc.usage_summary)
                _add_usage_totals(cumulative_usage_totals, llm_cc.usage_summary)

                for cb in code_blocks:
                    for sub_cc in cb.result.llm_calls:
                        _add_usage_totals(iteration_totals, sub_cc.usage_summary)
                        _add_usage_totals(cumulative_usage_totals, sub_cc.usage_summary)

                iteration_usage = _clone_usage_totals(iteration_totals)
                cumulative_usage = _clone_usage_totals(cumulative_usage_totals)

            iteration = Iteration(
                correlation_id=correlation_id,
                prompt=current_prompt,
                response=response,
                code_blocks=code_blocks,
                final_answer=final_answer,
                iteration_time=iteration_time,
                iteration_usage_summary=iteration_usage,
                cumulative_usage_summary=cumulative_usage,
            )
            if self.logger is not None:
                self.logger.log_iteration(iteration)

            if final_answer is not None:
                time_end = time.perf_counter()
                return ChatCompletion(
                    root_model=self.llm.model_name,
                    prompt=prompt,
                    response=final_answer,
                    usage_summary=_clone_usage_totals(root_usage_totals),
                    execution_time=time_end - time_start,
                )

            message_history.extend(format_iteration(iteration))

        final_prompt: Prompt = message_history + [
            {
                "role": "user",
                "content": "Please provide a final answer to the user's question based on the information provided.",
            }
        ]
        last_cc = await self.llm.acomplete(LLMRequest(prompt=final_prompt))
        _add_usage_totals(root_usage_totals, last_cc.usage_summary)
        extracted = await afind_final_answer(last_cc.response)
        time_end = time.perf_counter()
        return ChatCompletion(
            root_model=self.llm.model_name,
            prompt=prompt,
            response=extracted if extracted is not None else last_cc.response,
            usage_summary=_clone_usage_totals(root_usage_totals),
            execution_time=time_end - time_start,
        )
