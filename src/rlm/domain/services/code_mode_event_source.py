"""
Code-mode event source for StateMachine integration.

This module bridges the orchestrator's stateful dependencies (LLM, Environment, Logger)
into the event-driven StateMachine model. The event source observes the current state
and context, performs necessary side effects, and returns events that encode outcomes.

The event source is callable with signature:
    (state: CodeModeState, ctx: CodeModeContext) -> CodeModeEvent | None

Usage:
    source = CodeModeEventSource(llm=llm, environment=env, logger=logger)
    machine = build_code_mode_machine()
    final_state, final_ctx = machine.run(CodeModeState.INIT, ctx, source)

"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlm.domain.models.iteration import CodeBlock, Iteration
from rlm.domain.models.llm_request import LLMRequest
from rlm.domain.models.orchestration_types import (
    CodeBlocksFound,
    CodeExecuted,
    CodeModeContext,
    CodeModeEvent,
    CodeModeState,
    DepthExceeded,
    FinalAnswerFound,
    LLMResponseReceived,
    MaxIterationsReached,
)
from rlm.domain.models.query_metadata import QueryMetadata
from rlm.domain.models.usage import ModelUsageSummary, UsageSummary
from rlm.domain.services.parsing import find_code_blocks, find_final_answer, format_iteration
from rlm.domain.services.prompts import (
    RLM_SYSTEM_PROMPT,
    build_rlm_system_prompt,
    build_user_prompt,
)

if TYPE_CHECKING:
    from rlm.domain.ports import EnvironmentPort, LLMPort, LoggerPort


@dataclass
class CodeModeEventSource:
    """
    Event source that bridges orchestrator dependencies to StateMachine events.

    This callable observes the current state and context, interacts with LLM and
    Environment as needed, and returns events that encode what happened.

    Attributes:
        llm: LLM port for generating responses.
        environment: Environment port for code execution.
        logger: Optional logger for iteration tracking.
        system_prompt: System prompt for LLM (default from prompts module).

    """

    llm: LLMPort
    environment: EnvironmentPort
    logger: LoggerPort | None = None
    system_prompt: str = ""

    def __post_init__(self) -> None:
        """Initialize system prompt if not provided."""
        if not self.system_prompt:
            self.system_prompt = RLM_SYSTEM_PROMPT

    def __call__(
        self,
        state: CodeModeState,
        ctx: CodeModeContext,
    ) -> CodeModeEvent | None:
        """
        Generate an event based on the current state and context.

        Args:
            state: Current state of the state machine.
            ctx: Mutable context containing orchestration state.

        Returns:
            An event that encodes what happened, or None for terminal states.

        """
        if state == CodeModeState.INIT:
            return self._handle_init(ctx)
        if state == CodeModeState.PROMPTING:
            return self._handle_prompting(ctx)
        if state == CodeModeState.EXECUTING:
            return self._handle_executing(ctx)
        if state == CodeModeState.SHALLOW_CALL:
            return self._handle_shallow_call(ctx)
        if state == CodeModeState.DONE:
            return None

        return None

    def _handle_init(self, ctx: CodeModeContext) -> CodeModeEvent:
        """
        Handle INIT state - check depth, load context, call LLM.

        Flow:
        1. If depth >= max_depth, return DepthExceeded (shallow call path)
        2. Load context into environment
        3. Build initial message history
        4. Call LLM and return LLMResponseReceived

        """
        # Check depth limit first
        if ctx.depth >= ctx.max_depth:
            return DepthExceeded()

        # Load context into environment (legacy-compatible semantics)
        self.environment.load_context(ctx.prompt)  # type: ignore[arg-type]

        # Build initial message history
        if ctx.prompt is None:
            raise ValueError("CodeModeContext.prompt must be set before calling event source")
        query_metadata = QueryMetadata.from_context(ctx.prompt)
        ctx.message_history = build_rlm_system_prompt(self.system_prompt, query_metadata)

        # Build current prompt and call LLM
        current_prompt = [
            *ctx.message_history,
            build_user_prompt(root_prompt=ctx.root_prompt, iteration=ctx.iteration),
        ]

        completion = self.llm.complete(LLMRequest(prompt=current_prompt))
        ctx.last_completion = completion
        ctx.last_response = completion.response

        # Track usage
        self._add_usage(ctx, completion.usage_summary)

        return LLMResponseReceived(
            completion=completion,
            response_text=completion.response,
        )

    def _handle_prompting(self, ctx: CodeModeContext) -> CodeModeEvent:
        """
        Handle PROMPTING state - analyze response for code or final answer.

        Priority matches original orchestrator behavior:
        1. Check for code blocks FIRST (execute before checking final answer)
        2. Check for final answer only if no code blocks

        This ensures code is always executed before returning, matching the
        original behavior where nested LLM calls in code blocks are tracked.

        """
        response = ctx.last_response or ""

        # Check for code blocks FIRST - they take priority
        code_blocks = find_code_blocks(response)
        if code_blocks:
            ctx.pending_code_blocks = code_blocks
            return CodeBlocksFound(blocks=code_blocks)

        # No code blocks - check for final answer
        final_answer = find_final_answer(response, environment=self.environment)
        if final_answer is not None:
            # Log iteration before returning
            self._log_iteration(ctx, code_blocks=[], final_answer=final_answer)
            return FinalAnswerFound(answer=final_answer)

        # No code blocks and no final answer - continue iterating
        ctx.pending_code_blocks = []
        return CodeBlocksFound(blocks=[])

    def _handle_executing(self, ctx: CodeModeContext) -> CodeModeEvent:
        """
        Handle EXECUTING state - execute pending code blocks.

        Flow:
        1. Execute each pending code block
        2. Check for final answer in the ORIGINAL response (after code execution)
        3. Log iteration
        4. Increment iteration counter
        5. Check if max iterations reached → ask for final answer
        6. Otherwise, call LLM for next iteration

        Important: Final answer check happens AFTER code execution, matching
        the original orchestrator behavior.

        """
        code_blocks: list[CodeBlock] = []
        pending = ctx.pending_code_blocks or []

        # Execute pending code blocks
        for code in pending:
            result = self.environment.execute_code(code)
            result.correlation_id = ctx.correlation_id
            code_blocks.append(CodeBlock(code=code, result=result))

            # Note: Do NOT track nested LLM call usage here. The broker already
            # tracks subcall usage, and run_completion merges broker usage with
            # orchestrator usage at the end. Adding here would double-count.

        ctx.code_blocks = code_blocks
        ctx.pending_code_blocks = []

        # Check for final answer in the original response (AFTER code execution)
        final_answer = find_final_answer(ctx.last_response or "", environment=self.environment)
        if final_answer is not None:
            # Log iteration with code blocks and final answer
            self._log_iteration(ctx, code_blocks=code_blocks, final_answer=final_answer)
            ctx.final_answer = final_answer
            return FinalAnswerFound(answer=final_answer)

        # No final answer yet - log iteration and continue
        self._log_iteration(ctx, code_blocks=code_blocks, final_answer=None)

        # Update message history with iteration results
        self._update_message_history(ctx, code_blocks)

        # Increment iteration counter
        ctx.iteration += 1

        # Check if max iterations reached
        if ctx.iteration >= ctx.max_iterations:
            # Ask for final answer
            final_prompt = [
                *ctx.message_history,
                {
                    "role": "user",
                    "content": "Please provide a final answer to the user's question based on the information provided.",
                },
            ]
            completion = self.llm.complete(LLMRequest(prompt=final_prompt))
            ctx.last_completion = completion
            ctx.last_response = completion.response
            self._add_usage(ctx, completion.usage_summary)

            # Extract final answer if present
            final_answer = find_final_answer(completion.response)
            if final_answer is not None:
                ctx.final_answer = final_answer
                return FinalAnswerFound(answer=final_answer)

            # No explicit final answer, use the response
            ctx.final_answer = completion.response
            return MaxIterationsReached()

        # Not at max iterations - call LLM for next iteration
        current_prompt = [
            *ctx.message_history,
            build_user_prompt(root_prompt=ctx.root_prompt, iteration=ctx.iteration),
        ]
        completion = self.llm.complete(LLMRequest(prompt=current_prompt))
        ctx.last_completion = completion
        ctx.last_response = completion.response
        self._add_usage(ctx, completion.usage_summary)

        return CodeExecuted(code_blocks=code_blocks)

    def _handle_shallow_call(self, ctx: CodeModeContext) -> CodeModeEvent:
        """
        Handle SHALLOW_CALL state - direct LLM call without code execution.

        This path is taken when depth >= max_depth. Makes a simple LLM call
        and extracts any final answer if present.
        """
        if ctx.prompt is None:
            raise ValueError("CodeModeContext.prompt must be set for shallow call")
        completion = self.llm.complete(LLMRequest(prompt=ctx.prompt))
        ctx.last_completion = completion
        ctx.last_response = completion.response
        self._add_usage(ctx, completion.usage_summary)

        # Try to extract final answer
        final_answer = find_final_answer(completion.response)
        if final_answer is not None:
            ctx.final_answer = final_answer

        return LLMResponseReceived(
            completion=completion,
            response_text=completion.response,
        )

    def _add_usage(self, ctx: CodeModeContext, summary: UsageSummary) -> None:
        """Add usage from a completion to the running totals."""
        if not hasattr(summary, "model_usage_summaries"):
            return

        for model, mus in summary.model_usage_summaries.items():
            current = ctx.root_usage_totals.get(model)
            if current is None:
                ctx.root_usage_totals[model] = ModelUsageSummary(
                    total_calls=mus.total_calls,
                    total_input_tokens=mus.total_input_tokens,
                    total_output_tokens=mus.total_output_tokens,
                )
            else:
                current.total_calls += mus.total_calls
                current.total_input_tokens += mus.total_input_tokens
                current.total_output_tokens += mus.total_output_tokens

    def _update_message_history(
        self,
        ctx: CodeModeContext,
        code_blocks: list[CodeBlock],
    ) -> None:
        """Update message history with iteration results."""
        # Create a minimal iteration object for formatting
        iteration = Iteration(
            correlation_id=ctx.correlation_id,
            prompt=ctx.message_history,
            response=ctx.last_response or "",
            code_blocks=code_blocks,
            final_answer=None,
            iteration_time=0.0,
        )

        ctx.message_history.extend(format_iteration(iteration))

    def _log_iteration(
        self,
        ctx: CodeModeContext,
        code_blocks: list[CodeBlock],
        final_answer: str | None,
    ) -> None:
        """
        Log an iteration if a logger is configured.

        Computes iteration and cumulative usage summaries for the logger.
        """
        if self.logger is None:
            return

        # Build iteration-specific usage (current LLM call + any nested calls from code)
        iteration_totals: dict[str, ModelUsageSummary] = {}
        if ctx.last_completion is not None:
            self._add_usage_to_dict(iteration_totals, ctx.last_completion.usage_summary)

        # Add nested LLM call usage from code execution
        for cb in code_blocks:
            for sub_cc in cb.result.llm_calls:
                self._add_usage_to_dict(iteration_totals, sub_cc.usage_summary)

        iteration_usage = self._clone_usage(iteration_totals) if iteration_totals else None
        cumulative_usage = (
            self._clone_usage(ctx.root_usage_totals) if ctx.root_usage_totals else None
        )

        iteration = Iteration(
            correlation_id=ctx.correlation_id,
            prompt=ctx.message_history,
            response=ctx.last_response or "",
            code_blocks=code_blocks,
            final_answer=final_answer,
            iteration_time=0.0,  # Event source doesn't track timing
            iteration_usage_summary=iteration_usage,
            cumulative_usage_summary=cumulative_usage,
        )
        self.logger.log_iteration(iteration)

    def _add_usage_to_dict(
        self,
        totals: dict[str, ModelUsageSummary],
        summary: UsageSummary,
    ) -> None:
        """Add usage from a summary into a totals dict (mutating totals)."""
        if not hasattr(summary, "model_usage_summaries"):
            return

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

    def _clone_usage(self, totals: dict[str, ModelUsageSummary]) -> UsageSummary:
        """Clone usage totals into a UsageSummary."""
        return UsageSummary(
            model_usage_summaries={
                model: ModelUsageSummary(
                    total_calls=mus.total_calls,
                    total_input_tokens=mus.total_input_tokens,
                    total_output_tokens=mus.total_output_tokens,
                )
                for model, mus in totals.items()
            },
        )


@dataclass
class AsyncCodeModeEventSource:
    """
    Async event source that bridges orchestrator dependencies to StateMachine events.

    This is the async counterpart to CodeModeEventSource. Python's callable protocol
    limitation requires separate sync/async classes since `__call__` cannot be both
    sync and async in the same class.

    Attributes:
        llm: LLM port for generating responses.
        environment: Environment port for code execution.
        logger: Optional logger for iteration tracking.
        system_prompt: System prompt for LLM (default from prompts module).

    """

    llm: LLMPort
    environment: EnvironmentPort
    logger: LoggerPort | None = None
    system_prompt: str = ""

    def __post_init__(self) -> None:
        """Initialize system prompt if not provided."""
        if not self.system_prompt:
            self.system_prompt = RLM_SYSTEM_PROMPT

    async def __call__(
        self,
        state: CodeModeState,
        ctx: CodeModeContext,
    ) -> CodeModeEvent | None:
        """
        Generate an event based on the current state and context (async).

        Args:
            state: Current state of the state machine.
            ctx: Mutable context containing orchestration state.

        Returns:
            An event that encodes what happened, or None for terminal states.

        """
        if state == CodeModeState.INIT:
            return await self._handle_init(ctx)
        if state == CodeModeState.PROMPTING:
            return self._handle_prompting(ctx)  # Sync - no I/O
        if state == CodeModeState.EXECUTING:
            return await self._handle_executing(ctx)
        if state == CodeModeState.SHALLOW_CALL:
            return await self._handle_shallow_call(ctx)
        if state == CodeModeState.DONE:
            return None

        return None

    async def _handle_init(self, ctx: CodeModeContext) -> CodeModeEvent:
        """Handle INIT state async - check depth, load context, call LLM."""
        # Check depth limit first
        if ctx.depth >= ctx.max_depth:
            return DepthExceeded()

        # Load context into environment (sync → async bridge)
        await asyncio.to_thread(self.environment.load_context, ctx.prompt)  # type: ignore[arg-type]

        # Build initial message history
        if ctx.prompt is None:
            raise ValueError("CodeModeContext.prompt must be set before calling event source")
        query_metadata = QueryMetadata.from_context(ctx.prompt)
        ctx.message_history = build_rlm_system_prompt(self.system_prompt, query_metadata)

        # Build current prompt and call LLM (async)
        current_prompt = [
            *ctx.message_history,
            build_user_prompt(root_prompt=ctx.root_prompt, iteration=ctx.iteration),
        ]

        completion = await self.llm.acomplete(LLMRequest(prompt=current_prompt))
        ctx.last_completion = completion
        ctx.last_response = completion.response

        # Track usage
        self._add_usage(ctx, completion.usage_summary)

        return LLMResponseReceived(
            completion=completion,
            response_text=completion.response,
        )

    def _handle_prompting(self, ctx: CodeModeContext) -> CodeModeEvent:
        """
        Handle PROMPTING state - analyze response for code or final answer.

        This is sync since it doesn't do I/O - just parses the response.

        Priority matches original orchestrator behavior:
        1. Check for code blocks FIRST (execute before checking final answer)
        2. Check for final answer only if no code blocks

        This ensures code is always executed before returning, matching the
        original behavior where nested LLM calls in code blocks are tracked.
        """
        response = ctx.last_response or ""

        # Check for code blocks FIRST - they take priority
        code_blocks = find_code_blocks(response)
        if code_blocks:
            ctx.pending_code_blocks = code_blocks
            return CodeBlocksFound(blocks=code_blocks)

        # No code blocks - check for final answer
        final_answer = find_final_answer(response, environment=self.environment)
        if final_answer is not None:
            # Log iteration before returning
            self._log_iteration(ctx, code_blocks=[], final_answer=final_answer)
            return FinalAnswerFound(answer=final_answer)

        # No code blocks and no final answer - continue iterating
        ctx.pending_code_blocks = []
        return CodeBlocksFound(blocks=[])

    async def _handle_executing(self, ctx: CodeModeContext) -> CodeModeEvent:
        """
        Handle EXECUTING state async - execute pending code blocks.

        Flow:
        1. Execute each pending code block
        2. Check for final answer in the ORIGINAL response (after code execution)
        3. Log iteration
        4. Increment iteration counter
        5. Check if max iterations reached → ask for final answer
        6. Otherwise, call LLM for next iteration

        Important: Final answer check happens AFTER code execution, matching
        the original orchestrator behavior.
        """
        code_blocks: list[CodeBlock] = []
        pending = ctx.pending_code_blocks or []

        # Execute pending code blocks
        for code in pending:
            # Code execution is sync, so bridge to async
            result = await asyncio.to_thread(self.environment.execute_code, code)
            result.correlation_id = ctx.correlation_id
            code_blocks.append(CodeBlock(code=code, result=result))

            # Note: Do NOT track nested LLM call usage here. The broker already
            # tracks subcall usage, and run_completion merges broker usage with
            # orchestrator usage at the end. Adding here would double-count.

        ctx.code_blocks = code_blocks
        ctx.pending_code_blocks = []

        # Check for final answer in the original response (AFTER code execution)
        final_answer = find_final_answer(ctx.last_response or "", environment=self.environment)
        if final_answer is not None:
            # Log iteration with code blocks and final answer
            self._log_iteration(ctx, code_blocks=code_blocks, final_answer=final_answer)
            ctx.final_answer = final_answer
            return FinalAnswerFound(answer=final_answer)

        # No final answer yet - log iteration and continue
        self._log_iteration(ctx, code_blocks=code_blocks, final_answer=None)

        # Update message history with iteration results
        self._update_message_history(ctx, code_blocks)

        # Increment iteration counter
        ctx.iteration += 1

        # Check if max iterations reached
        if ctx.iteration >= ctx.max_iterations:
            # Ask for final answer
            final_prompt = [
                *ctx.message_history,
                {
                    "role": "user",
                    "content": "Please provide a final answer to the user's question based on the information provided.",
                },
            ]
            completion = await self.llm.acomplete(LLMRequest(prompt=final_prompt))
            ctx.last_completion = completion
            ctx.last_response = completion.response
            self._add_usage(ctx, completion.usage_summary)

            # Extract final answer if present
            final_answer = find_final_answer(completion.response)
            if final_answer is not None:
                ctx.final_answer = final_answer
                return FinalAnswerFound(answer=final_answer)

            # No explicit final answer, use the response
            ctx.final_answer = completion.response
            return MaxIterationsReached()

        # Not at max iterations - call LLM for next iteration
        current_prompt = [
            *ctx.message_history,
            build_user_prompt(root_prompt=ctx.root_prompt, iteration=ctx.iteration),
        ]
        completion = await self.llm.acomplete(LLMRequest(prompt=current_prompt))
        ctx.last_completion = completion
        ctx.last_response = completion.response
        self._add_usage(ctx, completion.usage_summary)

        return CodeExecuted(code_blocks=code_blocks)

    async def _handle_shallow_call(self, ctx: CodeModeContext) -> CodeModeEvent:
        """Handle SHALLOW_CALL state async - direct LLM call without code execution."""
        if ctx.prompt is None:
            raise ValueError("CodeModeContext.prompt must be set for shallow call")
        completion = await self.llm.acomplete(LLMRequest(prompt=ctx.prompt))
        ctx.last_completion = completion
        ctx.last_response = completion.response
        self._add_usage(ctx, completion.usage_summary)

        # Try to extract final answer
        final_answer = find_final_answer(completion.response)
        if final_answer is not None:
            ctx.final_answer = final_answer

        return LLMResponseReceived(
            completion=completion,
            response_text=completion.response,
        )

    def _add_usage(self, ctx: CodeModeContext, summary: UsageSummary) -> None:
        """Add usage from a completion to the running totals."""
        if not hasattr(summary, "model_usage_summaries"):
            return

        for model, mus in summary.model_usage_summaries.items():
            current = ctx.root_usage_totals.get(model)
            if current is None:
                ctx.root_usage_totals[model] = ModelUsageSummary(
                    total_calls=mus.total_calls,
                    total_input_tokens=mus.total_input_tokens,
                    total_output_tokens=mus.total_output_tokens,
                )
            else:
                current.total_calls += mus.total_calls
                current.total_input_tokens += mus.total_input_tokens
                current.total_output_tokens += mus.total_output_tokens

    def _update_message_history(
        self,
        ctx: CodeModeContext,
        code_blocks: list[CodeBlock],
    ) -> None:
        """Update message history with iteration results."""
        iteration = Iteration(
            correlation_id=ctx.correlation_id,
            prompt=ctx.message_history,
            response=ctx.last_response or "",
            code_blocks=code_blocks,
            final_answer=None,
            iteration_time=0.0,
        )
        ctx.message_history.extend(format_iteration(iteration))

    def _log_iteration(
        self,
        ctx: CodeModeContext,
        code_blocks: list[CodeBlock],
        final_answer: str | None,
    ) -> None:
        """
        Log an iteration if a logger is configured.

        Computes iteration and cumulative usage summaries for the logger.
        """
        if self.logger is None:
            return

        # Build iteration-specific usage (current LLM call + any nested calls from code)
        iteration_totals: dict[str, ModelUsageSummary] = {}
        if ctx.last_completion is not None:
            self._add_usage_to_dict(iteration_totals, ctx.last_completion.usage_summary)

        # Add nested LLM call usage from code execution
        for cb in code_blocks:
            for sub_cc in cb.result.llm_calls:
                self._add_usage_to_dict(iteration_totals, sub_cc.usage_summary)

        iteration_usage = self._clone_usage(iteration_totals) if iteration_totals else None
        cumulative_usage = (
            self._clone_usage(ctx.root_usage_totals) if ctx.root_usage_totals else None
        )

        iteration = Iteration(
            correlation_id=ctx.correlation_id,
            prompt=ctx.message_history,
            response=ctx.last_response or "",
            code_blocks=code_blocks,
            final_answer=final_answer,
            iteration_time=0.0,
            iteration_usage_summary=iteration_usage,
            cumulative_usage_summary=cumulative_usage,
        )
        self.logger.log_iteration(iteration)

    def _add_usage_to_dict(
        self,
        totals: dict[str, ModelUsageSummary],
        summary: UsageSummary,
    ) -> None:
        """Add usage from a summary into a totals dict (mutating totals)."""
        if not hasattr(summary, "model_usage_summaries"):
            return

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

    def _clone_usage(self, totals: dict[str, ModelUsageSummary]) -> UsageSummary:
        """Clone usage totals into a UsageSummary."""
        return UsageSummary(
            model_usage_summaries={
                model: ModelUsageSummary(
                    total_calls=mus.total_calls,
                    total_input_tokens=mus.total_input_tokens,
                    total_output_tokens=mus.total_output_tokens,
                )
                for model, mus in totals.items()
            },
        )
