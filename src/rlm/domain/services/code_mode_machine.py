"""
Code-mode state machine for RLM orchestration.

This module provides a declarative StateMachine configuration for code-mode
orchestration. It replaces the nested for-loop in RLMOrchestrator.completion()
with explicit state transitions, guards, and actions.

State Machine Flow:
    INIT → SHALLOW_CALL (if depth >= max_depth)
    INIT → PROMPTING (normal case)
    PROMPTING → EXECUTING (code blocks found)
    PROMPTING → DONE (final answer found)
    EXECUTING → PROMPTING (continue iteration)
    EXECUTING → DONE (final answer or max iterations)
    SHALLOW_CALL → DONE (after single LLM call)

Usage:
    machine = build_code_mode_machine()
    ctx = CodeModeContext(prompt=prompt, max_iterations=30)
    final_state, final_ctx = machine.run(CodeModeState.INIT, ctx, event_source)

"""

from __future__ import annotations

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
from rlm.domain.models.state_machine import StateMachine

# ============================================================================
# Guards - Predicates that control transition flow
# Guards accept CodeModeEvent union for type safety with StateMachine.
# ============================================================================


def depth_exceeded_guard(_event: CodeModeEvent, ctx: CodeModeContext) -> bool:
    """
    Check if recursion depth has been exceeded.

    Returns:
        True if depth >= max_depth (should switch to shallow call mode)

    """
    return ctx.depth >= ctx.max_depth


def continue_loop_guard(_event: CodeModeEvent, ctx: CodeModeContext) -> bool:
    """
    Check if iteration loop should continue.

    The action will increment iteration, so we check if the NEXT iteration
    would be within bounds. iteration is 0-indexed, so after incrementing
    we want iteration < max_iterations.

    Returns:
        True if iteration + 1 < max_iterations (more iterations allowed)

    """
    return ctx.iteration + 1 < ctx.max_iterations


# ============================================================================
# Actions - Side effects triggered during transitions
# Actions accept CodeModeEvent union for type safety, then narrow internally.
# ============================================================================


def store_final_answer(event: CodeModeEvent, ctx: CodeModeContext) -> None:
    """Store the extracted final answer in context."""
    if isinstance(event, FinalAnswerFound):
        ctx.final_answer = event.answer


def increment_iteration(event: CodeModeEvent, ctx: CodeModeContext) -> None:
    """Increment iteration counter and store code blocks."""
    if isinstance(event, CodeExecuted):
        ctx.iteration += 1
        ctx.code_blocks = list(event.code_blocks)


def store_code_blocks_from_executed(event: CodeModeEvent, ctx: CodeModeContext) -> None:
    """Store executed code blocks in context."""
    if isinstance(event, CodeExecuted):
        ctx.code_blocks = list(event.code_blocks)


def store_llm_response(event: CodeModeEvent, ctx: CodeModeContext) -> None:
    """Store LLM completion in context."""
    if isinstance(event, LLMResponseReceived):
        ctx.last_completion = event.completion


# ============================================================================
# State Machine Builder
# ============================================================================


def build_code_mode_machine() -> StateMachine[CodeModeState, CodeModeEvent, CodeModeContext]:
    """
    Build the code-mode orchestration state machine.

    This creates a fully configured StateMachine with:
    - All code-mode states registered
    - All transitions with appropriate guards and actions
    - DONE marked as terminal state

    Returns:
        Configured StateMachine ready for execution

    """
    machine: StateMachine[CodeModeState, CodeModeEvent, CodeModeContext] = StateMachine()

    # Register states
    machine.state(CodeModeState.INIT)
    machine.state(CodeModeState.SHALLOW_CALL)
    machine.state(CodeModeState.PROMPTING)
    machine.state(CodeModeState.EXECUTING)
    machine.state(CodeModeState.DONE)

    # Mark terminal states
    machine.terminal(CodeModeState.DONE)

    # ─────────────────────────────────────────────────────────────────────────
    # INIT transitions
    # ─────────────────────────────────────────────────────────────────────────

    # INIT -> SHALLOW_CALL when depth exceeded
    machine.transition(
        CodeModeState.INIT,
        DepthExceeded,
        CodeModeState.SHALLOW_CALL,
        guard=depth_exceeded_guard,
    )

    # INIT -> PROMPTING on first LLM response (normal flow)
    machine.transition(
        CodeModeState.INIT,
        LLMResponseReceived,
        CodeModeState.PROMPTING,
        action=store_llm_response,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PROMPTING transitions
    # ─────────────────────────────────────────────────────────────────────────

    # PROMPTING -> DONE when final answer found
    machine.transition(
        CodeModeState.PROMPTING,
        FinalAnswerFound,
        CodeModeState.DONE,
        action=store_final_answer,
    )

    # PROMPTING -> EXECUTING when code blocks found
    machine.transition(
        CodeModeState.PROMPTING,
        CodeBlocksFound,
        CodeModeState.EXECUTING,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTING transitions
    # ─────────────────────────────────────────────────────────────────────────

    # EXECUTING -> DONE when final answer found
    machine.transition(
        CodeModeState.EXECUTING,
        FinalAnswerFound,
        CodeModeState.DONE,
        action=store_final_answer,
    )

    # EXECUTING -> DONE when max iterations reached
    machine.transition(
        CodeModeState.EXECUTING,
        MaxIterationsReached,
        CodeModeState.DONE,
    )

    # EXECUTING -> PROMPTING when code executed (continue loop)
    # Note: No guard/action needed - iteration tracking is in CodeModeEventSource
    machine.transition(
        CodeModeState.EXECUTING,
        CodeExecuted,
        CodeModeState.PROMPTING,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SHALLOW_CALL transitions
    # ─────────────────────────────────────────────────────────────────────────

    # SHALLOW_CALL -> DONE on LLM response
    machine.transition(
        CodeModeState.SHALLOW_CALL,
        LLMResponseReceived,
        CodeModeState.DONE,
        action=store_llm_response,
    )

    return machine
