"""
Tools-mode state machine for RLM orchestration.

This module provides a declarative StateMachine configuration for tools-mode
orchestration. It replaces the nested for-loop in _tool_calling_loop() with
explicit state transitions, guards, and actions.

State Machine Flow:
    INIT → PROMPTING (build conversation, call LLM)
    PROMPTING → EXECUTING_TOOLS (tool calls found)
    PROMPTING → DONE (no tool calls = final answer)
    PROMPTING → DONE (policy stop)
    EXECUTING_TOOLS → PROMPTING (tools executed, continue)
    EXECUTING_TOOLS → DONE (max iterations reached)
    EXECUTING_TOOLS → DONE (policy stop before next LLM call)

Usage:
    machine = build_tools_mode_machine()
    ctx = ToolsModeContext(max_iterations=10)
    final_state, final_ctx = machine.run(ToolsModeState.INIT, ctx, event_source)

"""

from __future__ import annotations

from rlm.domain.models.orchestration_types import (
    LLMResponseReceived,
    MaxIterationsReached,
    NoToolCalls,
    PolicyStop,
    ToolCallsFound,
    ToolsExecuted,
    ToolsModeContext,
    ToolsModeEvent,
    ToolsModeState,
)
from rlm.domain.models.state_machine import StateMachine

# ============================================================================
# Actions - Side effects triggered during transitions
# Actions accept ToolsModeEvent union for type safety, then narrow internally.
# ============================================================================


def tools_store_llm_response(event: ToolsModeEvent, ctx: ToolsModeContext) -> None:
    """Store LLM completion in context."""
    if isinstance(event, LLMResponseReceived):
        ctx.last_completion = event.completion


# Note: Iteration tracking is handled by ToolsModeEventSource, not the state machine.
# The event source increments ctx.iteration after tool execution and returns
# MaxIterationsReached when the limit is hit.


# ============================================================================
# State Machine Builder
# ============================================================================


def build_tools_mode_machine() -> StateMachine[ToolsModeState, ToolsModeEvent, ToolsModeContext]:
    """
    Build the tools-mode orchestration state machine.

    This creates a fully configured StateMachine with:
    - All tools-mode states registered
    - All transitions with appropriate guards and actions
    - DONE marked as terminal state

    Returns:
        Configured StateMachine ready for execution

    """
    machine: StateMachine[ToolsModeState, ToolsModeEvent, ToolsModeContext] = StateMachine()

    # Register states
    machine.state(ToolsModeState.INIT)
    machine.state(ToolsModeState.PROMPTING)
    machine.state(ToolsModeState.EXECUTING_TOOLS)
    machine.state(ToolsModeState.DONE)

    # Mark terminal states
    machine.terminal(ToolsModeState.DONE)

    # ─────────────────────────────────────────────────────────────────────────
    # INIT transitions
    # ─────────────────────────────────────────────────────────────────────────

    # INIT -> PROMPTING on first LLM response
    machine.transition(
        ToolsModeState.INIT,
        LLMResponseReceived,
        ToolsModeState.PROMPTING,
        action=tools_store_llm_response,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PROMPTING transitions
    # ─────────────────────────────────────────────────────────────────────────

    # PROMPTING -> DONE when no tool calls (final answer)
    machine.transition(
        ToolsModeState.PROMPTING,
        NoToolCalls,
        ToolsModeState.DONE,
    )

    # PROMPTING -> DONE on policy stop
    machine.transition(
        ToolsModeState.PROMPTING,
        PolicyStop,
        ToolsModeState.DONE,
    )

    # PROMPTING -> EXECUTING_TOOLS when tool calls found
    machine.transition(
        ToolsModeState.PROMPTING,
        ToolCallsFound,
        ToolsModeState.EXECUTING_TOOLS,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTING_TOOLS transitions
    # ─────────────────────────────────────────────────────────────────────────

    # EXECUTING_TOOLS -> DONE when max iterations reached
    machine.transition(
        ToolsModeState.EXECUTING_TOOLS,
        MaxIterationsReached,
        ToolsModeState.DONE,
    )

    # EXECUTING_TOOLS -> DONE on policy stop (checked before calling next iteration's LLM)
    machine.transition(
        ToolsModeState.EXECUTING_TOOLS,
        PolicyStop,
        ToolsModeState.DONE,
    )

    # EXECUTING_TOOLS -> PROMPTING when tools executed (continue loop)
    # Note: No guard/action needed - iteration tracking is in ToolsModeEventSource
    machine.transition(
        ToolsModeState.EXECUTING_TOOLS,
        ToolsExecuted,
        ToolsModeState.PROMPTING,
    )

    return machine
