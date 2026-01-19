"""
Tools-mode event source for StateMachine integration.

This module bridges the orchestrator's tool-calling dependencies (LLM, ToolRegistry)
into the event-driven StateMachine model. The event source observes the current state
and context, performs necessary side effects, and returns events that encode outcomes.

The event source is callable with signature:
    (state: ToolsModeState, ctx: ToolsModeContext) -> ToolsModeEvent | None

Usage:
    source = ToolsModeEventSource(llm=llm, tool_registry=registry)
    machine = build_tools_mode_machine()
    final_state, final_ctx = machine.run(ToolsModeState.INIT, ctx, source)

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rlm.domain.agent_ports import ToolCallRequest, ToolCallResult, ToolMessage
from rlm.domain.errors import ToolNotFoundError
from rlm.domain.models.llm_request import LLMRequest, ToolChoice
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
from rlm.domain.models.usage import ModelUsageSummary
from rlm.domain.services.prompts import RLM_SYSTEM_PROMPT
from rlm.domain.services.tool_serialization import tool_json_default

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from rlm.domain.agent_ports import StoppingPolicy, ToolDefinition, ToolRegistryPort
    from rlm.domain.models.usage import UsageSummary
    from rlm.domain.ports import LLMPort
    from rlm.domain.types import Prompt

    # Type alias for the sync summarizer callback
    ConversationSummarizer = Callable[
        [list[dict[str, Any]], list[ToolDefinition], dict[str, ModelUsageSummary]],
        list[dict[str, Any]],
    ]

    # Type alias for the async summarizer callback
    AsyncConversationSummarizer = Callable[
        [list[dict[str, Any]], list[ToolDefinition], dict[str, ModelUsageSummary]],
        Awaitable[list[dict[str, Any]]],
    ]


_POLICY_STOP_RESPONSE = "[Stopped by custom policy]"


def _mark_policy_stop(ctx: ToolsModeContext) -> None:
    ctx.policy_stop = True
    ctx.last_response = _POLICY_STOP_RESPONSE


@dataclass
class ToolsModeEventSource:
    """
    Event source that bridges tool-calling dependencies to StateMachine events.

    This callable observes the current state and context, interacts with LLM and
    tool registry as needed, and returns events that encode what happened.

    Attributes:
        llm: LLM port for generating responses.
        tool_registry: Tool registry port for looking up and executing tools.
        stopping_policy: Optional policy for early termination.
        system_prompt: System prompt for LLM (default from prompts module).
        summarizer: Optional callback to summarize long conversations before LLM calls.
                    Signature: (conversation, tool_definitions, usage_totals) -> conversation

    """

    llm: LLMPort
    tool_registry: ToolRegistryPort
    stopping_policy: StoppingPolicy | None = None
    system_prompt: str = ""
    summarizer: ConversationSummarizer | None = None

    def __post_init__(self) -> None:
        """Initialize system prompt if not provided."""
        if not self.system_prompt:
            self.system_prompt = RLM_SYSTEM_PROMPT

    def _maybe_summarize(self, ctx: ToolsModeContext) -> None:
        """
        Apply conversation summarization if a summarizer is configured.

        Modifies ctx.conversation in place if summarization is triggered.
        """
        if self.summarizer is None:
            return
        if not ctx.tool_definitions:
            return

        ctx.conversation = self.summarizer(
            ctx.conversation,
            ctx.tool_definitions,
            ctx.usage_totals,
        )

    def __call__(
        self,
        state: ToolsModeState,
        ctx: ToolsModeContext,
    ) -> ToolsModeEvent | None:
        """
        Generate an event based on the current state and context.

        Args:
            state: Current state of the state machine.
            ctx: Mutable context containing orchestration state.

        Returns:
            An event that encodes what happened, or None for terminal states.

        """
        if state == ToolsModeState.INIT:
            return self._handle_init(ctx)
        if state == ToolsModeState.PROMPTING:
            return self._handle_prompting(ctx)
        if state == ToolsModeState.EXECUTING_TOOLS:
            return self._handle_executing_tools(ctx)
        if state == ToolsModeState.DONE:
            return None

        return None

    def _handle_init(self, ctx: ToolsModeContext) -> ToolsModeEvent:
        """
        Handle INIT state - build conversation and call LLM.

        Flow:
        1. Build initial conversation with system prompt
        2. Add user prompt
        3. Apply context window summarization if needed
        4. Call LLM with tools
        5. Return LLMResponseReceived

        """
        # Build initial conversation
        ctx.conversation = self._build_tool_conversation(ctx.prompt)

        # Get tool definitions
        ctx.tool_definitions = self.tool_registry.list_definitions()

        # Apply context summarization if conversation is too large
        self._maybe_summarize(ctx)

        # Build request and call LLM
        tool_choice: ToolChoice = ctx.tool_choice if ctx.tool_choice is not None else "auto"
        request = LLMRequest(
            prompt=ctx.conversation,
            tools=ctx.tool_definitions,
            tool_choice=tool_choice,
        )
        completion = self.llm.complete(request)

        # Update context
        ctx.last_completion = completion
        ctx.last_response = completion.response
        self._add_usage(ctx, completion.usage_summary)

        # Notify stopping policy of iteration completion (INIT counts as iteration 0)
        self._on_iteration_complete(ctx)

        # Store tool calls for PROMPTING to process
        ctx.pending_tool_calls = list(completion.tool_calls) if completion.tool_calls else []

        return LLMResponseReceived(
            completion=completion,
            response_text=completion.response,
        )

    def _handle_prompting(self, ctx: ToolsModeContext) -> ToolsModeEvent:
        """
        Handle PROMPTING state - analyze response for tool calls or final answer.

        Priority:
        1. Check stopping policy
        2. Check for tool calls
        3. No tool calls = final answer

        """
        # Check stopping policy first
        if self._should_stop(ctx):
            _mark_policy_stop(ctx)
            return PolicyStop()

        # Check for tool calls
        if ctx.pending_tool_calls:
            return ToolCallsFound(tool_calls=ctx.pending_tool_calls)

        # No tool calls = final answer
        return NoToolCalls()

    def _handle_executing_tools(self, ctx: ToolsModeContext) -> ToolsModeEvent:
        """
        Handle EXECUTING_TOOLS state - execute pending tool calls.

        Flow:
        1. Add assistant message with tool calls to conversation
        2. Execute each tool call
        3. Add results to conversation
        4. Increment iteration counter
        5. Check max iterations - if reached, force final answer
        6. Call LLM for next iteration
        7. Return ToolsExecuted

        """
        # Add assistant tool call message to conversation
        if ctx.last_completion and ctx.pending_tool_calls:
            assistant_message = self._build_assistant_tool_call_message(
                ctx.pending_tool_calls,
                ctx.last_response or "",
            )
            ctx.conversation.append(assistant_message)

        # Execute each tool call
        results: list[ToolCallResult] = []
        for tool_call in ctx.pending_tool_calls:
            result = self._execute_tool_call(tool_call)
            results.append(result)

            # Add tool result to conversation
            tool_message = self._build_tool_result_message(result)
            ctx.conversation.append(tool_message)  # type: ignore[arg-type]

        # Clear pending tool calls
        ctx.pending_tool_calls = []

        # Increment iteration counter
        ctx.iteration += 1

        # Check stopping policy BEFORE calling LLM (matches original _tool_calling_loop)
        if self._should_stop(ctx):
            _mark_policy_stop(ctx)
            return PolicyStop()

        # Check if we've hit max iterations
        if ctx.iteration >= ctx.max_iterations:
            # Force a final answer by adding prompt and setting tool_choice="none"
            ctx.conversation.append(
                {
                    "role": "user",
                    "content": "Please provide your final answer based on the tool results.",
                },
            )
            # Apply context summarization before final answer request
            self._maybe_summarize(ctx)
            request = LLMRequest(
                prompt=ctx.conversation,
                tools=ctx.tool_definitions,
                tool_choice="none",  # Force text response
            )
            completion = self.llm.complete(request)
            ctx.last_completion = completion
            ctx.last_response = completion.response
            self._add_usage(ctx, completion.usage_summary)
            ctx.pending_tool_calls = []  # Ensure no more tool calls
            # Return MaxIterationsReached to signal state machine to transition to DONE
            return MaxIterationsReached()

        # Apply context summarization before next iteration
        self._maybe_summarize(ctx)

        # Call LLM for next iteration
        tool_choice: ToolChoice = ctx.tool_choice if ctx.tool_choice is not None else "auto"
        request = LLMRequest(
            prompt=ctx.conversation,
            tools=ctx.tool_definitions,
            tool_choice=tool_choice,
        )
        completion = self.llm.complete(request)

        # Update context
        ctx.last_completion = completion
        ctx.last_response = completion.response
        self._add_usage(ctx, completion.usage_summary)

        # Notify stopping policy of iteration completion
        self._on_iteration_complete(ctx)

        # Store new tool calls for PROMPTING
        ctx.pending_tool_calls = list(completion.tool_calls) if completion.tool_calls else []

        return ToolsExecuted(results=results)

    def _build_tool_conversation(self, prompt: Prompt | None) -> list[dict[str, Any]]:
        """Create the initial tool-mode conversation history."""
        conversation: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        if prompt is None:
            return conversation

        if isinstance(prompt, str):
            conversation.append({"role": "user", "content": prompt})
        elif isinstance(prompt, dict):
            conversation.append(dict(prompt))  # Convert Mapping to dict
        elif isinstance(prompt, list):
            conversation.extend(dict(msg) for msg in prompt)  # Convert Mappings to dicts
        else:
            conversation.append({"role": "user", "content": str(prompt)})

        return conversation

    def _execute_tool_call(self, tool_call: ToolCallRequest) -> ToolCallResult:
        """
        Execute a single tool call and return the result.

        Tool execution is a domain concern - the ToolPort protocol defines
        the contract. Exception handling happens here because tools are
        user-defined and can raise any exception type.

        Raises:
            ToolNotFoundError: If the requested tool is not in the registry.

        """
        tool = self.tool_registry.get(tool_call["name"])
        if tool is None:
            raise ToolNotFoundError(tool_call["name"])

        try:
            result = tool.execute(**tool_call["arguments"])
            return ToolCallResult(
                id=tool_call["id"],
                name=tool_call["name"],
                result=result,
                error=None,
            )
        except Exception as exc:
            # Capture and return as error for LLM context.
            return ToolCallResult(
                id=tool_call["id"],
                name=tool_call["name"],
                result=None,
                error=str(exc),
            )

    def _build_tool_result_message(self, result: ToolCallResult) -> ToolMessage:
        """Format a tool execution result as a conversation message."""
        payload: Any = (
            {"error": result["error"]} if result["error"] is not None else result["result"]
        )

        try:
            content = json.dumps(payload, default=tool_json_default)
        except Exception as exc:
            content = json.dumps({"error": f"Tool result serialization failed: {exc}"})

        return ToolMessage(
            role="tool",
            tool_call_id=result["id"],
            content=content,
        )

    def _build_assistant_tool_call_message(
        self,
        tool_calls: list[ToolCallRequest],
        response_text: str = "",
    ) -> dict[str, Any]:
        """Build an assistant message containing tool calls."""
        return {
            "role": "assistant",
            "content": response_text,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in tool_calls
            ],
        }

    def _should_stop(self, ctx: ToolsModeContext) -> bool:
        """Check if the stopping policy requests early termination."""
        if self.stopping_policy is None:
            return False

        policy_context = self._build_policy_context(ctx)
        return self.stopping_policy.should_stop(policy_context)

    def _add_usage(self, ctx: ToolsModeContext, summary: UsageSummary) -> None:
        """Add usage from a completion to the running totals."""
        if not hasattr(summary, "model_usage_summaries"):
            return

        for model, mus in summary.model_usage_summaries.items():
            current = ctx.usage_totals.get(model)
            if current is None:
                ctx.usage_totals[model] = ModelUsageSummary(
                    total_calls=mus.total_calls,
                    total_input_tokens=mus.total_input_tokens,
                    total_output_tokens=mus.total_output_tokens,
                )
            else:
                current.total_calls += mus.total_calls
                current.total_input_tokens += mus.total_input_tokens
                current.total_output_tokens += mus.total_output_tokens

    def _build_policy_context(self, ctx: ToolsModeContext) -> dict[str, Any]:
        """Build context dict for policy callbacks."""
        return {
            "iteration": ctx.iteration,
            "max_iterations": ctx.max_iterations,
            "depth": ctx.depth,
            "history": ctx.conversation,
            "last_result": ctx.last_completion,
        }

    def _on_iteration_complete(self, ctx: ToolsModeContext) -> None:
        """Notify policy that an iteration completed."""
        if self.stopping_policy is None:
            return
        if ctx.last_completion is None:
            return

        policy_context = self._build_policy_context(ctx)
        self.stopping_policy.on_iteration_complete(policy_context, ctx.last_completion)


# ============================================================================
# Async Event Source for StateMachine.arun()
# ============================================================================


@dataclass
class AsyncToolsModeEventSource:
    """
    Async event source for tools-mode StateMachine integration.

    This is the async counterpart to ToolsModeEventSource, designed for use with
    StateMachine.arun(). It uses async LLM calls while tool execution remains
    synchronous (tools are user-defined and may not support async).

    Attributes:
        llm: LLM port for generating responses (uses acomplete).
        tool_registry: Tool registry port for looking up and executing tools.
        stopping_policy: Optional policy for early termination.
        system_prompt: System prompt for LLM (default from prompts module).
        summarizer: Optional async callback to summarize long conversations.

    """

    llm: LLMPort
    tool_registry: ToolRegistryPort
    stopping_policy: StoppingPolicy | None = None
    system_prompt: str = ""
    summarizer: AsyncConversationSummarizer | None = None

    def __post_init__(self) -> None:
        """Initialize system prompt if not provided."""
        if not self.system_prompt:
            self.system_prompt = RLM_SYSTEM_PROMPT

    async def _maybe_summarize(self, ctx: ToolsModeContext) -> None:
        """Apply conversation summarization if a summarizer is configured (async)."""
        if self.summarizer is None:
            return
        if not ctx.tool_definitions:
            return

        ctx.conversation = await self.summarizer(
            ctx.conversation,
            ctx.tool_definitions,
            ctx.usage_totals,
        )

    async def __call__(
        self,
        state: ToolsModeState,
        ctx: ToolsModeContext,
    ) -> ToolsModeEvent | None:
        """
        Generate an event based on the current state and context (async).

        Args:
            state: Current state of the state machine.
            ctx: Mutable context containing orchestration state.

        Returns:
            An event that encodes what happened, or None for terminal states.

        """
        if state == ToolsModeState.INIT:
            return await self._handle_init(ctx)
        if state == ToolsModeState.PROMPTING:
            return self._handle_prompting(ctx)
        if state == ToolsModeState.EXECUTING_TOOLS:
            return await self._handle_executing_tools(ctx)
        if state == ToolsModeState.DONE:
            return None

        return None

    async def _handle_init(self, ctx: ToolsModeContext) -> ToolsModeEvent:
        """Handle INIT state - build conversation and call LLM (async)."""
        # Build initial conversation
        ctx.conversation = self._build_tool_conversation(ctx.prompt)

        # Get tool definitions
        ctx.tool_definitions = self.tool_registry.list_definitions()

        # Apply context summarization if conversation is too large
        await self._maybe_summarize(ctx)

        # Build request and call LLM (async)
        tool_choice: ToolChoice = ctx.tool_choice if ctx.tool_choice is not None else "auto"
        request = LLMRequest(
            prompt=ctx.conversation,
            tools=ctx.tool_definitions,
            tool_choice=tool_choice,
        )
        completion = await self.llm.acomplete(request)

        # Update context
        ctx.last_completion = completion
        ctx.last_response = completion.response
        self._add_usage(ctx, completion.usage_summary)

        # Notify stopping policy of iteration completion (INIT counts as iteration 0)
        self._on_iteration_complete(ctx)

        # Store tool calls for PROMPTING to process
        ctx.pending_tool_calls = list(completion.tool_calls) if completion.tool_calls else []

        return LLMResponseReceived(
            completion=completion,
            response_text=completion.response,
        )

    def _handle_prompting(self, ctx: ToolsModeContext) -> ToolsModeEvent:
        """
        Handle PROMPTING state - analyze response for tool calls or final answer.

        Note: This method is sync because it doesn't perform any I/O.
        """
        # Check stopping policy first
        if self._should_stop(ctx):
            _mark_policy_stop(ctx)
            return PolicyStop()

        # Check for tool calls
        if ctx.pending_tool_calls:
            return ToolCallsFound(tool_calls=ctx.pending_tool_calls)

        # No tool calls = final answer
        return NoToolCalls()

    async def _handle_executing_tools(self, ctx: ToolsModeContext) -> ToolsModeEvent:
        """Handle EXECUTING_TOOLS state - execute pending tool calls (async)."""
        # Add assistant tool call message to conversation
        if ctx.last_completion and ctx.pending_tool_calls:
            assistant_message = self._build_assistant_tool_call_message(
                ctx.pending_tool_calls,
                ctx.last_response or "",
            )
            ctx.conversation.append(assistant_message)

        # Execute each tool call (sync - tools may not support async)
        results: list[ToolCallResult] = []
        for tool_call in ctx.pending_tool_calls:
            result = self._execute_tool_call(tool_call)
            results.append(result)

            # Add tool result to conversation
            tool_message = self._build_tool_result_message(result)
            ctx.conversation.append(tool_message)  # type: ignore[arg-type]

        # Clear pending tool calls
        ctx.pending_tool_calls = []

        # Increment iteration counter
        ctx.iteration += 1

        # Check stopping policy BEFORE calling LLM (matches original _tool_calling_loop)
        if self._should_stop(ctx):
            _mark_policy_stop(ctx)
            return PolicyStop()

        # Check if we've hit max iterations
        if ctx.iteration >= ctx.max_iterations:
            # Force a final answer by adding prompt and setting tool_choice="none"
            ctx.conversation.append(
                {
                    "role": "user",
                    "content": "Please provide your final answer based on the tool results.",
                },
            )
            # Apply context summarization before final answer request
            await self._maybe_summarize(ctx)
            request = LLMRequest(
                prompt=ctx.conversation,
                tools=ctx.tool_definitions,
                tool_choice="none",  # Force text response
            )
            completion = await self.llm.acomplete(request)
            ctx.last_completion = completion
            ctx.last_response = completion.response
            self._add_usage(ctx, completion.usage_summary)
            ctx.pending_tool_calls = []  # Ensure no more tool calls
            # Return MaxIterationsReached to signal state machine to transition to DONE
            return MaxIterationsReached()

        # Apply context summarization before next iteration
        await self._maybe_summarize(ctx)

        # Call LLM for next iteration (async)
        tool_choice: ToolChoice = ctx.tool_choice if ctx.tool_choice is not None else "auto"
        request = LLMRequest(
            prompt=ctx.conversation,
            tools=ctx.tool_definitions,
            tool_choice=tool_choice,
        )
        completion = await self.llm.acomplete(request)

        # Update context
        ctx.last_completion = completion
        ctx.last_response = completion.response
        self._add_usage(ctx, completion.usage_summary)

        # Notify stopping policy of iteration completion
        self._on_iteration_complete(ctx)

        # Store new tool calls for PROMPTING
        ctx.pending_tool_calls = list(completion.tool_calls) if completion.tool_calls else []

        return ToolsExecuted(results=results)

    # ─────────────────────────────────────────────────────────────────────────
    # Helper methods (shared with sync version - could be extracted to mixin)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_tool_conversation(self, prompt: Prompt | None) -> list[dict[str, Any]]:
        """Create the initial tool-mode conversation history."""
        conversation: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        if prompt is None:
            return conversation

        if isinstance(prompt, str):
            conversation.append({"role": "user", "content": prompt})
        elif isinstance(prompt, dict):
            conversation.append(dict(prompt))  # Convert Mapping to dict
        elif isinstance(prompt, list):
            conversation.extend(dict(msg) for msg in prompt)  # Convert Mappings to dicts
        else:
            conversation.append({"role": "user", "content": str(prompt)})

        return conversation

    def _execute_tool_call(self, tool_call: ToolCallRequest) -> ToolCallResult:
        """
        Execute a single tool call and return the result.

        Note: Tool execution is synchronous - user-defined tools may not support async.
        """
        tool = self.tool_registry.get(tool_call["name"])
        if tool is None:
            raise ToolNotFoundError(tool_call["name"])

        try:
            result = tool.execute(**tool_call["arguments"])
            return ToolCallResult(
                id=tool_call["id"],
                name=tool_call["name"],
                result=result,
                error=None,
            )
        except Exception as exc:
            # Capture and return as error for LLM context.
            return ToolCallResult(
                id=tool_call["id"],
                name=tool_call["name"],
                result=None,
                error=str(exc),
            )

    def _build_tool_result_message(self, result: ToolCallResult) -> ToolMessage:
        """Format a tool execution result as a conversation message."""
        payload: Any = (
            {"error": result["error"]} if result["error"] is not None else result["result"]
        )

        try:
            content = json.dumps(payload, default=tool_json_default)
        except Exception as exc:
            content = json.dumps({"error": f"Tool result serialization failed: {exc}"})

        return ToolMessage(
            role="tool",
            tool_call_id=result["id"],
            content=content,
        )

    def _build_assistant_tool_call_message(
        self,
        tool_calls: list[ToolCallRequest],
        response_text: str = "",
    ) -> dict[str, Any]:
        """Build an assistant message containing tool calls."""
        return {
            "role": "assistant",
            "content": response_text,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in tool_calls
            ],
        }

    def _should_stop(self, ctx: ToolsModeContext) -> bool:
        """Check if the stopping policy requests early termination."""
        if self.stopping_policy is None:
            return False

        policy_context = self._build_policy_context(ctx)
        return self.stopping_policy.should_stop(policy_context)

    def _add_usage(self, ctx: ToolsModeContext, summary: UsageSummary) -> None:
        """Add usage from a completion to the running totals."""
        if not hasattr(summary, "model_usage_summaries"):
            return

        for model, mus in summary.model_usage_summaries.items():
            current = ctx.usage_totals.get(model)
            if current is None:
                ctx.usage_totals[model] = ModelUsageSummary(
                    total_calls=mus.total_calls,
                    total_input_tokens=mus.total_input_tokens,
                    total_output_tokens=mus.total_output_tokens,
                )
            else:
                current.total_calls += mus.total_calls
                current.total_input_tokens += mus.total_input_tokens
                current.total_output_tokens += mus.total_output_tokens

    def _build_policy_context(self, ctx: ToolsModeContext) -> dict[str, Any]:
        """Build context dict for policy callbacks."""
        return {
            "iteration": ctx.iteration,
            "max_iterations": ctx.max_iterations,
            "depth": ctx.depth,
            "history": ctx.conversation,
            "last_result": ctx.last_completion,
        }

    def _on_iteration_complete(self, ctx: ToolsModeContext) -> None:
        """Notify policy that an iteration completed."""
        if self.stopping_policy is None:
            return
        if ctx.last_completion is None:
            return

        policy_context = self._build_policy_context(ctx)
        self.stopping_policy.on_iteration_complete(policy_context, ctx.last_completion)
