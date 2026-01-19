from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rlm.domain.agent_ports import (
    AgentModeName,
    ContextCompressor,
    NestedCallPolicy,
    StoppingPolicy,
    ToolCallRequest,
    ToolCallResult,
    ToolMessage,
)
from rlm.domain.errors import ToolNotFoundError
from rlm.domain.models.completion import ChatCompletion
from rlm.domain.models.llm_request import LLMRequest, ToolChoice
from rlm.domain.models.orchestration_types import (
    CodeModeContext,
    CodeModeState,
    ToolsModeContext,
    ToolsModeState,
)
from rlm.domain.models.usage import ModelUsageSummary, UsageSummary
from rlm.domain.services.code_mode_event_source import (
    AsyncCodeModeEventSource,
    CodeModeEventSource,
)
from rlm.domain.services.code_mode_machine import build_code_mode_machine
from rlm.domain.services.prompts import (
    RLM_SYSTEM_PROMPT,
)
from rlm.domain.services.tools_mode_event_source import (
    AsyncToolsModeEventSource,
    ToolsModeEventSource,
)
from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

if TYPE_CHECKING:
    from rlm.domain.agent_ports import ToolDefinition, ToolRegistryPort
    from rlm.domain.ports import EnvironmentPort, LLMPort, LoggerPort
    from rlm.domain.types import Prompt

# Backward compatibility alias
AgentMode = AgentModeName


def _add_usage_totals(
    totals: dict[str, ModelUsageSummary],
    summary: UsageSummary,
    /,
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
        },
    )


# Alias for backwards compatibility - implementation moved to tool_serialization.py
from rlm.domain.services.tool_serialization import tool_json_default as _tool_json_default


@dataclass(slots=True, frozen=True)
class RLMOrchestrator:
    """
    Pure domain orchestrator (Phase 2).

    This implements the legacy iteration loop semantics using only domain ports.
    Environment/broker lifecycle is handled outside (composition root).

    Agent Modes:
        - "code" (default): LLM generates Python code in ```repl blocks, which is
          executed in the environment. This is RLM's native paradigm.
        - "tools": LLM uses function calling to invoke registered tools. This
          mode complements code execution for structured tool interactions.

    Extension Protocols (Phase 2.7):
        - stopping_policy: Controls when iteration loops terminate. Inject custom
          implementations for EIG-gated stopping, entropy-based termination, etc.
        - context_compressor: Compresses nested call returns before bubbling up.
          Use for context budget management in deep orchestrator trees.
        - nested_call_policy: Determines when nested llm_query() calls should
          spawn sub-orchestrators vs. simple LLM calls.

    Note:
        Tool calling mode ("tools") requires a tool_registry. If agent_mode is
        "tools" but no registry is provided, a ValueError is raised at runtime.

    """

    llm: LLMPort
    environment: EnvironmentPort
    logger: LoggerPort | None = None
    system_prompt: str = RLM_SYSTEM_PROMPT

    # Agent capability extensions (Phase 1 - Core)
    agent_mode: AgentMode = "code"
    tool_registry: ToolRegistryPort | None = None

    # Tool calling configuration (Phase 2.4)
    max_tool_iterations: int = 10
    context_window_tokens: int | None = None
    tool_summary_trigger_ratio: float = 0.92
    tool_summary_keep_last_messages: int = 6
    tool_summary_min_messages: int = 8

    # Extension protocols (Phase 2.7-2.8)
    stopping_policy: StoppingPolicy | None = None
    context_compressor: ContextCompressor | None = None
    nested_call_policy: NestedCallPolicy | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Extension Protocol Helpers (Phase 2.7-2.8)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_policy_context(
        self,
        *,
        iteration: int,
        max_iterations: int,
        depth: int = 0,
        history: list[dict[str, Any]] | None = None,
        last_result: ChatCompletion | None = None,
    ) -> dict[str, Any]:
        """
        Build context dict for policy callbacks.

        This context is passed to StoppingPolicy methods and can be extended
        by external apps to track custom state (beliefs, EIG, etc.).
        """
        return {
            "iteration": iteration,
            "max_iterations": max_iterations,
            "agent_mode": self.agent_mode,
            "depth": depth,
            "history": history or [],
            "last_result": last_result,
        }

    def _should_stop(self, context: dict[str, Any]) -> bool:
        """
        Check if the iteration loop should stop early.

        Uses the injected StoppingPolicy if available, otherwise returns False.
        """
        if self.stopping_policy is None:
            return False
        return self.stopping_policy.should_stop(context)

    def _on_iteration_complete(self, context: dict[str, Any], result: ChatCompletion) -> None:
        """
        Notify policy that an iteration completed.

        Allows external apps to track state, update beliefs, etc.
        """
        if self.stopping_policy is not None:
            self.stopping_policy.on_iteration_complete(context, result)

    def _compress_result(self, result: str, max_tokens: int | None = None) -> str:
        """
        Compress a nested call result before returning to parent.

        Uses the injected ContextCompressor if available, otherwise passthrough.
        """
        if self.context_compressor is None:
            return result
        return self.context_compressor.compress(result, max_tokens)

    def _should_orchestrate_nested(self, prompt: str, depth: int) -> bool:
        """
        Check if a nested call should spawn a sub-orchestrator.

        Uses the injected NestedCallPolicy if available, otherwise returns False.
        """
        if self.nested_call_policy is None:
            return False
        return self.nested_call_policy.should_orchestrate(prompt, depth)

    # ─────────────────────────────────────────────────────────────────────────
    # Tool Calling Helpers (Phase 2.4)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_tool_definitions(self) -> list[ToolDefinition]:
        """Extract tool definitions from the registry for LLM context."""
        if self.tool_registry is None:
            return []
        return self.tool_registry.list_definitions()

    def _assert_tool_mode_supported(self) -> None:
        if self.tool_registry is None:
            raise ValueError("agent_mode='tools' requires a tool_registry to be provided")

        supports_tools = getattr(self.llm, "supports_tools", None)
        if supports_tools is False:
            raise ValueError(
                "agent_mode='tools' requires an LLM adapter that supports tool calling",
            )
        if supports_tools is None:
            tool_prompt_format = getattr(self.llm, "tool_prompt_format", "openai")
            if tool_prompt_format != "openai":
                raise ValueError(
                    "agent_mode='tools' requires supports_tools=True for non-OpenAI formats; "
                    f"adapter reports tool_prompt_format={tool_prompt_format!r}",
                )

    def _execute_tool_call(self, tool_call: ToolCallRequest, /) -> ToolCallResult:
        """
        Execute a single tool call and return the result.

        Args:
            tool_call: The tool call request from the LLM.

        Returns:
            ToolCallResult with either result or error populated.

        Raises:
            ToolNotFoundError: If the tool is not in the registry.

        """
        assert self.tool_registry is not None  # Caller ensures this

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
        except Exception as e:
            return ToolCallResult(
                id=tool_call["id"],
                name=tool_call["name"],
                result=None,
                error=str(e),
            )

    async def _aexecute_tool_call(self, tool_call: ToolCallRequest, /) -> ToolCallResult:
        """
        Execute a single tool call asynchronously and return the result.

        Args:
            tool_call: The tool call request from the LLM.

        Returns:
            ToolCallResult with either result or error populated.

        Raises:
            ToolNotFoundError: If the tool is not in the registry.

        """
        assert self.tool_registry is not None  # Caller ensures this

        tool = self.tool_registry.get(tool_call["name"])
        if tool is None:
            raise ToolNotFoundError(tool_call["name"])

        try:
            result = await tool.aexecute(**tool_call["arguments"])
            return ToolCallResult(
                id=tool_call["id"],
                name=tool_call["name"],
                result=result,
                error=None,
            )
        except Exception as e:
            return ToolCallResult(
                id=tool_call["id"],
                name=tool_call["name"],
                result=None,
                error=str(e),
            )

    def _build_tool_result_message(self, result: ToolCallResult, /) -> ToolMessage:
        """
        Format a tool execution result as a conversation message.

        The content is JSON-serialized for consistent parsing by the LLM.
        """
        payload: Any = (
            {"error": result["error"]} if result["error"] is not None else result["result"]
        )

        try:
            content = json.dumps(payload, default=_tool_json_default)
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
        """
        Build an assistant message containing tool calls.

        This follows the OpenAI chat format for assistant messages with tool_calls.
        """
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

    def _build_tool_conversation(self, prompt: Prompt, /) -> list[dict[str, Any]]:
        """Create the initial tool-mode conversation history."""
        conversation: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        if isinstance(prompt, str):
            conversation.append({"role": "user", "content": prompt})
        elif isinstance(prompt, dict):
            conversation.append(prompt)  # type: ignore[arg-type]
        elif isinstance(prompt, list):
            conversation.extend(prompt)  # type: ignore[arg-type]
        else:
            conversation.append({"role": "user", "content": str(prompt)})

        return conversation

    def _build_tool_completion(
        self,
        *,
        completion: ChatCompletion,
        prompt: Prompt,
        usage_totals: dict[str, ModelUsageSummary],
        time_start: float,
        finish_reason: str | None = None,
    ) -> ChatCompletion:
        time_end = time.perf_counter()
        return ChatCompletion(
            root_model=completion.root_model,
            prompt=prompt,
            response=completion.response,
            usage_summary=_clone_usage_totals(usage_totals),
            execution_time=time_end - time_start,
            tool_calls=completion.tool_calls,
            finish_reason=finish_reason or completion.finish_reason,
        )

    def _tools_mode_completion(
        self,
        prompt: Prompt,
        *,
        tool_choice: ToolChoice | None = None,
        depth: int = 0,
    ) -> ChatCompletion:
        """
        Execute tools-mode completion using StateMachine.

        This method replaces the manual iteration loop in _tool_calling_loop()
        with a declarative state machine approach. The state machine handles:
        - LLM calls with tool definitions
        - Tool execution via the registry
        - Policy-based early stopping
        - Conversation history management

        Args:
            prompt: The user's prompt to process.
            tool_choice: Tool selection constraint for the LLM.
            depth: Current recursion depth (for nested orchestration).

        Returns:
            ChatCompletion with the final response and accumulated usage.

        """
        time_start = time.perf_counter()

        # Build event source with orchestrator dependencies
        assert self.tool_registry is not None  # Caller ensures this

        # Create summarizer callback that wraps the orchestrator's summarization logic
        def summarizer(
            conversation: list[dict[str, Any]],
            tool_definitions: list[ToolDefinition],
            usage_totals: dict[str, ModelUsageSummary],
        ) -> list[dict[str, Any]]:
            return self._maybe_summarize_tool_conversation(
                conversation,
                tool_definitions=tool_definitions,
                usage_totals=usage_totals,
            )

        source = ToolsModeEventSource(
            llm=self.llm,
            tool_registry=self.tool_registry,
            stopping_policy=self.stopping_policy,
            system_prompt=self.system_prompt,
            summarizer=summarizer,
        )

        # Build context for state machine
        ctx = ToolsModeContext(
            prompt=prompt,
            max_iterations=self.max_tool_iterations,
            depth=depth,
            tool_choice=tool_choice,
        )

        # Run state machine to completion
        machine = build_tools_mode_machine()
        final_state, final_ctx = machine.run(ToolsModeState.INIT, ctx, source)

        # Determine finish reason based on final state and context
        finish_reason: str | None = None
        if final_state == ToolsModeState.DONE:
            # Check if we hit max iterations (iteration counter exceeds limit)
            if final_ctx.iteration >= final_ctx.max_iterations:
                finish_reason = "max_iterations"
            # Check if policy stopped us
            elif final_ctx.policy_stop or final_ctx.last_response == "[Stopped by custom policy]":
                finish_reason = "policy_stop"
            else:
                # Normal completion - LLM returned final answer without tool calls
                finish_reason = "stop"

        # Build final ChatCompletion from context
        time_end = time.perf_counter()
        return ChatCompletion(
            root_model=self.llm.model_name,
            prompt=prompt,
            response=final_ctx.last_response or "",
            usage_summary=_clone_usage_totals(final_ctx.usage_totals),
            execution_time=time_end - time_start,
            tool_calls=final_ctx.last_completion.tool_calls if final_ctx.last_completion else None,
            finish_reason=finish_reason,
        )

    async def _atools_mode_completion(
        self,
        prompt: Prompt,
        *,
        tool_choice: ToolChoice | None = None,
        depth: int = 0,
    ) -> ChatCompletion:
        """
        Execute async tools-mode completion using StateMachine.arun().

        This is the async counterpart to _tools_mode_completion().

        Args:
            prompt: The user's prompt to process.
            tool_choice: Tool selection constraint for the LLM.
            depth: Current recursion depth (for nested orchestration).

        Returns:
            ChatCompletion with the final response and accumulated usage.

        """
        time_start = time.perf_counter()

        # Build async event source with orchestrator dependencies
        assert self.tool_registry is not None  # Caller ensures this

        # Create async summarizer callback that wraps the orchestrator's summarization logic
        async def summarizer(
            conversation: list[dict[str, Any]],
            tool_definitions: list[ToolDefinition],
            usage_totals: dict[str, ModelUsageSummary],
        ) -> list[dict[str, Any]]:
            return await self._maybe_asummarize_tool_conversation(
                conversation,
                tool_definitions=tool_definitions,
                usage_totals=usage_totals,
            )

        source = AsyncToolsModeEventSource(
            llm=self.llm,
            tool_registry=self.tool_registry,
            stopping_policy=self.stopping_policy,
            system_prompt=self.system_prompt,
            summarizer=summarizer,
        )

        # Build context for state machine
        ctx = ToolsModeContext(
            prompt=prompt,
            max_iterations=self.max_tool_iterations,
            depth=depth,
            tool_choice=tool_choice,
        )

        # Run state machine to completion (async)
        machine = build_tools_mode_machine()
        final_state, final_ctx = await machine.arun(ToolsModeState.INIT, ctx, source)

        # Determine finish reason based on final state and context
        finish_reason: str | None = None
        if final_state == ToolsModeState.DONE:
            # Check if we hit max iterations (iteration counter exceeds limit)
            if final_ctx.iteration >= final_ctx.max_iterations:
                finish_reason = "max_iterations"
            # Check if policy stopped us
            elif final_ctx.policy_stop or final_ctx.last_response == "[Stopped by custom policy]":
                finish_reason = "policy_stop"
            else:
                # Normal completion - LLM returned final answer without tool calls
                finish_reason = "stop"

        # Build final ChatCompletion from context
        time_end = time.perf_counter()
        return ChatCompletion(
            root_model=self.llm.model_name,
            prompt=prompt,
            response=final_ctx.last_response or "",
            usage_summary=_clone_usage_totals(final_ctx.usage_totals),
            execution_time=time_end - time_start,
            tool_calls=final_ctx.last_completion.tool_calls if final_ctx.last_completion else None,
            finish_reason=finish_reason,
        )

    def _code_mode_completion(
        self,
        prompt: Prompt,
        *,
        root_prompt: str | None = None,
        max_iterations: int = 30,
        max_depth: int = 1,
        depth: int = 0,
        correlation_id: str | None = None,
    ) -> ChatCompletion:
        """
        Execute code-mode completion using StateMachine.

        This method replaces the manual iteration loop in completion() with
        a declarative state machine approach. The state machine handles:
        - LLM calls with code block detection
        - Code execution in the environment
        - Final answer extraction
        - Max iteration enforcement

        Args:
            prompt: The user's prompt to process.
            root_prompt: Optional override for the user prompt text.
            max_iterations: Maximum number of code execution iterations.
            max_depth: Maximum recursion depth for nested orchestration.
            depth: Current recursion depth.
            correlation_id: Optional correlation ID for logging.

        Returns:
            ChatCompletion with the final response and accumulated usage.

        """
        time_start = time.perf_counter()

        # Build event source with orchestrator dependencies
        source = CodeModeEventSource(
            llm=self.llm,
            environment=self.environment,
            logger=self.logger,
            system_prompt=self.system_prompt,
        )

        # Build context for state machine
        ctx = CodeModeContext(
            prompt=prompt,
            root_prompt=root_prompt,
            max_iterations=max_iterations,
            max_depth=max_depth,
            depth=depth,
            correlation_id=correlation_id,
        )

        # Run state machine to completion
        machine = build_code_mode_machine()
        _final_state, final_ctx = machine.run(CodeModeState.INIT, ctx, source)

        # Build final ChatCompletion from context
        time_end = time.perf_counter()
        response = (
            final_ctx.final_answer if final_ctx.final_answer else (final_ctx.last_response or "")
        )
        return ChatCompletion(
            root_model=self.llm.model_name,
            prompt=prompt,
            response=response,
            usage_summary=_clone_usage_totals(final_ctx.root_usage_totals),
            execution_time=time_end - time_start,
        )

    async def _acode_mode_completion(
        self,
        prompt: Prompt,
        *,
        root_prompt: str | None = None,
        max_iterations: int = 30,
        max_depth: int = 1,
        depth: int = 0,
        correlation_id: str | None = None,
    ) -> ChatCompletion:
        """
        Execute async code-mode completion using StateMachine.arun().

        This is the async counterpart to _code_mode_completion().

        Args:
            prompt: The user's prompt to process.
            root_prompt: Optional override for the user prompt text.
            max_iterations: Maximum number of code execution iterations.
            max_depth: Maximum recursion depth for nested orchestration.
            depth: Current recursion depth.
            correlation_id: Optional correlation ID for logging.

        Returns:
            ChatCompletion with the final response and accumulated usage.

        """
        time_start = time.perf_counter()

        # Build async event source with orchestrator dependencies
        source = AsyncCodeModeEventSource(
            llm=self.llm,
            environment=self.environment,
            logger=self.logger,
            system_prompt=self.system_prompt,
        )

        # Build context for state machine
        ctx = CodeModeContext(
            prompt=prompt,
            root_prompt=root_prompt,
            max_iterations=max_iterations,
            max_depth=max_depth,
            depth=depth,
            correlation_id=correlation_id,
        )

        # Run state machine to completion (async)
        machine = build_code_mode_machine()
        _final_state, final_ctx = await machine.arun(CodeModeState.INIT, ctx, source)

        # Build final ChatCompletion from context
        time_end = time.perf_counter()
        response = (
            final_ctx.final_answer if final_ctx.final_answer else (final_ctx.last_response or "")
        )
        return ChatCompletion(
            root_model=self.llm.model_name,
            prompt=prompt,
            response=response,
            usage_summary=_clone_usage_totals(final_ctx.root_usage_totals),
            execution_time=time_end - time_start,
        )

    def _context_window_tokens(self) -> int | None:
        """Return the best-known context window size for summarization."""
        candidates = (
            self.context_window_tokens,
            getattr(self.llm, "context_window_tokens", None),
            getattr(self.llm, "context_window", None),
            getattr(self.llm, "max_context_tokens", None),
        )
        for candidate in candidates:
            if isinstance(candidate, int) and candidate > 0:
                return candidate
        return None

    def _estimate_prompt_tokens_fallback(
        self,
        prompt: Prompt,
        tools: list[ToolDefinition] | None,
        /,
    ) -> int:
        payload: dict[str, Any] = {"prompt": prompt}
        if tools:
            payload["tools"] = tools
        try:
            raw = json.dumps(payload, ensure_ascii=True, default=str)
        except TypeError:
            raw = str(payload)
        return max(1, len(raw) // 4)

    def _estimate_prompt_tokens(self, prompt: Prompt, tools: list[ToolDefinition] | None, /) -> int:
        counter = getattr(self.llm, "count_prompt_tokens", None)
        if callable(counter):
            try:
                count = counter(LLMRequest(prompt=prompt, tools=tools))
            except Exception:
                count = None
            if isinstance(count, int) and count > 0:
                return count
        return self._estimate_prompt_tokens_fallback(prompt, tools)

    async def _aestimate_prompt_tokens(
        self,
        prompt: Prompt,
        tools: list[ToolDefinition] | None,
        /,
    ) -> int:
        counter = getattr(self.llm, "count_prompt_tokens", None)
        if callable(counter):
            try:
                count = await asyncio.to_thread(counter, LLMRequest(prompt=prompt, tools=tools))
            except Exception:
                count = None
            if isinstance(count, int) and count > 0:
                return count
        return self._estimate_prompt_tokens_fallback(prompt, tools)

    def _build_tool_summary_prompt(self, messages: list[dict[str, Any]], /) -> Prompt:
        summary_instructions = (
            "Summarize the conversation history for a tool-calling agent. "
            "Preserve the user goal, constraints, tool calls (ids, names, args), "
            "tool results/errors, and any partial decisions. Keep it compact but "
            "high-fidelity so the agent can continue without losing detail."
        )
        serialized = json.dumps(messages, ensure_ascii=True, default=str)
        return [
            {"role": "system", "content": summary_instructions},
            {"role": "user", "content": f"Messages:\n{serialized}"},
        ]

    def _maybe_summarize_tool_conversation(
        self,
        conversation: list[dict[str, Any]],
        *,
        tool_definitions: list[ToolDefinition],
        usage_totals: dict[str, ModelUsageSummary],
    ) -> list[dict[str, Any]]:
        context_window = self._context_window_tokens()
        if context_window is None:
            return conversation
        if len(conversation) < self.tool_summary_min_messages:
            return conversation

        estimated_tokens = self._estimate_prompt_tokens(conversation, tool_definitions)
        trigger_at = int(context_window * self.tool_summary_trigger_ratio)
        if estimated_tokens < trigger_at:
            return conversation

        keep_last = max(0, self.tool_summary_keep_last_messages)
        head_start = 1 if conversation and conversation[0].get("role") == "system" else 0
        tail = conversation[-keep_last:] if keep_last else []
        head = conversation[head_start : len(conversation) - len(tail)]
        if not head:
            return conversation

        summary_prompt = self._build_tool_summary_prompt(head)
        summary_completion = self.llm.complete(
            LLMRequest(prompt=summary_prompt, tool_choice="none"),
        )
        _add_usage_totals(usage_totals, summary_completion.usage_summary)
        summary_text = summary_completion.response.strip()
        if not summary_text:
            return conversation

        summary_message = {
            "role": "assistant",
            "content": f"Summary of prior conversation:\n{summary_text}",
        }
        rebuilt: list[dict[str, Any]] = []
        if head_start:
            rebuilt.append(conversation[0])
        rebuilt.append(summary_message)
        rebuilt.extend(tail)
        return rebuilt

    async def _maybe_asummarize_tool_conversation(
        self,
        conversation: list[dict[str, Any]],
        *,
        tool_definitions: list[ToolDefinition],
        usage_totals: dict[str, ModelUsageSummary],
    ) -> list[dict[str, Any]]:
        context_window = self._context_window_tokens()
        if context_window is None:
            return conversation
        if len(conversation) < self.tool_summary_min_messages:
            return conversation

        estimated_tokens = await self._aestimate_prompt_tokens(conversation, tool_definitions)
        trigger_at = int(context_window * self.tool_summary_trigger_ratio)
        if estimated_tokens < trigger_at:
            return conversation

        keep_last = max(0, self.tool_summary_keep_last_messages)
        head_start = 1 if conversation and conversation[0].get("role") == "system" else 0
        tail = conversation[-keep_last:] if keep_last else []
        head = conversation[head_start : len(conversation) - len(tail)]
        if not head:
            return conversation

        summary_prompt = self._build_tool_summary_prompt(head)
        summary_completion = await self.llm.acomplete(
            LLMRequest(prompt=summary_prompt, tool_choice="none"),
        )
        _add_usage_totals(usage_totals, summary_completion.usage_summary)
        summary_text = summary_completion.response.strip()
        if not summary_text:
            return conversation

        summary_message = {
            "role": "assistant",
            "content": f"Summary of prior conversation:\n{summary_text}",
        }
        rebuilt: list[dict[str, Any]] = []
        if head_start:
            rebuilt.append(conversation[0])
        rebuilt.append(summary_message)
        rebuilt.extend(tail)
        return rebuilt

    def _tool_calling_loop(
        self,
        prompt: Prompt,
        *,
        tool_definitions: list[ToolDefinition],
        usage_totals: dict[str, ModelUsageSummary],
        tool_choice: ToolChoice | None,
        depth: int = 0,
    ) -> ChatCompletion:
        """
        Execute the multi-turn tool calling loop (sync).

        The loop continues until:
        - The LLM returns a response without tool_calls (final answer)
        - max_tool_iterations is reached
        - StoppingPolicy.should_stop() returns True (custom early termination)

        Args:
            prompt: The initial user prompt.
            tool_definitions: Tools available for the LLM to call.
            usage_totals: Running usage totals to accumulate into.
            tool_choice: Tool choice constraint for the LLM.
            depth: Current recursion depth (for nested orchestration).

        Returns:
            ChatCompletion with the final response.

        """
        time_start = time.perf_counter()

        conversation = self._build_tool_conversation(prompt)
        request_tool_choice: ToolChoice = tool_choice if tool_choice is not None else "auto"

        # Build policy context for external state tracking
        policy_context = self._build_policy_context(
            iteration=0,
            max_iterations=self.max_tool_iterations,
            depth=depth,
            history=conversation,
            last_result=None,
        )

        for i in range(self.max_tool_iterations):
            # Update iteration in context
            policy_context["iteration"] = i
            policy_context["history"] = conversation

            # Check custom stopping policy
            if self._should_stop(policy_context):
                # Custom early stop - return current state
                return ChatCompletion(
                    root_model=self.llm.model_name,
                    prompt=prompt,
                    response="[Stopped by custom policy]",
                    usage_summary=_clone_usage_totals(usage_totals),
                    execution_time=time.perf_counter() - time_start,
                    finish_reason="policy_stop",
                )

            conversation = self._maybe_summarize_tool_conversation(
                conversation,
                tool_definitions=tool_definitions,
                usage_totals=usage_totals,
            )
            # Create request with tools
            request = LLMRequest(
                prompt=conversation,
                tools=tool_definitions,
                tool_choice=request_tool_choice,
            )

            # Call LLM
            completion = self.llm.complete(request)
            _add_usage_totals(usage_totals, completion.usage_summary)

            # Notify policy of iteration completion
            policy_context["last_result"] = completion
            self._on_iteration_complete(policy_context, completion)

            # If no tool calls, we have a final answer
            if not completion.tool_calls:
                return self._build_tool_completion(
                    completion=completion,
                    prompt=prompt,
                    usage_totals=usage_totals,
                    time_start=time_start,
                )

            # Add assistant's tool call message to conversation
            conversation.append(
                self._build_assistant_tool_call_message(completion.tool_calls, completion.response),
            )

            # Execute each tool call and add results to conversation
            for tool_call in completion.tool_calls:
                result = self._execute_tool_call(tool_call)
                tool_message = self._build_tool_result_message(result)
                conversation.append(tool_message)  # type: ignore[arg-type]

        # Max iterations reached - request final answer
        conversation.append(
            {
                "role": "user",
                "content": "Please provide your final answer based on the tool results.",
            },
        )
        conversation = self._maybe_summarize_tool_conversation(
            conversation,
            tool_definitions=tool_definitions,
            usage_totals=usage_totals,
        )
        final_request = LLMRequest(
            prompt=conversation,
            tools=tool_definitions,
            tool_choice="none",  # Force text response
        )
        final_completion = self.llm.complete(final_request)
        _add_usage_totals(usage_totals, final_completion.usage_summary)

        return self._build_tool_completion(
            completion=final_completion,
            prompt=prompt,
            usage_totals=usage_totals,
            time_start=time_start,
            finish_reason="max_iterations",
        )

    async def _atool_calling_loop(
        self,
        prompt: Prompt,
        *,
        tool_definitions: list[ToolDefinition],
        usage_totals: dict[str, ModelUsageSummary],
        tool_choice: ToolChoice | None,
        depth: int = 0,
    ) -> ChatCompletion:
        """
        Execute the multi-turn tool calling loop (async).

        Same logic as _tool_calling_loop but uses async LLM calls and tool execution.
        Includes StoppingPolicy integration for custom early termination.
        """
        time_start = time.perf_counter()

        conversation = self._build_tool_conversation(prompt)
        request_tool_choice: ToolChoice = tool_choice if tool_choice is not None else "auto"

        # Build policy context for external state tracking
        policy_context = self._build_policy_context(
            iteration=0,
            max_iterations=self.max_tool_iterations,
            depth=depth,
            history=conversation,
            last_result=None,
        )

        for i in range(self.max_tool_iterations):
            # Update iteration in context
            policy_context["iteration"] = i
            policy_context["history"] = conversation

            # Check custom stopping policy
            if self._should_stop(policy_context):
                # Custom early stop - return current state
                return ChatCompletion(
                    root_model=self.llm.model_name,
                    prompt=prompt,
                    response="[Stopped by custom policy]",
                    usage_summary=_clone_usage_totals(usage_totals),
                    execution_time=time.perf_counter() - time_start,
                    finish_reason="policy_stop",
                )

            conversation = await self._maybe_asummarize_tool_conversation(
                conversation,
                tool_definitions=tool_definitions,
                usage_totals=usage_totals,
            )
            # Create request with tools
            request = LLMRequest(
                prompt=conversation,
                tools=tool_definitions,
                tool_choice=request_tool_choice,
            )

            # Call LLM
            completion = await self.llm.acomplete(request)
            _add_usage_totals(usage_totals, completion.usage_summary)

            # Notify policy of iteration completion
            policy_context["last_result"] = completion
            self._on_iteration_complete(policy_context, completion)

            # If no tool calls, we have a final answer
            if not completion.tool_calls:
                return self._build_tool_completion(
                    completion=completion,
                    prompt=prompt,
                    usage_totals=usage_totals,
                    time_start=time_start,
                )

            # Add assistant's tool call message to conversation
            conversation.append(
                self._build_assistant_tool_call_message(completion.tool_calls, completion.response),
            )

            # Execute each tool call and add results to conversation
            for tool_call in completion.tool_calls:
                result = await self._aexecute_tool_call(tool_call)
                tool_message = self._build_tool_result_message(result)
                conversation.append(tool_message)  # type: ignore[arg-type]

        # Max iterations reached - request final answer
        conversation.append(
            {
                "role": "user",
                "content": "Please provide your final answer based on the tool results.",
            },
        )
        conversation = await self._maybe_asummarize_tool_conversation(
            conversation,
            tool_definitions=tool_definitions,
            usage_totals=usage_totals,
        )
        final_request = LLMRequest(
            prompt=conversation,
            tools=tool_definitions,
            tool_choice="none",  # Force text response
        )
        final_completion = await self.llm.acomplete(final_request)
        _add_usage_totals(usage_totals, final_completion.usage_summary)

        return self._build_tool_completion(
            completion=final_completion,
            prompt=prompt,
            usage_totals=usage_totals,
            time_start=time_start,
            finish_reason="max_iterations",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main Completion Methods
    # ─────────────────────────────────────────────────────────────────────────

    def completion(
        self,
        prompt: Prompt,
        *,
        root_prompt: str | None = None,
        max_depth: int = 1,
        depth: int = 0,
        max_iterations: int = 30,
        correlation_id: str | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> ChatCompletion:
        # Validate agent mode configuration
        if self.agent_mode == "tools":
            self._assert_tool_mode_supported()
            return self._tools_mode_completion(
                prompt,
                tool_choice=tool_choice,
                depth=depth,
            )

        # Code-mode: use StateMachine-based approach
        return self._code_mode_completion(
            prompt,
            root_prompt=root_prompt,
            max_iterations=max_iterations,
            max_depth=max_depth,
            depth=depth,
            correlation_id=correlation_id,
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
        tool_choice: ToolChoice | None = None,
    ) -> ChatCompletion:
        """
        Async variant of `completion()`.

        Notes:
        - We still execute code blocks sequentially to preserve environment semantics.
        - We use `asyncio.TaskGroup` + `asyncio.to_thread` to avoid blocking the event loop
          while loading context / executing code.

        """
        # Validate agent mode configuration
        if self.agent_mode == "tools":
            self._assert_tool_mode_supported()
            return await self._atools_mode_completion(
                prompt,
                tool_choice=tool_choice,
                depth=depth,
            )

        # Code-mode: use StateMachine-based approach
        return await self._acode_mode_completion(
            prompt,
            root_prompt=root_prompt,
            max_iterations=max_iterations,
            max_depth=max_depth,
            depth=depth,
            correlation_id=correlation_id,
        )
