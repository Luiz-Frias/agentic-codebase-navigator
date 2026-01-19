from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
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
from rlm.domain.models.iteration import CodeBlock, Iteration
from rlm.domain.models.llm_request import LLMRequest, ToolChoice
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

if TYPE_CHECKING:
    from rlm.domain.agent_ports import ToolDefinition, ToolRegistryPort

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
    """Snapshot totals into a standalone UsageSummary.

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


def _tool_json_default(value: Any, /) -> Any:
    """Coerce tool results into JSON-friendly structures or raise TypeError."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set):
        return list(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = base64.b64encode(bytes(value)).decode("ascii")
        return {"__bytes__": encoded}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        return as_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@dataclass(slots=True, frozen=True)
class RLMOrchestrator:
    """Pure domain orchestrator (Phase 2).

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
        """Build context dict for policy callbacks.

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
        """Check if the iteration loop should stop early.

        Uses the injected StoppingPolicy if available, otherwise returns False.
        """
        if self.stopping_policy is None:
            return False
        return self.stopping_policy.should_stop(context)

    def _on_iteration_complete(self, context: dict[str, Any], result: ChatCompletion) -> None:
        """Notify policy that an iteration completed.

        Allows external apps to track state, update beliefs, etc.
        """
        if self.stopping_policy is not None:
            self.stopping_policy.on_iteration_complete(context, result)

    def _compress_result(self, result: str, max_tokens: int | None = None) -> str:
        """Compress a nested call result before returning to parent.

        Uses the injected ContextCompressor if available, otherwise passthrough.
        """
        if self.context_compressor is None:
            return result
        return self.context_compressor.compress(result, max_tokens)

    def _should_orchestrate_nested(self, prompt: str, depth: int) -> bool:
        """Check if a nested call should spawn a sub-orchestrator.

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
        """Execute a single tool call and return the result.

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
        """Execute a single tool call asynchronously and return the result.

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
        """Format a tool execution result as a conversation message.

        The content is JSON-serialized for consistent parsing by the LLM.
        """
        payload: Any
        if result["error"] is not None:
            payload = {"error": result["error"]}
        else:
            payload = result["result"]

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
        """Build an assistant message containing tool calls.

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
        """Execute the multi-turn tool calling loop (sync).

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
                final_completion = ChatCompletion(
                    root_model=self.llm.model_name,
                    prompt=prompt,
                    response="[Stopped by custom policy]",
                    usage_summary=_clone_usage_totals(usage_totals),
                    execution_time=time.perf_counter() - time_start,
                    finish_reason="policy_stop",
                )
                return final_completion

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
        """Execute the multi-turn tool calling loop (async).

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
                final_completion = ChatCompletion(
                    root_model=self.llm.model_name,
                    prompt=prompt,
                    response="[Stopped by custom policy]",
                    usage_summary=_clone_usage_totals(usage_totals),
                    execution_time=time.perf_counter() - time_start,
                    finish_reason="policy_stop",
                )
                return final_completion

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
        time_start = time.perf_counter()

        # Validate agent mode configuration
        if self.agent_mode == "tools":
            self._assert_tool_mode_supported()
            tool_definitions = self._build_tool_definitions()
            usage_totals: dict[str, ModelUsageSummary] = {}
            return self._tool_calling_loop(
                prompt,
                tool_definitions=tool_definitions,
                usage_totals=usage_totals,
                tool_choice=tool_choice,
            )

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
            self.system_prompt,
            query_metadata,
        )

        for i in range(max_iterations):
            iter_start = time.perf_counter()

            current_prompt: Prompt = message_history + [
                build_user_prompt(root_prompt=root_prompt, iteration=i),
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
            },
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
        tool_choice: ToolChoice | None = None,
    ) -> ChatCompletion:
        """Async variant of `completion()`.

        Notes:
        - We still execute code blocks sequentially to preserve environment semantics.
        - We use `asyncio.TaskGroup` + `asyncio.to_thread` to avoid blocking the event loop
          while loading context / executing code.

        """
        time_start = time.perf_counter()

        # Validate agent mode configuration
        if self.agent_mode == "tools":
            self._assert_tool_mode_supported()
            tool_definitions = self._build_tool_definitions()
            usage_totals: dict[str, ModelUsageSummary] = {}
            return await self._atool_calling_loop(
                prompt,
                tool_definitions=tool_definitions,
                usage_totals=usage_totals,
                tool_choice=tool_choice,
            )

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
            self.system_prompt,
            query_metadata,
        )

        for i in range(max_iterations):
            iter_start = time.perf_counter()
            current_prompt: Prompt = message_history + [
                build_user_prompt(root_prompt=root_prompt, iteration=i),
            ]

            # On the first iteration, load context and run the LLM call concurrently.
            if i == 0:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(
                        asyncio.to_thread(self.environment.load_context, prompt),  # type: ignore[arg-type]
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
            },
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
