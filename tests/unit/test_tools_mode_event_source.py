"""
TDD tests for ToolsModeEventSource.

These tests verify that the event source correctly produces events based on
the current state and context for tools-mode orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

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

pytestmark = pytest.mark.unit


# ============================================================================
# Test Fixtures - Mock Dependencies
# ============================================================================


@dataclass
class MockUsageSummary:
    """Minimal mock for UsageSummary."""

    model_usage_summaries: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockChatCompletion:
    """Minimal mock for ChatCompletion."""

    response: str
    root_model: str = "mock-model"
    usage_summary: MockUsageSummary = field(default_factory=MockUsageSummary)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None


@dataclass
class MockToolDefinition:
    """Minimal mock for ToolDefinition."""

    name: str
    description: str = "Mock tool"
    parameters: dict[str, Any] = field(default_factory=dict)


class MockTool:
    """Mock tool that returns configurable results."""

    def __init__(self, name: str, result: Any = "mock result") -> None:
        self.name = name
        self.result = result

    def execute(self, **kwargs: Any) -> Any:
        return self.result

    async def aexecute(self, **kwargs: Any) -> Any:
        return self.result


class MockToolRegistry:
    """Mock tool registry."""

    def __init__(self, tools: dict[str, MockTool] | None = None) -> None:
        self.tools = tools or {}

    def get(self, name: str) -> MockTool | None:
        return self.tools.get(name)

    def list_definitions(self) -> list[MockToolDefinition]:
        return [MockToolDefinition(name=name) for name in self.tools]


class MockLLM:
    """Mock LLM that returns configurable responses."""

    def __init__(
        self,
        responses: list[MockChatCompletion] | None = None,
    ) -> None:
        self.responses = responses or [MockChatCompletion(response="Default response")]
        self.call_count = 0
        self.model_name = "mock-model"
        self.supports_tools = True

    def complete(self, request: Any) -> MockChatCompletion:
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


class MockStoppingPolicy:
    """Mock stopping policy."""

    def __init__(self, should_stop: bool = False) -> None:
        self._should_stop = should_stop
        self.iterations_seen: list[dict[str, Any]] = []

    def should_stop(self, context: dict[str, Any]) -> bool:
        return self._should_stop

    def on_iteration_complete(self, context: dict[str, Any], result: Any) -> None:
        self.iterations_seen.append(context.copy())


# ============================================================================
# Phase 1: Event Source Protocol Tests
# ============================================================================


class TestToolsModeEventSourceProtocol:
    """Verify the event source follows the expected protocol."""

    def test_event_source_is_callable(self) -> None:
        """Event source is callable with (state, context) -> event."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource

        llm = MockLLM()
        registry = MockToolRegistry()
        source = ToolsModeEventSource(llm=llm, tool_registry=registry)

        ctx = ToolsModeContext(prompt="Test prompt")
        result = source(ToolsModeState.INIT, ctx)
        assert result is None or isinstance(result, tuple(ToolsModeEvent.__args__))  # type: ignore[attr-defined]

    def test_event_source_has_llm_and_registry(self) -> None:
        """Event source stores LLM and tool registry dependencies."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource

        llm = MockLLM()
        registry = MockToolRegistry()
        source = ToolsModeEventSource(llm=llm, tool_registry=registry)

        assert source.llm is llm
        assert source.tool_registry is registry

    def test_event_source_accepts_optional_stopping_policy(self) -> None:
        """Event source accepts an optional stopping policy."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource

        llm = MockLLM()
        registry = MockToolRegistry()
        policy = MockStoppingPolicy()
        source = ToolsModeEventSource(
            llm=llm,
            tool_registry=registry,
            stopping_policy=policy,
        )

        assert source.stopping_policy is policy


# ============================================================================
# Phase 2: INIT State Event Generation
# ============================================================================


class TestToolsModeEventSourceINITState:
    """Verify event generation from INIT state."""

    def test_init_builds_tool_conversation(self) -> None:
        """INIT builds initial conversation with system prompt."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource

        llm = MockLLM(responses=[MockChatCompletion(response="Hello")])
        registry = MockToolRegistry()
        source = ToolsModeEventSource(llm=llm, tool_registry=registry)

        ctx = ToolsModeContext(prompt="Test question")
        source(ToolsModeState.INIT, ctx)

        assert len(ctx.conversation) > 0

    def test_init_returns_llm_response_received(self) -> None:
        """INIT returns LLMResponseReceived after calling LLM."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource

        llm = MockLLM(responses=[MockChatCompletion(response="LLM response")])
        registry = MockToolRegistry()
        source = ToolsModeEventSource(llm=llm, tool_registry=registry)

        ctx = ToolsModeContext(prompt="Test question")
        event = source(ToolsModeState.INIT, ctx)

        assert isinstance(event, LLMResponseReceived)
        assert event.response_text == "LLM response"


# ============================================================================
# Phase 3: PROMPTING State Event Generation
# ============================================================================


class TestToolsModeEventSourcePROMPTINGState:
    """Verify event generation from PROMPTING state."""

    def test_prompting_returns_tool_calls_found_when_tools_requested(self) -> None:
        """PROMPTING returns ToolCallsFound when LLM requests tools."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource

        tool_calls = [
            {"id": "call_1", "name": "search", "arguments": {"q": "test"}},
        ]
        llm = MockLLM()
        registry = MockToolRegistry()
        source = ToolsModeEventSource(llm=llm, tool_registry=registry)

        ctx = ToolsModeContext(prompt="Test")
        ctx.pending_tool_calls = tool_calls  # Simulate tool calls found

        event = source(ToolsModeState.PROMPTING, ctx)

        assert isinstance(event, ToolCallsFound)
        assert len(event.tool_calls) == 1

    def test_prompting_returns_no_tool_calls_for_final_answer(self) -> None:
        """PROMPTING returns NoToolCalls when LLM provides final answer."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource

        llm = MockLLM()
        registry = MockToolRegistry()
        source = ToolsModeEventSource(llm=llm, tool_registry=registry)

        ctx = ToolsModeContext(prompt="Test")
        ctx.pending_tool_calls = []  # No tool calls = final answer

        event = source(ToolsModeState.PROMPTING, ctx)

        assert isinstance(event, NoToolCalls)

    def test_prompting_returns_policy_stop_when_policy_says_stop(self) -> None:
        """PROMPTING returns PolicyStop when stopping policy triggers."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource

        llm = MockLLM()
        registry = MockToolRegistry()
        policy = MockStoppingPolicy(should_stop=True)
        source = ToolsModeEventSource(
            llm=llm,
            tool_registry=registry,
            stopping_policy=policy,
        )

        ctx = ToolsModeContext(prompt="Test")
        ctx.pending_tool_calls = []  # No tool calls

        event = source(ToolsModeState.PROMPTING, ctx)

        assert isinstance(event, PolicyStop)
        assert ctx.policy_stop is True
        assert ctx.last_response == "[Stopped by custom policy]"


# ============================================================================
# Phase 4: EXECUTING_TOOLS State Event Generation
# ============================================================================


class TestToolsModeEventSourceEXECUTINGTOOLSState:
    """Verify event generation from EXECUTING_TOOLS state."""

    def test_executing_tools_executes_tool_calls(self) -> None:
        """EXECUTING_TOOLS executes pending tool calls."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource

        search_tool = MockTool(name="search", result="found it")
        llm = MockLLM()
        registry = MockToolRegistry(tools={"search": search_tool})
        source = ToolsModeEventSource(llm=llm, tool_registry=registry)

        ctx = ToolsModeContext(prompt="Test")
        ctx.pending_tool_calls = [
            {"id": "call_1", "name": "search", "arguments": {"q": "test"}},
        ]

        event = source(ToolsModeState.EXECUTING_TOOLS, ctx)

        assert isinstance(event, ToolsExecuted)
        assert len(event.results) == 1

    def test_executing_tools_returns_max_iterations_reached(self) -> None:
        """EXECUTING_TOOLS returns MaxIterationsReached at limit."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource

        search_tool = MockTool(name="search", result="result")
        llm = MockLLM()
        registry = MockToolRegistry(tools={"search": search_tool})
        source = ToolsModeEventSource(llm=llm, tool_registry=registry)

        ctx = ToolsModeContext(prompt="Test", iteration=9, max_iterations=10)
        ctx.pending_tool_calls = [
            {"id": "call_1", "name": "search", "arguments": {}},
        ]

        event = source(ToolsModeState.EXECUTING_TOOLS, ctx)

        # Should execute tools then check iterations
        assert isinstance(event, (ToolsExecuted, MaxIterationsReached))

    def test_executing_tools_handles_tool_errors(self) -> None:
        """EXECUTING_TOOLS captures errors from tool execution."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource

        class FailingTool:
            name = "failing"

            def execute(self, **kwargs: Any) -> Any:
                raise ValueError("Tool failed!")

        llm = MockLLM()
        registry = MockToolRegistry(tools={"failing": FailingTool()})  # type: ignore[dict-item]
        source = ToolsModeEventSource(llm=llm, tool_registry=registry)

        ctx = ToolsModeContext(prompt="Test")
        ctx.pending_tool_calls = [
            {"id": "call_1", "name": "failing", "arguments": {}},
        ]

        event = source(ToolsModeState.EXECUTING_TOOLS, ctx)

        assert isinstance(event, ToolsExecuted)
        # Error should be captured in result
        assert event.results[0]["error"] is not None

    def test_executing_tools_marks_policy_stop_when_policy_triggers(self) -> None:
        """EXECUTING_TOOLS marks policy stop before returning PolicyStop."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource

        search_tool = MockTool(name="search", result="found it")
        llm = MockLLM()
        registry = MockToolRegistry(tools={"search": search_tool})
        policy = MockStoppingPolicy(should_stop=True)
        source = ToolsModeEventSource(
            llm=llm,
            tool_registry=registry,
            stopping_policy=policy,
        )

        ctx = ToolsModeContext(prompt="Test")
        ctx.pending_tool_calls = [
            {"id": "call_1", "name": "search", "arguments": {"q": "test"}},
        ]

        event = source(ToolsModeState.EXECUTING_TOOLS, ctx)

        assert isinstance(event, PolicyStop)
        assert ctx.policy_stop is True
        assert ctx.last_response == "[Stopped by custom policy]"


# ============================================================================
# Phase 5: DONE State Event Generation
# ============================================================================


class TestToolsModeEventSourceDONEState:
    """Verify DONE state is terminal and produces no events."""

    def test_done_returns_none(self) -> None:
        """DONE state returns None (terminal, no more events)."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource

        llm = MockLLM()
        registry = MockToolRegistry()
        source = ToolsModeEventSource(llm=llm, tool_registry=registry)

        ctx = ToolsModeContext(prompt="Test")
        event = source(ToolsModeState.DONE, ctx)

        assert event is None


# ============================================================================
# Phase 6: Context Updates
# ============================================================================


class TestToolsModeEventSourceContextUpdates:
    """Verify event source properly updates context."""

    def test_init_updates_conversation(self) -> None:
        """INIT initializes conversation in context."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource

        llm = MockLLM(responses=[MockChatCompletion(response="Response")])
        registry = MockToolRegistry()
        source = ToolsModeEventSource(llm=llm, tool_registry=registry)

        ctx = ToolsModeContext(prompt="Test question")
        source(ToolsModeState.INIT, ctx)

        assert len(ctx.conversation) > 0

    def test_executing_tools_updates_conversation_with_results(self) -> None:
        """EXECUTING_TOOLS adds tool results to conversation."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource

        search_tool = MockTool(name="search", result="found")
        llm = MockLLM()
        registry = MockToolRegistry(tools={"search": search_tool})
        source = ToolsModeEventSource(llm=llm, tool_registry=registry)

        ctx = ToolsModeContext(prompt="Test")
        ctx.conversation = [{"role": "user", "content": "test"}]
        ctx.pending_tool_calls = [
            {"id": "call_1", "name": "search", "arguments": {}},
        ]

        source(ToolsModeState.EXECUTING_TOOLS, ctx)

        # Should have added tool result message
        assert len(ctx.conversation) > 1


# ============================================================================
# Phase 7: Integration with StateMachine
# ============================================================================


class TestToolsModeEventSourceStateMachineIntegration:
    """Verify event source works correctly with StateMachine."""

    def test_full_flow_init_to_done_no_tools(self) -> None:
        """Full flow: INIT -> PROMPTING -> DONE when no tools called."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        llm = MockLLM(responses=[MockChatCompletion(response="Direct answer")])
        registry = MockToolRegistry()
        source = ToolsModeEventSource(llm=llm, tool_registry=registry)

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext(prompt="What is 2+2?")

        final_state, final_ctx = machine.run(ToolsModeState.INIT, ctx, source)

        assert final_state == ToolsModeState.DONE

    def test_full_flow_with_tool_execution(self) -> None:
        """Full flow with tool: INIT -> PROMPTING -> EXECUTING -> PROMPTING -> DONE."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        search_tool = MockTool(name="search", result="Found: 42")

        # First response has tool call, second has final answer
        responses = [
            MockChatCompletion(
                response="Let me search...",
                tool_calls=[{"id": "call_1", "name": "search", "arguments": {"q": "answer"}}],
            ),
            MockChatCompletion(response="The answer is 42"),
        ]
        llm = MockLLM(responses=responses)
        registry = MockToolRegistry(tools={"search": search_tool})
        source = ToolsModeEventSource(llm=llm, tool_registry=registry)

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext(prompt="Search for the answer", max_iterations=10)

        final_state, final_ctx = machine.run(ToolsModeState.INIT, ctx, source)

        assert final_state == ToolsModeState.DONE
        assert final_ctx.iteration >= 1

    def test_policy_stop_flow(self) -> None:
        """Full flow with policy stop: INIT -> PROMPTING -> DONE."""
        from rlm.domain.services.tools_mode_event_source import ToolsModeEventSource
        from rlm.domain.services.tools_mode_machine import build_tools_mode_machine

        llm = MockLLM(responses=[MockChatCompletion(response="Working...")])
        registry = MockToolRegistry()
        policy = MockStoppingPolicy(should_stop=True)
        source = ToolsModeEventSource(
            llm=llm,
            tool_registry=registry,
            stopping_policy=policy,
        )

        machine = build_tools_mode_machine()
        ctx = ToolsModeContext(prompt="Test")

        final_state, final_ctx = machine.run(ToolsModeState.INIT, ctx, source)

        assert final_state == ToolsModeState.DONE
        assert final_ctx.policy_stop is True
        assert final_ctx.last_response == "[Stopped by custom policy]"
