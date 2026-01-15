"""
Integration tests for extension protocol injection.

Tests the full stack: RLM facade → orchestrator with custom policies injected.
Verifies that custom StoppingPolicy, ContextCompressor, and NestedCallPolicy
implementations are invoked correctly during execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.api.rlm import RLM
from rlm.domain.agent_ports import (
    NestedConfig,
    ToolCallRequest,
)
from rlm.domain.models import ChatCompletion


def _make_tool_call(tool_id: str, name: str, arguments: dict[str, Any]) -> ToolCallRequest:
    """Helper to create a ToolCallRequest."""
    return ToolCallRequest(id=tool_id, name=name, arguments=arguments)


# ---------------------------------------------------------------------------
# Custom Policy Implementations for Testing
# ---------------------------------------------------------------------------


@dataclass
class TrackingStoppingPolicy:
    """
    Custom stopping policy that tracks invocations for testing.

    Allows configuring early stopping based on iteration count or
    response content.
    """

    stop_at_iteration: int = 30
    stop_on_keyword: str | None = None
    invocation_log: list[dict[str, Any]] = field(default_factory=list)
    iteration_results: list[str] = field(default_factory=list)

    def should_stop(self, context: dict[str, Any]) -> bool:
        """Record invocation and check stopping conditions."""
        self.invocation_log.append({"method": "should_stop", "context": dict(context)})

        iteration = context.get("iteration", 0)
        if iteration >= self.stop_at_iteration:
            return True

        # Check for keyword in last result
        if self.stop_on_keyword and self.iteration_results:
            if self.stop_on_keyword in self.iteration_results[-1]:
                return True

        return False

    def on_iteration_complete(self, context: dict[str, Any], result: ChatCompletion) -> None:
        """Record iteration results for tracking."""
        self.invocation_log.append(
            {
                "method": "on_iteration_complete",
                "context": dict(context),
                "result_response": result.response,
            }
        )
        self.iteration_results.append(result.response)


@dataclass
class TrackingContextCompressor:
    """
    Custom context compressor that tracks invocations for testing.
    """

    prefix: str = "[COMPRESSED] "
    invocation_log: list[dict[str, Any]] = field(default_factory=list)

    def compress(self, result: str, max_tokens: int | None = None) -> str:
        """Record invocation and apply prefix."""
        self.invocation_log.append(
            {
                "method": "compress",
                "result_length": len(result),
                "max_tokens": max_tokens,
            }
        )
        return self.prefix + result


@dataclass
class TrackingNestedCallPolicy:
    """
    Custom nested call policy that tracks invocations for testing.
    """

    orchestrate_at_depth: int = 0  # Never orchestrate by default
    invocation_log: list[dict[str, Any]] = field(default_factory=list)

    def should_orchestrate(self, prompt: str, depth: int) -> bool:
        """Record invocation and check depth threshold."""
        self.invocation_log.append(
            {
                "method": "should_orchestrate",
                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                "depth": depth,
            }
        )
        return depth >= self.orchestrate_at_depth and self.orchestrate_at_depth > 0

    def get_nested_config(self) -> NestedConfig:
        """Return nested config."""
        self.invocation_log.append({"method": "get_nested_config"})
        return NestedConfig(agent_mode="code", max_iterations=5)


# ---------------------------------------------------------------------------
# Helper Tools for Testing
# ---------------------------------------------------------------------------


def simple_tool(value: str) -> str:
    """A simple tool that returns the value with a prefix.

    Args:
        value: The input value
    """
    return f"Result: {value}"


def multi_call_tool(count: int) -> str:
    """A tool that returns a message about call count.

    Args:
        count: A number
    """
    return f"Called with count={count}"


# ---------------------------------------------------------------------------
# Integration Tests - StoppingPolicy
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_stopping_policy_is_invoked_in_tool_mode() -> None:
    """Full stack test: Custom StoppingPolicy is invoked during tool calling loop."""
    policy = TrackingStoppingPolicy(stop_at_iteration=30)

    llm = MockLLMAdapter(
        model="mock-test",
        script=[
            # First call: LLM requests tool call
            {
                "tool_calls": [_make_tool_call("call_1", "simple_tool", {"value": "test"})],
                "response": "",
                "finish_reason": "tool_calls",
            },
            # Second call: LLM returns final answer
            "The result is: Result: test",
        ],
    )

    rlm = RLM(
        llm=llm,
        tools=[simple_tool],
        agent_mode="tools",
        stopping_policy=policy,
    )

    result = rlm.completion("Call the tool with 'test'")

    # Verify the policy was invoked
    assert len(policy.invocation_log) > 0

    # Verify should_stop was called
    should_stop_calls = [log for log in policy.invocation_log if log["method"] == "should_stop"]
    assert len(should_stop_calls) >= 1

    # Verify on_iteration_complete was called
    iteration_complete_calls = [
        log for log in policy.invocation_log if log["method"] == "on_iteration_complete"
    ]
    assert len(iteration_complete_calls) >= 1

    # Verify result is correct
    assert "Result: test" in result.response


@pytest.mark.integration
def test_stopping_policy_can_stop_early() -> None:
    """Full stack test: StoppingPolicy can stop iteration loop early."""
    # Policy that stops after first iteration
    policy = TrackingStoppingPolicy(stop_at_iteration=1)

    llm = MockLLMAdapter(
        model="mock-test",
        script=[
            # This response would normally trigger another iteration
            {
                "tool_calls": [_make_tool_call("call_1", "simple_tool", {"value": "one"})],
                "finish_reason": "tool_calls",
            },
            # This should not be reached due to early stopping
            {
                "tool_calls": [_make_tool_call("call_2", "simple_tool", {"value": "two"})],
                "finish_reason": "tool_calls",
            },
            "Final answer",
        ],
    )

    rlm = RLM(
        llm=llm,
        tools=[simple_tool],
        agent_mode="tools",
        stopping_policy=policy,
    )

    result = rlm.completion("Run tools")

    # With early stopping, the loop should exit before reaching "Final answer"
    # The result will be the tool result or intermediate state
    assert result is not None
    # Verify only one iteration was tracked
    assert len(policy.iteration_results) <= 1


@pytest.mark.integration
def test_stopping_policy_tracks_all_iterations() -> None:
    """Full stack test: StoppingPolicy tracks all iterations in multi-turn."""
    policy = TrackingStoppingPolicy(stop_at_iteration=30)

    llm = MockLLMAdapter(
        model="mock-test",
        script=[
            {
                "tool_calls": [_make_tool_call("call_1", "multi_call_tool", {"count": 1})],
                "finish_reason": "tool_calls",
            },
            {
                "tool_calls": [_make_tool_call("call_2", "multi_call_tool", {"count": 2})],
                "finish_reason": "tool_calls",
            },
            "All done!",
        ],
    )

    rlm = RLM(
        llm=llm,
        tools=[multi_call_tool],
        agent_mode="tools",
        stopping_policy=policy,
    )

    result = rlm.completion("Run multiple tools")

    # Should have tracked multiple iterations
    assert len(policy.iteration_results) >= 2
    assert result.response == "All done!"


# ---------------------------------------------------------------------------
# Integration Tests - ContextCompressor
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_context_compressor_injection() -> None:
    """Full stack test: Custom ContextCompressor is accepted by RLM."""
    compressor = TrackingContextCompressor(prefix="[TEST] ")

    llm = MockLLMAdapter(
        model="mock-test",
        script=["FINAL(Answer)"],
    )

    # Should not raise - compressor is accepted
    rlm = RLM(
        llm=llm,
        context_compressor=compressor,
    )

    result = rlm.completion("Test")
    assert result.response == "Answer"


# ---------------------------------------------------------------------------
# Integration Tests - NestedCallPolicy
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_nested_call_policy_injection() -> None:
    """Full stack test: Custom NestedCallPolicy is accepted by RLM."""
    policy = TrackingNestedCallPolicy(orchestrate_at_depth=0)

    llm = MockLLMAdapter(
        model="mock-test",
        script=["FINAL(Answer)"],
    )

    # Should not raise - policy is accepted
    rlm = RLM(
        llm=llm,
        nested_call_policy=policy,
    )

    result = rlm.completion("Test")
    assert result.response == "Answer"


# ---------------------------------------------------------------------------
# Integration Tests - All Policies Together
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_all_policies_can_be_injected_together() -> None:
    """Full stack test: All three policy types can be injected simultaneously."""
    stopping_policy = TrackingStoppingPolicy()
    compressor = TrackingContextCompressor()
    nested_policy = TrackingNestedCallPolicy()

    llm = MockLLMAdapter(
        model="mock-test",
        script=[
            {
                "tool_calls": [_make_tool_call("call_1", "simple_tool", {"value": "combo"})],
                "finish_reason": "tool_calls",
            },
            "Combined policies work!",
        ],
    )

    rlm = RLM(
        llm=llm,
        tools=[simple_tool],
        agent_mode="tools",
        stopping_policy=stopping_policy,
        context_compressor=compressor,
        nested_call_policy=nested_policy,
    )

    result = rlm.completion("Test with all policies")

    # Verify stopping policy was invoked
    assert len(stopping_policy.invocation_log) > 0

    # Verify result is correct
    assert result.response == "Combined policies work!"


@pytest.mark.integration
async def test_async_path_with_policies() -> None:
    """Full stack test: Async completion works with injected policies."""
    policy = TrackingStoppingPolicy()

    llm = MockLLMAdapter(
        model="mock-test",
        script=[
            {
                "tool_calls": [_make_tool_call("call_1", "simple_tool", {"value": "async"})],
                "finish_reason": "tool_calls",
            },
            "Async with policies works!",
        ],
    )

    rlm = RLM(
        llm=llm,
        tools=[simple_tool],
        agent_mode="tools",
        stopping_policy=policy,
    )

    result = await rlm.acompletion("Async test")

    # Verify policy was invoked in async path
    assert len(policy.invocation_log) > 0
    assert result.response == "Async with policies works!"


# ---------------------------------------------------------------------------
# Integration Tests - Code Mode with Policies
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_code_mode_with_stopping_policy() -> None:
    """Full stack test: StoppingPolicy works in code mode."""
    policy = TrackingStoppingPolicy(stop_at_iteration=30)

    llm = MockLLMAdapter(
        model="mock-test",
        script=["FINAL(Code mode with policy!)"],
    )

    rlm = RLM(
        llm=llm,
        agent_mode="code",
        stopping_policy=policy,
    )

    result = rlm.completion("Test prompt")

    # Code mode should still work with policy
    assert result.response == "Code mode with policy!"
