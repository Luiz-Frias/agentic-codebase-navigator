"""Unit tests for extension protocols and default implementations (Phase 2.7).

Tests cover:
- Protocol contract verification (runtime_checkable)
- DefaultStoppingPolicy behavior
- NoOpContextCompressor behavior
- SimpleNestedCallPolicy behavior
- Protocol duck typing (any object with matching methods works)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rlm.adapters.policies import (
    DefaultStoppingPolicy,
    NoOpContextCompressor,
    SimpleNestedCallPolicy,
)
from rlm.domain.agent_ports import (
    ContextCompressor,
    NestedCallPolicy,
    NestedConfig,
    StoppingPolicy,
)
from rlm.domain.models import ChatCompletion
from rlm.domain.models.usage import UsageSummary


def _make_chat_completion(response: str = "test") -> ChatCompletion:
    """Helper to create a ChatCompletion for testing."""
    return ChatCompletion(
        root_model="test",
        prompt="test",
        response=response,
        usage_summary=UsageSummary(),
        execution_time=0.1,
    )


# ---------------------------------------------------------------------------
# Protocol runtime_checkable verification
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stopping_policy_protocol_is_runtime_checkable() -> None:
    """StoppingPolicy protocol can be checked at runtime via isinstance."""
    policy = DefaultStoppingPolicy()
    assert isinstance(policy, StoppingPolicy)


@pytest.mark.unit
def test_context_compressor_protocol_is_runtime_checkable() -> None:
    """ContextCompressor protocol can be checked at runtime via isinstance."""
    compressor = NoOpContextCompressor()
    assert isinstance(compressor, ContextCompressor)


@pytest.mark.unit
def test_nested_call_policy_protocol_is_runtime_checkable() -> None:
    """NestedCallPolicy protocol can be checked at runtime via isinstance."""
    policy = SimpleNestedCallPolicy()
    assert isinstance(policy, NestedCallPolicy)


# ---------------------------------------------------------------------------
# Duck typing verification - custom implementations
# ---------------------------------------------------------------------------


@dataclass
class CustomStoppingPolicy:
    """Custom stopping policy for testing duck typing."""

    stop_at_iteration: int = 5
    iterations_seen: list[int] | None = None

    def __post_init__(self) -> None:
        self.iterations_seen = []

    def should_stop(self, context: dict[str, Any]) -> bool:
        iteration = context.get("iteration", 0)
        return iteration >= self.stop_at_iteration

    def on_iteration_complete(self, context: dict[str, Any], result: ChatCompletion) -> None:
        if self.iterations_seen is not None:
            self.iterations_seen.append(context.get("iteration", 0))


@dataclass
class CustomContextCompressor:
    """Custom context compressor for testing duck typing."""

    prefix: str = "[compressed] "

    def compress(self, result: str, max_tokens: int | None = None) -> str:
        if max_tokens is not None and len(result) > max_tokens:
            return self.prefix + result[:max_tokens] + "..."
        return self.prefix + result


@dataclass
class CustomNestedCallPolicy:
    """Custom nested call policy for testing duck typing."""

    orchestrate_above_depth: int = 2
    nested_max_iterations: int = 10

    def should_orchestrate(self, prompt: str, depth: int) -> bool:
        # Only orchestrate for "complex" prompts above threshold depth
        return depth >= self.orchestrate_above_depth and "complex" in prompt.lower()

    def get_nested_config(self) -> NestedConfig:
        return NestedConfig(
            agent_mode="tools",
            max_iterations=self.nested_max_iterations,
        )


@pytest.mark.unit
def test_custom_stopping_policy_satisfies_protocol() -> None:
    """Custom implementation satisfies StoppingPolicy protocol via duck typing."""
    policy = CustomStoppingPolicy(stop_at_iteration=3)
    assert isinstance(policy, StoppingPolicy)


@pytest.mark.unit
def test_custom_context_compressor_satisfies_protocol() -> None:
    """Custom implementation satisfies ContextCompressor protocol via duck typing."""
    compressor = CustomContextCompressor()
    assert isinstance(compressor, ContextCompressor)


@pytest.mark.unit
def test_custom_nested_call_policy_satisfies_protocol() -> None:
    """Custom implementation satisfies NestedCallPolicy protocol via duck typing."""
    policy = CustomNestedCallPolicy()
    assert isinstance(policy, NestedCallPolicy)


# ---------------------------------------------------------------------------
# DefaultStoppingPolicy tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_stopping_policy_stops_at_max_iterations() -> None:
    """DefaultStoppingPolicy returns True when iteration >= max_iterations."""
    policy = DefaultStoppingPolicy()

    # Below max - should not stop
    assert not policy.should_stop({"iteration": 0, "max_iterations": 30})
    assert not policy.should_stop({"iteration": 15, "max_iterations": 30})
    assert not policy.should_stop({"iteration": 29, "max_iterations": 30})

    # At max - should stop
    assert policy.should_stop({"iteration": 30, "max_iterations": 30})

    # Above max - should stop
    assert policy.should_stop({"iteration": 31, "max_iterations": 30})


@pytest.mark.unit
def test_default_stopping_policy_uses_default_max_iterations() -> None:
    """DefaultStoppingPolicy defaults to 30 max_iterations if not in context."""
    policy = DefaultStoppingPolicy()

    # Uses default of 30
    assert not policy.should_stop({"iteration": 29})
    assert policy.should_stop({"iteration": 30})


@pytest.mark.unit
def test_default_stopping_policy_defaults_iteration_to_zero() -> None:
    """DefaultStoppingPolicy defaults iteration to 0 if not in context."""
    policy = DefaultStoppingPolicy()

    # Empty context - iteration defaults to 0
    assert not policy.should_stop({})
    assert not policy.should_stop({"max_iterations": 30})


@pytest.mark.unit
def test_default_stopping_policy_on_iteration_complete_is_noop() -> None:
    """DefaultStoppingPolicy.on_iteration_complete is a no-op."""
    policy = DefaultStoppingPolicy()
    dummy_result = _make_chat_completion()

    # Should not raise - just a no-op
    policy.on_iteration_complete({}, dummy_result)
    policy.on_iteration_complete({"iteration": 5}, dummy_result)


@pytest.mark.unit
def test_default_stopping_policy_with_zero_max_iterations() -> None:
    """DefaultStoppingPolicy stops immediately if max_iterations is 0."""
    policy = DefaultStoppingPolicy()

    assert policy.should_stop({"iteration": 0, "max_iterations": 0})


# ---------------------------------------------------------------------------
# NoOpContextCompressor tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_noop_context_compressor_returns_unchanged() -> None:
    """NoOpContextCompressor returns result unchanged (passthrough)."""
    compressor = NoOpContextCompressor()

    result = "This is a very long result from a nested call..."
    assert compressor.compress(result) == result


@pytest.mark.unit
def test_noop_context_compressor_ignores_max_tokens() -> None:
    """NoOpContextCompressor ignores max_tokens hint (passed positionally)."""
    compressor = NoOpContextCompressor()

    long_result = "x" * 10000
    # Note: second param is _max_tokens, must be passed positionally
    assert compressor.compress(long_result, 100) == long_result
    assert compressor.compress(long_result, 10) == long_result


@pytest.mark.unit
def test_noop_context_compressor_handles_empty_string() -> None:
    """NoOpContextCompressor handles empty string."""
    compressor = NoOpContextCompressor()

    assert compressor.compress("") == ""
    assert compressor.compress("", 100) == ""


@pytest.mark.unit
def test_noop_context_compressor_handles_special_characters() -> None:
    """NoOpContextCompressor handles special characters."""
    compressor = NoOpContextCompressor()

    special = "Hello\n\tWorld\r\n🎉 Special chars: <>&\"'"
    assert compressor.compress(special) == special


# ---------------------------------------------------------------------------
# SimpleNestedCallPolicy tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_simple_nested_call_policy_never_orchestrates() -> None:
    """SimpleNestedCallPolicy always returns False for should_orchestrate."""
    policy = SimpleNestedCallPolicy()

    # Never orchestrates regardless of prompt or depth
    # Note: params are _prompt, _depth - must be passed positionally
    assert not policy.should_orchestrate("What is 2+2?", 0)
    assert not policy.should_orchestrate("Complex multi-step task", 0)
    assert not policy.should_orchestrate("Simple question", 1)
    assert not policy.should_orchestrate("Deep recursive task", 10)
    assert not policy.should_orchestrate("", 100)


@pytest.mark.unit
def test_simple_nested_call_policy_returns_empty_config() -> None:
    """SimpleNestedCallPolicy returns empty NestedConfig."""
    policy = SimpleNestedCallPolicy()

    config = policy.get_nested_config()

    # Returns empty NestedConfig (TypedDict with no required keys)
    assert config == {}
    assert isinstance(config, dict)


# ---------------------------------------------------------------------------
# Custom implementation behavior tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_custom_stopping_policy_behavior() -> None:
    """CustomStoppingPolicy tracks iterations and stops at threshold."""
    policy = CustomStoppingPolicy(stop_at_iteration=3)
    dummy_result = _make_chat_completion()

    # Track iterations
    for i in range(5):
        if not policy.should_stop({"iteration": i}):
            policy.on_iteration_complete({"iteration": i}, dummy_result)

    # Should have tracked iterations 0, 1, 2 (stopped at 3)
    assert policy.iterations_seen == [0, 1, 2]


@pytest.mark.unit
def test_custom_context_compressor_behavior() -> None:
    """CustomContextCompressor truncates and prefixes."""
    compressor = CustomContextCompressor(prefix="[COMPRESSED] ")

    # Without max_tokens - just prefix
    assert compressor.compress("hello") == "[COMPRESSED] hello"

    # With max_tokens - truncate
    result = compressor.compress("hello world", max_tokens=5)
    assert result == "[COMPRESSED] hello..."


@pytest.mark.unit
def test_custom_nested_call_policy_behavior() -> None:
    """CustomNestedCallPolicy orchestrates based on depth and prompt."""
    policy = CustomNestedCallPolicy(orchestrate_above_depth=2)

    # Below depth threshold - no orchestration
    assert not policy.should_orchestrate("complex task", 0)
    assert not policy.should_orchestrate("complex task", 1)

    # At depth threshold but no "complex" keyword - no orchestration
    assert not policy.should_orchestrate("simple task", 2)

    # At depth threshold with "complex" keyword - orchestrate
    assert policy.should_orchestrate("This is a complex task", 2)
    assert policy.should_orchestrate("COMPLEX analysis", 3)

    # Config should have custom settings
    config = policy.get_nested_config()
    assert config.get("agent_mode") == "tools"
    assert config.get("max_iterations") == 10


# ---------------------------------------------------------------------------
# NestedConfig TypedDict tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_nested_config_is_total_false() -> None:
    """NestedConfig TypedDict has total=False (all keys optional)."""
    # Empty config is valid
    config: NestedConfig = {}
    assert config == {}

    # Partial config is valid
    config = NestedConfig(agent_mode="tools")
    assert config == {"agent_mode": "tools"}

    # Full config is valid
    config = NestedConfig(
        agent_mode="code",
        max_iterations=10,
        max_depth=3,
    )
    # Use .get() for TypedDict total=False to avoid pyright warnings
    assert config.get("agent_mode") == "code"
    assert config.get("max_iterations") == 10
    assert config.get("max_depth") == 3


@pytest.mark.unit
def test_nested_config_agent_mode_values() -> None:
    """NestedConfig agent_mode accepts 'code' or 'tools'."""
    config_code: NestedConfig = {"agent_mode": "code"}
    config_tools: NestedConfig = {"agent_mode": "tools"}

    # Use .get() for TypedDict total=False
    assert config_code.get("agent_mode") == "code"
    assert config_tools.get("agent_mode") == "tools"


# ---------------------------------------------------------------------------
# Public API exports tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_protocols_exported_from_rlm_init() -> None:
    """Extension protocols are exported from rlm.__init__."""
    import rlm

    # Protocols
    assert hasattr(rlm, "StoppingPolicy")
    assert hasattr(rlm, "ContextCompressor")
    assert hasattr(rlm, "NestedCallPolicy")
    assert hasattr(rlm, "NestedConfig")

    # Default implementations
    assert hasattr(rlm, "DefaultStoppingPolicy")
    assert hasattr(rlm, "NoOpContextCompressor")
    assert hasattr(rlm, "SimpleNestedCallPolicy")


@pytest.mark.unit
def test_protocols_in_rlm_all() -> None:
    """Extension protocols are in rlm.__all__."""
    import rlm

    expected_exports = [
        "StoppingPolicy",
        "ContextCompressor",
        "NestedCallPolicy",
        "NestedConfig",
        "DefaultStoppingPolicy",
        "NoOpContextCompressor",
        "SimpleNestedCallPolicy",
    ]

    for name in expected_exports:
        assert name in rlm.__all__, f"{name} not in rlm.__all__"
