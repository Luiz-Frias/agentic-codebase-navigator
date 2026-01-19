"""
Default nested call policy implementation.

Provides simple LLM call behavior without sub-orchestrator spawning.
External apps can implement NestedCallPolicy to enable recursive
orchestration for complex multi-agent scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

from rlm.domain.agent_ports import NestedConfig


@dataclass
class SimpleNestedCallPolicy:
    """
    Simple nested call policy that always uses direct LLM calls.

    This is the default behavior when no custom NestedCallPolicy is provided.
    Nested llm_query() calls from generated code are handled as simple
    LLM completions without spawning sub-orchestrators.

    For recursive orchestration, implement a custom policy that:
    - Determines when nested calls should spawn orchestrators (e.g., depth-based)
    - Configures nested orchestrators with appropriate tools and limits
    - Manages context budget across the orchestrator tree

    Example:
        policy = SimpleNestedCallPolicy()

        # Always returns False - no orchestration
        assert not policy.should_orchestrate("What is 2+2?", depth=0)
        assert not policy.should_orchestrate("Complex task", depth=1)

        # Config is empty since orchestration is never triggered
        config = policy.get_nested_config()
        assert config == {}

    """

    def should_orchestrate(self, _prompt: str, _depth: int) -> bool:
        """
        Always returns False - nested calls use simple LLM completion.

        Args:
            _prompt: The prompt being passed (unused in default).
            _depth: Current recursion depth (unused in default).

        Returns:
            False - nested calls should not spawn orchestrators.

        """
        # Default: never spawn nested orchestrators
        return False

    def get_nested_config(self) -> NestedConfig:
        """
        Return empty config since orchestration is never triggered.

        This method is only called when should_orchestrate() returns True,
        which never happens for the default policy. Returns empty dict
        for type safety.

        Returns:
            Empty NestedConfig dict.

        """
        # Return empty config - this should never be called
        return NestedConfig()
