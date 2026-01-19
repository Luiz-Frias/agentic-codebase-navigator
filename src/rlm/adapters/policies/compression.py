"""
Default context compressor implementation.

Provides passthrough behavior without any compression.
External apps can implement ContextCompressor to add summarization,
truncation, or other compression strategies for nested call returns.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NoOpContextCompressor:
    """
    No-operation context compressor that passes through unchanged.

    This is the default behavior when no custom ContextCompressor is provided.
    Nested call results are returned to the parent orchestrator as-is.

    For context-budget-aware applications, implement a custom compressor that:
    - Summarizes long results using an LLM
    - Truncates to a maximum length
    - Extracts key information based on the task

    Example:
        compressor = NoOpContextCompressor()
        result = "This is a very long result from a nested call..."

        # Passthrough - result unchanged
        compressed = compressor.compress(result)
        assert compressed == result

        # max_tokens hint is ignored
        compressed = compressor.compress(result, max_tokens=100)
        assert compressed == result

    """

    def compress(self, result: str, _max_tokens: int | None = None) -> str:
        """
        Return result unchanged (passthrough).

        Args:
            result: The full result string from a nested orchestrator.
            _max_tokens: Optional token budget hint (ignored by default).

        Returns:
            The result string unchanged.

        """
        # Default policy: passthrough without compression
        return result
