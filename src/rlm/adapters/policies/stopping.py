"""
Default stopping policy implementation.

Provides basic max_iterations enforcement without custom stopping criteria.
External apps can implement StoppingPolicy to add EIG-gated stopping,
entropy-based termination, or other custom stopping logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rlm.domain.models import ChatCompletion


@dataclass
class DefaultStoppingPolicy:
    """
    Default stopping policy that enforces max_iterations limit.

    This is the default behavior when no custom StoppingPolicy is provided.
    It simply checks if the current iteration exceeds max_iterations.

    The on_iteration_complete callback is a no-op since no state tracking
    is needed for the simple max_iterations check.

    Example:
        policy = DefaultStoppingPolicy()
        context = {"iteration": 0, "max_iterations": 10}

        # First iteration
        assert not policy.should_stop(context)  # False, can continue

        # After max iterations
        context["iteration"] = 10
        assert policy.should_stop(context)  # True, should stop

    """

    def should_stop(self, context: dict[str, Any]) -> bool:
        """
        Return True if iteration >= max_iterations.

        Args:
            context: Orchestrator state dict with 'iteration' and 'max_iterations' keys.

        Returns:
            True if the iteration limit has been reached.

        """
        iteration = context.get("iteration", 0)
        max_iterations = context.get("max_iterations", 30)
        return iteration >= max_iterations

    def on_iteration_complete(
        self,
        _context: dict[str, Any],
        _result: ChatCompletion,
    ) -> None:
        """
        No-op callback for default policy.

        Custom implementations can use this to track state, update beliefs,
        compute metrics, etc. The default policy doesn't need any state.

        Args:
            _context: Mutable orchestrator state dict (unused in default).
            _result: The ChatCompletion from this iteration (unused in default).

        """
        # Default policy doesn't track any state
