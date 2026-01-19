"""
Infrastructure logging for RLM internals.

This module provides observability into infrastructure-level events (cleanup failures,
resource lifecycle issues) separate from domain-level logging (LoggerPort for iterations,
metadata).

Design Principles:
- Zero-configuration startup with sensible defaults
- Async-safe with enqueue=True (non-blocking)
- Exception context preservation for debugging
- Minimal overhead when not actively logging

Usage:
    from rlm.infrastructure.logging import warn_cleanup_failure

    try:
        resource.cleanup()
    except Exception as exc:
        warn_cleanup_failure("ResourceName", exc)
"""

from __future__ import annotations

import sys

import loguru
from loguru import logger

# ---------------------------------------------------------------------------
# Logger Configuration
# ---------------------------------------------------------------------------
# Remove default handler and configure with sensible defaults for infrastructure
# logging. This configuration:
# - Only emits WARNING and above (cleanup failures are warnings, not errors)
# - Uses structured format for log aggregation compatibility
# - Enables async-safe logging with enqueue=True
# - Includes exception context when available

# Module-level flag to track if we've configured the logger
_configured = False


def _configure_logger() -> None:
    """Configure loguru with infrastructure logging defaults (idempotent)."""
    global _configured
    if _configured:
        return

    # Remove default handler to avoid duplicate output
    logger.remove()

    # Add infrastructure logger with:
    # - WARNING level (cleanup failures are warnings, not errors)
    # - Structured format for log aggregation
    # - Async-safe (enqueue=True)
    # - Backtrace for debugging (only in non-production)
    logger.add(
        sys.stderr,
        level="WARNING",
        format=(
            "<dim>{time:YYYY-MM-DD HH:mm:ss.SSS}</dim> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        enqueue=True,  # Async-safe: non-blocking log writes
        backtrace=False,  # Disable full backtrace in production
        diagnose=False,  # Disable variable inspection in production
    )

    _configured = True


# Configure on module import (zero-configuration startup)
_configure_logger()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def warn_cleanup_failure(
    component: str,
    exc: BaseException,
    *,
    context: dict[str, object] | None = None,
) -> None:
    """
    Log a cleanup boundary failure with full exception context.

    This function is designed for best-effort cleanup code where exceptions
    must be suppressed but visibility is still needed for debugging and
    operational awareness.

    Args:
        component: Name of the component that failed to clean up
                   (e.g., "DockerEnvironment", "TcpBroker")
        exc: The exception that occurred during cleanup
        context: Optional additional context to include in the log

    Example:
        try:
            self._container.stop()
        except Exception as exc:
            warn_cleanup_failure("DockerEnvironment.cleanup", exc)

    """
    bound_logger = logger.bind(
        component=component,
        exc_type=type(exc).__name__,
        **(context or {}),
    )

    # Use opt(exception=True) to capture full stack trace
    bound_logger.opt(exception=exc).warning(
        "Cleanup failed in {component}: {exc_type}",
        component=component,
        exc_type=type(exc).__name__,
    )


def get_infrastructure_logger() -> loguru.Logger:
    """
    Get the infrastructure logger for advanced use cases.

    Most code should use warn_cleanup_failure() directly. This function
    is provided for cases where more control is needed.

    Returns:
        The configured loguru logger instance

    """
    return logger
