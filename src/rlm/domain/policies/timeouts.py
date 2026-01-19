from __future__ import annotations

from dataclasses import dataclass, field

# -----------------------------------------------------------------------------
# Default timeout values (centralized; referenced by policies + adapters)
# -----------------------------------------------------------------------------

# Broker (TCP server + async loop helper + client request/response)
DEFAULT_BROKER_ASYNC_LOOP_START_TIMEOUT_S: float = 2.0
DEFAULT_BROKER_THREAD_JOIN_TIMEOUT_S: float = 2.0
DEFAULT_BROKER_CLIENT_TIMEOUT_S: float = 300.0
DEFAULT_BROKER_BATCHED_COMPLETION_TIMEOUT_S: float = DEFAULT_BROKER_CLIENT_TIMEOUT_S

# Docker (daemon probe + subprocess operations + in-container proxy calls)
DEFAULT_DOCKER_DAEMON_PROBE_TIMEOUT_S: float = 2.0
DEFAULT_DOCKER_SUBPROCESS_TIMEOUT_S: float = 300.0
DEFAULT_DOCKER_PROXY_HTTP_TIMEOUT_S: float = 300.0
DEFAULT_DOCKER_STOP_GRACE_S: int = 2
DEFAULT_DOCKER_CLEANUP_SUBPROCESS_TIMEOUT_S: float = 5.0
DEFAULT_DOCKER_THREAD_JOIN_TIMEOUT_S: float = 2.0

# Local execution (subprocess watchdog + capped max timeout)
DEFAULT_LOCAL_EXECUTE_TIMEOUT_S: float = DEFAULT_BROKER_CLIENT_TIMEOUT_S
DEFAULT_LOCAL_EXECUTE_TIMEOUT_CAP_S: float = 3600.0
MAX_LOCAL_EXECUTE_TIMEOUT_CAP_S: float = 21600.0

# Cancellation behavior (grace period for cooperative cancellation)
DEFAULT_CANCELLATION_GRACE_TIMEOUT_S: float = 2.0


@dataclass(frozen=True, slots=True)
class BrokerTimeouts:
    """Timeouts for broker lifecycle + client request/response."""

    async_loop_start_timeout_s: float = DEFAULT_BROKER_ASYNC_LOOP_START_TIMEOUT_S
    thread_join_timeout_s: float = DEFAULT_BROKER_THREAD_JOIN_TIMEOUT_S
    client_timeout_s: float = DEFAULT_BROKER_CLIENT_TIMEOUT_S
    batched_completion_timeout_s: float = DEFAULT_BROKER_BATCHED_COMPLETION_TIMEOUT_S


@dataclass(frozen=True, slots=True)
class DockerTimeouts:
    """Timeouts for Docker-based environments and proxy calls."""

    daemon_probe_timeout_s: float = DEFAULT_DOCKER_DAEMON_PROBE_TIMEOUT_S
    subprocess_timeout_s: float = DEFAULT_DOCKER_SUBPROCESS_TIMEOUT_S
    proxy_http_timeout_s: float = DEFAULT_DOCKER_PROXY_HTTP_TIMEOUT_S
    stop_grace_s: int = DEFAULT_DOCKER_STOP_GRACE_S
    cleanup_subprocess_timeout_s: float = DEFAULT_DOCKER_CLEANUP_SUBPROCESS_TIMEOUT_S
    thread_join_timeout_s: float = DEFAULT_DOCKER_THREAD_JOIN_TIMEOUT_S


@dataclass(frozen=True, slots=True)
class LocalTimeouts:
    """Timeouts for local environments."""

    execute_timeout_s: float = DEFAULT_LOCAL_EXECUTE_TIMEOUT_S
    execute_timeout_cap_s: float = DEFAULT_LOCAL_EXECUTE_TIMEOUT_CAP_S


@dataclass(frozen=True, slots=True)
class CancellationPolicy:
    """Cancellation-related runtime policy knobs."""

    grace_timeout_s: float = DEFAULT_CANCELLATION_GRACE_TIMEOUT_S


@dataclass(frozen=True, slots=True)
class TimeoutPolicy:
    """Composite runtime timeout policy."""

    broker: BrokerTimeouts = field(default_factory=BrokerTimeouts)
    docker: DockerTimeouts = field(default_factory=DockerTimeouts)
    local: LocalTimeouts = field(default_factory=LocalTimeouts)
