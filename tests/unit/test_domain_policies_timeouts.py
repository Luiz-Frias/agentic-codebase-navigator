from __future__ import annotations

import pytest

from rlm.domain.policies import timeouts as t
from rlm.domain.policies.timeouts import (
    BrokerTimeouts,
    CancellationPolicy,
    DockerTimeouts,
    LocalTimeouts,
    TimeoutPolicy,
)


@pytest.mark.unit
def test_timeout_policy_defaults_match_module_constants() -> None:
    policy = TimeoutPolicy()

    assert policy.broker == BrokerTimeouts()
    assert policy.docker == DockerTimeouts()
    assert policy.local == LocalTimeouts()

    assert policy.broker.async_loop_start_timeout_s == t.DEFAULT_BROKER_ASYNC_LOOP_START_TIMEOUT_S
    assert policy.broker.thread_join_timeout_s == t.DEFAULT_BROKER_THREAD_JOIN_TIMEOUT_S
    assert policy.broker.client_timeout_s == t.DEFAULT_BROKER_CLIENT_TIMEOUT_S
    assert (
        policy.broker.batched_completion_timeout_s == t.DEFAULT_BROKER_BATCHED_COMPLETION_TIMEOUT_S
    )

    assert policy.docker.daemon_probe_timeout_s == t.DEFAULT_DOCKER_DAEMON_PROBE_TIMEOUT_S
    assert policy.docker.subprocess_timeout_s == t.DEFAULT_DOCKER_SUBPROCESS_TIMEOUT_S
    assert policy.docker.proxy_http_timeout_s == t.DEFAULT_DOCKER_PROXY_HTTP_TIMEOUT_S
    assert policy.docker.stop_grace_s == t.DEFAULT_DOCKER_STOP_GRACE_S
    assert (
        policy.docker.cleanup_subprocess_timeout_s == t.DEFAULT_DOCKER_CLEANUP_SUBPROCESS_TIMEOUT_S
    )
    assert policy.docker.thread_join_timeout_s == t.DEFAULT_DOCKER_THREAD_JOIN_TIMEOUT_S

    assert policy.local.execute_timeout_s == t.DEFAULT_LOCAL_EXECUTE_TIMEOUT_S
    assert policy.local.execute_timeout_cap_s == t.DEFAULT_LOCAL_EXECUTE_TIMEOUT_CAP_S
    assert t.DEFAULT_LOCAL_EXECUTE_TIMEOUT_CAP_S <= t.MAX_LOCAL_EXECUTE_TIMEOUT_CAP_S


@pytest.mark.unit
def test_timeout_and_cancellation_defaults_are_non_negative() -> None:
    policy = TimeoutPolicy()
    cancel = CancellationPolicy()

    assert policy.broker.async_loop_start_timeout_s >= 0
    assert policy.broker.thread_join_timeout_s >= 0
    assert policy.broker.client_timeout_s >= 0
    assert policy.broker.batched_completion_timeout_s >= 0

    assert policy.docker.daemon_probe_timeout_s >= 0
    assert policy.docker.subprocess_timeout_s >= 0
    assert policy.docker.proxy_http_timeout_s >= 0
    assert policy.docker.stop_grace_s >= 0
    assert policy.docker.cleanup_subprocess_timeout_s >= 0
    assert policy.docker.thread_join_timeout_s >= 0

    assert policy.local.execute_timeout_s >= 0
    assert policy.local.execute_timeout_cap_s >= 0

    assert cancel.grace_timeout_s >= 0
