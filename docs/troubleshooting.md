# Troubleshooting (timeouts, hangs, cleanup)

## Quick symptoms → likely cause

- **Completion hangs during `llm_query_batched(...)`**
  - Likely: a batched subcall is stuck (provider call never returns) or broker tasks aren’t making forward progress.
  - Mitigation: reduce broker batched timeout and enable cancellation grace (see **Broker timeouts** below).

- **Docker env hangs / never returns from code execution**
  - Likely: `docker exec` is blocked/stuck.
  - Mitigation: lower `DockerREPL(subprocess_timeout_s=...)` (see **Docker exec timeout** below). The timeout path triggers cleanup.

- **Cleanup flakiness / orphaned docker containers**
  - Likely: partial initialization or transient docker command failures.
  - Mitigation: rerun with Docker healthy; cleanup is idempotent and best-effort (see `tests/test_env_cleanup_hardening.py`).

## Correlation IDs (trace a single run end-to-end)

Every `run_completion()` generates a **correlation ID** and propagates it through:

- **Run metadata**: logged via `LoggerPort.log_metadata(...)`
- **Iterations**: `Iteration.correlation_id`
- **REPL results**: `ReplResult.correlation_id`
- **Docker env**: `RLM_CORRELATION_ID` is injected into `docker exec` (host → container)

Use this ID to stitch together logs and to debug “where it’s stuck”.

## Broker timeouts (batched subcalls)

The broker’s batched path has:

- **Batch completion timeout**: how long to wait for all async subcalls to finish
- **Cancellation grace**: after timeout, how long to wait for tasks to cancel cleanly

These are configured via:

- `rlm.domain.policies.timeouts.BrokerTimeouts`
- `rlm.domain.policies.timeouts.CancellationPolicy`

## Protocol client timeouts (TCP wire requests)

When using the TCP protocol client helpers (e.g. `request_completions_batched(...)`), you can pass:

- `timeout_s=...`: socket-level request/response timeout
- `max_message_bytes=...`: upper bound to protect against runaway payloads

## Docker exec timeout

`DockerREPL.execute_code(...)` uses a subprocess timeout for `docker exec`. On timeout it:

- returns a **safe** `TimeoutExpired: docker exec exceeded ...` error in `stderr`
- triggers `cleanup()` (stop container + stop proxy)

Configure via `DockerREPL(subprocess_timeout_s=...)`.

## Local execution timeout (best-effort)

`LocalREPL.execute_code(...)` can enforce a best-effort timeout via `SIGALRM`:

- only works on platforms where `SIGALRM` is available
- requires running on the **main thread**

Configure via `LocalREPL(execute_timeout_s=...)`.
