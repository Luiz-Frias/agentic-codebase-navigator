from __future__ import annotations

import io
import os
import signal
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlm.adapters.base import BaseEnvironmentAdapter
from rlm.domain.policies.timeouts import (
    DEFAULT_BROKER_CLIENT_TIMEOUT_S,
    DEFAULT_LOCAL_EXECUTE_TIMEOUT_S,
)
from rlm.infrastructure.comms.protocol import (
    request_completion,
    request_completions_batched,
)
from rlm.infrastructure.execution_namespace_policy import ExecutionNamespacePolicy
from rlm.infrastructure.logging import warn_cleanup_failure

if TYPE_CHECKING:
    from rlm.domain.models import BatchedLLMRequest, ChatCompletion, LLMRequest, ReplResult
    from rlm.domain.ports import BrokerPort
    from rlm.domain.types import ContextPayload, Prompt

# -----------------------------------------------------------------------------
# Process-wide safety guards
# -----------------------------------------------------------------------------

# `sys.stdout`/`sys.stderr` and `os.chdir()` are process-global. Guard them to avoid
# cross-thread corruption when multiple environments execute concurrently.
_PROCESS_EXEC_LOCK = threading.Lock()


@contextmanager
def _execution_timeout(timeout_s: float | None, /):
    """Best-effort execution timeout for runaway code (Local env only).

    Notes:
    - Uses SIGALRM when available and only in the main thread.
    - If unavailable, acts as a no-op (we still rely on broker timeouts for `llm_query`).

    """
    if timeout_s is None:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    prev_handler = signal.getsignal(signal.SIGALRM)
    prev_timer = signal.getitimer(signal.ITIMER_REAL)

    def _on_alarm(_signum: int, _frame):
        raise TimeoutError(f"Execution timed out after {timeout_s}s")

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        yield
    finally:
        # Restore any pre-existing timer/handler.
        signal.setitimer(signal.ITIMER_REAL, prev_timer[0], prev_timer[1])
        signal.signal(signal.SIGALRM, prev_handler)


class LocalEnvironmentAdapter(BaseEnvironmentAdapter):
    """Native Local environment adapter (Phase 05).

    Key semantics (legacy-compatible):
    - Persistent namespace across `execute_code` calls.
    - `context` variable set by `load_context(...)`.
    - `FINAL_VAR(name)` helper for final-answer extraction.
    - `llm_query()` and `llm_query_batched()` route through broker (or wire protocol)
      and record per-execution `ReplResult.llm_calls`.
    """

    environment_type: str = "local"

    def __init__(
        self,
        *,
        broker: BrokerPort | None = None,
        broker_address: tuple[str, int] | None = None,
        correlation_id: str | None = None,
        policy: ExecutionNamespacePolicy | None = None,
        context_payload: ContextPayload | None = None,
        setup_code: str | None = None,
        execute_timeout_s: float | None = DEFAULT_LOCAL_EXECUTE_TIMEOUT_S,
        broker_timeout_s: float = DEFAULT_BROKER_CLIENT_TIMEOUT_S,
        allowed_import_roots: set[str] | None = None,
    ) -> None:
        self._broker = broker
        self._broker_address = broker_address
        self._correlation_id = correlation_id

        if policy is None:
            if allowed_import_roots is None:
                policy = ExecutionNamespacePolicy()
            else:
                policy = ExecutionNamespacePolicy(
                    allowed_import_roots=frozenset(allowed_import_roots),
                )
        self._policy = policy

        self._execute_timeout_s = execute_timeout_s
        self._broker_timeout_s = broker_timeout_s

        self._tmp = tempfile.TemporaryDirectory(prefix="rlm_local_env_")
        self._session_dir = Path(self._tmp.name).resolve()

        # Per-execution pending call list (cleared on each execute_code).
        self._pending_llm_calls: list[ChatCompletion] = []

        # Persistent namespace (globals==locals).
        builtins_dict = self._policy.build_builtins(session_dir=self._session_dir)
        self._ns: dict[str, Any] = {
            "__builtins__": builtins_dict,
            "__name__": "__main__",
        }

        # Inject helpers + context variables.
        self._ns["FINAL_VAR"] = self._final_var
        self._ns["llm_query"] = self._llm_query
        self._ns["llm_query_batched"] = self._llm_query_batched
        self._ns["RLM_CORRELATION_ID"] = self._correlation_id
        self._ns["context"] = None

        if context_payload is not None:
            self.load_context(context_payload)

        if setup_code:
            # Best-effort: we still surface errors via stderr, like normal execution.
            self.execute_code(setup_code)

    @property
    def session_dir(self) -> Path:
        """Per-run session directory (temp cwd + allowed open root)."""
        return self._session_dir

    # ------------------------------------------------------------------
    # EnvironmentPort
    # ------------------------------------------------------------------

    def load_context(self, context_payload: ContextPayload, /) -> None:
        self._ns["context"] = context_payload

    def execute_code(self, code: str, /) -> ReplResult:
        start = time.perf_counter()
        self._pending_llm_calls = []

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # Guard stdout/stderr + cwd changes as they are process-global.
        with _PROCESS_EXEC_LOCK:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            old_cwd = os.getcwd()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                os.chdir(self._session_dir)
                with _execution_timeout(self._execute_timeout_s):
                    try:
                        exec(code, self._ns, self._ns)  # nosec B102 - core feature: controlled code execution
                    except Exception as exc:
                        # Keep formatting stable: no tracebacks, just type + message.
                        stderr_buf.write(f"\n{type(exc).__name__}: {exc}")
            finally:
                os.chdir(old_cwd)
                sys.stdout, sys.stderr = old_stdout, old_stderr

        end = time.perf_counter()
        return ReplResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            locals=self._snapshot_user_locals(),
            llm_calls=list(self._pending_llm_calls),
            execution_time=end - start,
        )

    def cleanup(self) -> None:
        # Best-effort idempotency.
        try:
            self._tmp.cleanup()
        except Exception as exc:  # nosec B110
            warn_cleanup_failure("LocalEnvironment.cleanup_tmp", exc)
        self._ns.clear()
        self._pending_llm_calls.clear()

    # ------------------------------------------------------------------
    # Helpers exposed to user code
    # ------------------------------------------------------------------

    def _final_var(self, variable_name: str) -> str:
        name = variable_name.strip().strip("\"'")
        if name in self._ns:
            try:
                return str(self._ns[name])
            except Exception as exc:
                return f"Error: Failed to stringify variable {name!r} - {exc}"
        return f"Error: Variable '{name}' not found"

    def _llm_query(
        self,
        prompt: Prompt,
        model: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Query the broker from within executed code.

        Legacy semantics:
        - returns the response string on success
        - returns "Error: ..." string on failure (does not raise)
        - records successful calls into `ReplResult.llm_calls`
        """
        cid = (
            correlation_id if (correlation_id is None or isinstance(correlation_id, str)) else None
        )
        if correlation_id is not None and cid is None:
            return "Error: Invalid correlation_id"

        try:
            cc = self._request_completion(prompt=prompt, model=model, correlation_id=cid)
        except Exception as exc:
            return f"Error: LM query failed - {exc}"
        self._pending_llm_calls.append(cc)
        return cc.response

    def _llm_query_batched(
        self,
        prompts: list[Prompt],
        model: str | None = None,
        correlation_id: str | None = None,
    ) -> list[str]:
        cid = (
            correlation_id if (correlation_id is None or isinstance(correlation_id, str)) else None
        )
        if correlation_id is not None and cid is None:
            return ["Error: Invalid correlation_id"] * (
                len(prompts) if isinstance(prompts, list) else 1
            )
        if not isinstance(prompts, list):
            return ["Error: Invalid prompts"]

        try:
            results, calls = self._request_completions_batched(
                prompts=prompts,
                model=model,
                correlation_id=cid,
            )
        except Exception as exc:
            return [f"Error: LM query failed - {exc}"] * len(prompts)

        self._pending_llm_calls.extend(calls)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _RESERVED_KEYS = {
        "__builtins__",
        "__name__",
        "FINAL_VAR",
        "llm_query",
        "llm_query_batched",
        "RLM_CORRELATION_ID",
        "context",
    }

    def _snapshot_user_locals(self) -> dict[str, Any]:
        # Mimic legacy behavior: avoid leaking internal keys and underscore-prefixed values.
        return {
            k: v
            for k, v in self._ns.items()
            if k not in self._RESERVED_KEYS and not k.startswith("_")
        }

    def _request_completion(
        self,
        *,
        prompt: Prompt,
        model: str | None,
        correlation_id: str | None,
    ) -> ChatCompletion:
        # Prefer in-process broker when available (deterministic, avoids sockets).
        if self._broker is not None:
            return self._broker.complete(LLMRequest(prompt=prompt, model=model))
        if self._broker_address is None:
            raise RuntimeError("No broker configured")
        return request_completion(
            self._broker_address,
            prompt,
            model=model,
            correlation_id=correlation_id or self._correlation_id,
            timeout_s=self._broker_timeout_s,
        )

    def _request_completions_batched(
        self,
        *,
        prompts: list[Prompt],
        model: str | None,
        correlation_id: str | None,
    ) -> tuple[list[str], list[ChatCompletion]]:
        if self._broker is not None:
            # Use the broker's batched interface (supports concurrency in the TCP broker).
            completions = self._broker.complete_batched(
                BatchedLLMRequest(prompts=prompts, model=model),
            )
            return [c.response for c in completions], list(completions)

        if self._broker_address is None:
            raise RuntimeError("No broker configured")

        results = request_completions_batched(
            self._broker_address,
            prompts,
            model=model,
            correlation_id=correlation_id or self._correlation_id,
            timeout_s=self._broker_timeout_s,
        )
        out: list[str] = []
        calls: list[ChatCompletion] = []
        for r in results:
            if r.error is not None:
                out.append(f"Error: {r.error}")
                continue
            assert r.chat_completion is not None
            calls.append(r.chat_completion)
            out.append(r.chat_completion.response)
        return out, calls
