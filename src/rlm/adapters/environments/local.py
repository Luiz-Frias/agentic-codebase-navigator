from __future__ import annotations

import io
import math
import os
import signal
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from rlm.adapters.base import BaseEnvironmentAdapter
from rlm.domain.models import BatchedLLMRequest, ChatCompletion, LLMRequest, ReplResult
from rlm.domain.models.serialization import serialize_value
from rlm.domain.policies.timeouts import (
    DEFAULT_BROKER_CLIENT_TIMEOUT_S,
    DEFAULT_LOCAL_EXECUTE_TIMEOUT_CAP_S,
    DEFAULT_LOCAL_EXECUTE_TIMEOUT_S,
    MAX_LOCAL_EXECUTE_TIMEOUT_CAP_S,
)
from rlm.infrastructure.comms.protocol import (
    request_completion,
    request_completions_batched,
)
from rlm.infrastructure.execution_namespace_policy import ExecutionNamespacePolicy
from rlm.infrastructure.logging import warn_cleanup_failure

if TYPE_CHECKING:
    from rlm.domain.ports import BrokerPort
    from rlm.domain.types import ContextPayload, Prompt


class _WorkerConnection(Protocol):
    def send(self, obj: object) -> None: ...

    def recv(self) -> object: ...

    def poll(self, timeout: float | None = None) -> bool: ...

    def close(self) -> None: ...


class _WorkerProcess(Protocol):
    @property
    def pid(self) -> int | None: ...

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


# -----------------------------------------------------------------------------
# Process-wide safety guards
# -----------------------------------------------------------------------------

# `sys.stdout`/`sys.stderr` and `os.chdir()` are process-global. Guard them to avoid
# cross-thread corruption when multiple environments execute concurrently.
_PROCESS_EXEC_LOCK = threading.Lock()

_RESERVED_KEYS: set[str] = {
    "__builtins__",
    "__name__",
    "FINAL_VAR",
    "llm_query",
    "llm_query_batched",
    "RLM_CORRELATION_ID",
    "context",
}


@contextmanager
def _execution_timeout(timeout_s: float | None, /):
    """
    Best-effort execution timeout for runaway code (Local env only).

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

    def _on_alarm(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"Execution timed out after {timeout_s}s")

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        yield
    finally:
        # Restore any pre-existing timer/handler.
        signal.setitimer(signal.ITIMER_REAL, prev_timer[0], prev_timer[1])
        signal.signal(signal.SIGALRM, prev_handler)


def _normalize_timeout(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number > 0")
    value_f = float(value)
    if not math.isfinite(value_f) or value_f <= 0:
        raise ValueError(f"{name} must be a number > 0")
    return value_f


def _snapshot_user_locals(ns: dict[str, Any]) -> dict[str, Any]:
    # Mimic legacy behavior: avoid leaking internal keys and underscore-prefixed values.
    return {k: v for k, v in ns.items() if k not in _RESERVED_KEYS and not k.startswith("_")}


def _local_worker_main(
    conn: _WorkerConnection,
    *,
    session_dir: str,
    allowed_import_roots: set[str],
    correlation_id: str | None,
    execute_timeout_s: float,
) -> None:
    # Isolate child process group so we can kill all descendants on timeout.
    if hasattr(os, "setsid"):
        with suppress(Exception):
            os.setsid()

    session_path = Path(session_dir).resolve()
    policy = ExecutionNamespacePolicy(allowed_import_roots=frozenset(allowed_import_roots))
    builtins_dict = policy.build_builtins(session_dir=session_path)
    ns: dict[str, Any] = {
        "__builtins__": builtins_dict,
        "__name__": "__main__",
    }

    request_lock = threading.Lock()

    def _final_var(variable_name: str) -> str:
        name = variable_name.strip().strip("\"'")
        if name in ns:
            try:
                return str(ns[name])
            except Exception as exc:
                return f"Error: Failed to stringify variable {name!r} - {exc}"
        return f"Error: Variable '{name}' not found"

    def _send_request(payload: dict[str, object]) -> dict[str, object]:
        with request_lock:
            conn.send(payload)
            resp = conn.recv()
        return resp if isinstance(resp, dict) else {"type": "llm_query_result", "ok": False}

    def _llm_query(
        prompt: Prompt,
        model: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        cid = (
            correlation_id if (correlation_id is None or isinstance(correlation_id, str)) else None
        )
        if correlation_id is not None and cid is None:
            return "Error: Invalid correlation_id"

        resp = _send_request(
            {
                "type": "llm_query",
                "prompt": prompt,
                "model": model,
                "correlation_id": cid,
            },
        )
        if resp.get("type") != "llm_query_result":
            return "Error: LM query failed - Invalid broker response"
        if resp.get("ok"):
            return str(resp.get("response", ""))
        return f"Error: LM query failed - {resp.get('error', 'Unknown error')}"

    def _llm_query_batched(
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

        resp = _send_request(
            {
                "type": "llm_query_batched",
                "prompts": prompts,
                "model": model,
                "correlation_id": cid,
            },
        )
        if resp.get("type") != "llm_query_batched_result":
            return ["Error: LM query failed - Invalid broker response"] * len(prompts)
        if resp.get("ok"):
            out = resp.get("responses", [])
            return [str(item) for item in out] if isinstance(out, list) else ["Error: Invalid"]
        return [f"Error: LM query failed - {resp.get('error', 'Unknown error')}"] * len(prompts)

    ns.update(
        {
            "FINAL_VAR": _final_var,
            "llm_query": _llm_query,
            "llm_query_batched": _llm_query_batched,
            "RLM_CORRELATION_ID": correlation_id,
            "context": None,
        },
    )

    def _execute_code(code: str) -> dict[str, object]:
        start = time.perf_counter()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # Guard stdout/stderr + cwd changes as they are process-global.
        with _PROCESS_EXEC_LOCK:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            old_cwd = Path.cwd()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                os.chdir(session_path)
                with _execution_timeout(execute_timeout_s):
                    try:
                        exec(code, ns, ns)  # nosec B102 - core feature: controlled code execution
                    except Exception as exc:
                        # Keep formatting stable: no tracebacks, just type + message.
                        stderr_buf.write(f"\n{type(exc).__name__}: {exc}")
            finally:
                os.chdir(old_cwd)
                sys.stdout, sys.stderr = old_stdout, old_stderr

        end = time.perf_counter()
        locals_snapshot = _snapshot_user_locals(ns)
        serialized_locals = {k: serialize_value(v) for k, v in locals_snapshot.items()}
        return {
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "locals": serialized_locals,
            "execution_time": end - start,
        }

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break

        if not isinstance(msg, dict):
            continue

        msg_type = msg.get("type")
        if msg_type == "execute":
            code = msg.get("code", "")
            result = _execute_code(str(code))
            conn.send({"type": "execute_result", **result})
        elif msg_type == "load_context":
            ns["context"] = msg.get("context")
        elif msg_type == "final_var":
            name = msg.get("name", "")
            conn.send({"type": "final_var_result", "value": _final_var(str(name))})
        elif msg_type == "shutdown":
            break

    conn.close()


class LocalEnvironmentAdapter(BaseEnvironmentAdapter):
    """
    Native Local environment adapter (Phase 05).

    Key semantics (legacy-compatible):
    - Persistent namespace across `execute_code` calls.
    - `context` variable set by `load_context(...)`.
    - `FINAL_VAR(name)` helper for final-answer extraction.
    - `llm_query()` and `llm_query_batched()` route through broker (or wire protocol)
      and record per-execution `ReplResult.llm_calls`.
    """

    environment_type: str = "local"

    _RESERVED_KEYS: ClassVar[set[str]] = _RESERVED_KEYS

    def __init__(
        self,
        *,
        broker: BrokerPort | None = None,
        broker_address: tuple[str, int] | None = None,
        correlation_id: str | None = None,
        policy: ExecutionNamespacePolicy | None = None,
        context_payload: ContextPayload | None = None,
        setup_code: str | None = None,
        execute_timeout_s: float = DEFAULT_LOCAL_EXECUTE_TIMEOUT_S,
        execute_timeout_cap_s: float = DEFAULT_LOCAL_EXECUTE_TIMEOUT_CAP_S,
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

        self._execute_timeout_cap_s = _normalize_timeout(
            execute_timeout_cap_s,
            name="execute_timeout_cap_s",
        )
        if self._execute_timeout_cap_s > MAX_LOCAL_EXECUTE_TIMEOUT_CAP_S:
            raise ValueError(
                f"execute_timeout_cap_s must be <= {MAX_LOCAL_EXECUTE_TIMEOUT_CAP_S}",
            )

        self._execute_timeout_s = _normalize_timeout(
            execute_timeout_s,
            name="execute_timeout_s",
        )
        if self._execute_timeout_s > self._execute_timeout_cap_s:
            raise ValueError(
                "execute_timeout_s must be <= execute_timeout_cap_s "
                f"({self._execute_timeout_cap_s})",
            )

        self._broker_timeout_s = broker_timeout_s

        self._tmp = tempfile.TemporaryDirectory(prefix="rlm_local_env_")
        self._session_dir = Path(self._tmp.name).resolve()

        self._pending_llm_calls: list[ChatCompletion] = []
        self._context_payload = context_payload
        self._setup_code = setup_code

        self._exec_lock = threading.Lock()
        self._mp_ctx = get_context("spawn")
        self._worker_conn: _WorkerConnection | None = None
        self._worker_process: _WorkerProcess | None = None

        if context_payload is not None or setup_code:
            self._ensure_worker()

    @property
    def session_dir(self) -> Path:
        """Per-run session directory (temp cwd + allowed open root)."""
        return self._session_dir

    # ------------------------------------------------------------------
    # EnvironmentPort
    # ------------------------------------------------------------------

    def load_context(self, context_payload: ContextPayload, /) -> None:
        self._context_payload = context_payload
        self._ensure_worker()
        if self._worker_conn is not None:
            try:
                self._worker_conn.send({"type": "load_context", "context": context_payload})
            except Exception:
                self._restart_worker()
                if self._worker_conn is not None:
                    self._worker_conn.send({"type": "load_context", "context": context_payload})

    def execute_code(self, code: str, /) -> ReplResult:
        with self._exec_lock:
            self._pending_llm_calls = []
            self._ensure_worker()
            return self._execute_in_worker(code)

    def cleanup(self) -> None:
        # Best-effort idempotency.
        self._shutdown_worker()
        try:
            self._tmp.cleanup()
        except Exception as exc:  # nosec B110
            warn_cleanup_failure("LocalEnvironment.cleanup_tmp", exc)
        self._pending_llm_calls.clear()

    # ------------------------------------------------------------------
    # Helpers exposed to user code
    # ------------------------------------------------------------------

    def _final_var(self, variable_name: str) -> str:
        with self._exec_lock:
            self._ensure_worker()
            if self._worker_conn is None:
                return "Error: Local worker unavailable"
            try:
                self._worker_conn.send({"type": "final_var", "name": variable_name})
            except (BrokenPipeError, OSError):
                self._restart_worker()
                if self._worker_conn is None:
                    return "Error: Local worker unavailable"
                self._worker_conn.send({"type": "final_var", "name": variable_name})
            deadline = time.monotonic() + self._execute_timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._restart_worker()
                    return f"Error: Execution timed out after {self._execute_timeout_s}s"
                if self._worker_conn.poll(timeout=min(remaining, 0.1)):
                    try:
                        msg = self._worker_conn.recv()
                    except EOFError:
                        self._restart_worker()
                        return "Error: Local worker unavailable"
                    if isinstance(msg, dict):
                        msg_type = msg.get("type")
                        if msg_type == "final_var_result":
                            return str(msg.get("value", ""))
                        if msg_type == "llm_query":
                            self._worker_conn.send(self._handle_llm_query(msg))
                        elif msg_type == "llm_query_batched":
                            self._worker_conn.send(self._handle_llm_query_batched(msg))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._worker_process is not None and self._worker_process.is_alive():
            return
        self._start_worker()

    def _start_worker(self) -> None:
        self._shutdown_worker()
        parent_conn, child_conn = self._mp_ctx.Pipe(duplex=True)
        proc = self._mp_ctx.Process(
            target=_local_worker_main,
            kwargs={
                "conn": child_conn,
                "session_dir": str(self._session_dir),
                "allowed_import_roots": set(self._policy.allowed_import_roots),
                "correlation_id": self._correlation_id,
                "execute_timeout_s": self._execute_timeout_s,
            },
        )
        proc.start()
        child_conn.close()

        self._worker_conn = parent_conn
        self._worker_process = proc

        if self._context_payload is not None and self._worker_conn is not None:
            self._worker_conn.send({"type": "load_context", "context": self._context_payload})
        if self._setup_code:
            self._execute_in_worker(self._setup_code, restart_on_failure=False)

    def _restart_worker(self) -> None:
        self._shutdown_worker(kill=True)
        self._start_worker()

    def _shutdown_worker(self, *, kill: bool = False) -> None:
        conn = self._worker_conn
        proc = self._worker_process
        self._worker_conn = None
        self._worker_process = None

        if conn is not None:
            with suppress(Exception):
                conn.send({"type": "shutdown"})
            with suppress(Exception):
                conn.close()

        if proc is not None:
            if proc.is_alive():
                if kill:
                    self._kill_process(proc)
                else:
                    proc.join(timeout=1)
            if proc.is_alive():
                self._kill_process(proc)

    def _kill_process(self, proc: _WorkerProcess) -> None:
        pid = getattr(proc, "pid", None)
        if pid is None:
            return
        pgid: int | None = None
        if hasattr(os, "getpgid") and hasattr(os, "killpg") and hasattr(signal, "SIGKILL"):
            with suppress(Exception):
                pgid = os.getpgid(pid)
            if pgid == pid:
                with suppress(Exception):
                    os.killpg(pid, signal.SIGKILL)
        with suppress(Exception):
            proc.terminate()
        proc.join(timeout=1)
        if proc.is_alive() and hasattr(proc, "kill"):
            with suppress(Exception):
                proc.kill()
            proc.join(timeout=1)

    def _execute_in_worker(self, code: str, *, restart_on_failure: bool = True) -> ReplResult:
        if self._worker_conn is None:
            raise RuntimeError("Local environment worker not available")
        try:
            self._worker_conn.send({"type": "execute", "code": code})
        except (BrokenPipeError, OSError):
            self._restart_worker()
            if self._worker_conn is None:
                raise RuntimeError("Local environment worker not available") from None
            self._worker_conn.send({"type": "execute", "code": code})

        start = time.monotonic()
        deadline = start + self._execute_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if restart_on_failure:
                    self._restart_worker()
                else:
                    self._shutdown_worker(kill=True)
                return ReplResult(
                    stdout="",
                    stderr=f"\nTimeoutError: Execution timed out after {self._execute_timeout_s}s",
                    locals={},
                    llm_calls=list(self._pending_llm_calls),
                    execution_time=time.monotonic() - start,
                )
            if self._worker_conn.poll(timeout=min(remaining, 0.1)):
                try:
                    msg = self._worker_conn.recv()
                except EOFError:
                    if restart_on_failure:
                        self._restart_worker()
                    else:
                        self._shutdown_worker(kill=True)
                    return ReplResult(
                        stdout="",
                        stderr="\nRuntimeError: Local worker crashed",
                        locals={},
                        llm_calls=list(self._pending_llm_calls),
                        execution_time=time.monotonic() - start,
                    )
                if not isinstance(msg, dict):
                    continue
                msg_type = msg.get("type")
                if msg_type == "execute_result":
                    return ReplResult(
                        stdout=str(msg.get("stdout", "")),
                        stderr=str(msg.get("stderr", "")),
                        locals=msg.get("locals", {}) if isinstance(msg.get("locals"), dict) else {},
                        llm_calls=list(self._pending_llm_calls),
                        execution_time=float(msg.get("execution_time", 0.0)),
                    )
                if msg_type == "llm_query":
                    self._worker_conn.send(self._handle_llm_query(msg))
                elif msg_type == "llm_query_batched":
                    self._worker_conn.send(self._handle_llm_query_batched(msg))

    def _handle_llm_query(self, msg: dict[str, object]) -> dict[str, object]:
        prompt = msg.get("prompt")
        model = msg.get("model")
        correlation_id = msg.get("correlation_id")
        try:
            cc = self._request_completion(
                prompt=prompt,  # type: ignore[arg-type]
                model=model if isinstance(model, str) or model is None else None,
                correlation_id=correlation_id if isinstance(correlation_id, str) else None,
            )
        except Exception as exc:
            return {"type": "llm_query_result", "ok": False, "error": str(exc)}
        self._pending_llm_calls.append(cc)
        return {"type": "llm_query_result", "ok": True, "response": cc.response}

    def _handle_llm_query_batched(self, msg: dict[str, object]) -> dict[str, object]:
        prompts = msg.get("prompts")
        model = msg.get("model")
        correlation_id = msg.get("correlation_id")
        if not isinstance(prompts, list):
            return {"type": "llm_query_batched_result", "ok": False, "error": "Invalid prompts"}
        try:
            results, calls = self._request_completions_batched(
                prompts=prompts,  # type: ignore[arg-type]
                model=model if isinstance(model, str) or model is None else None,
                correlation_id=correlation_id if isinstance(correlation_id, str) else None,
            )
        except Exception as exc:
            return {"type": "llm_query_batched_result", "ok": False, "error": str(exc)}
        self._pending_llm_calls.extend(calls)
        return {"type": "llm_query_batched_result", "ok": True, "responses": results}

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
            cc = r.chat_completion
            if r.error is not None or cc is None:
                out.append(f"Error: {r.error or 'No completion returned'}")
                continue
            calls.append(cc)
            out.append(cc.response)
        return out, calls
