from __future__ import annotations

import base64
import json
import os
import subprocess  # nosec B404 - required for Docker CLI integration
import tempfile
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any, cast

from rlm.adapters.base import BaseEnvironmentAdapter
from rlm.domain.models import ChatCompletion, LLMRequest, ReplResult
from rlm.domain.models.result import Err
from rlm.domain.models.validation import Validator
from rlm.domain.policies.timeouts import (
    DEFAULT_DOCKER_CLEANUP_SUBPROCESS_TIMEOUT_S,
    DEFAULT_DOCKER_PROXY_HTTP_TIMEOUT_S,
    DEFAULT_DOCKER_STOP_GRACE_S,
    DEFAULT_DOCKER_SUBPROCESS_TIMEOUT_S,
    DEFAULT_DOCKER_THREAD_JOIN_TIMEOUT_S,
)

if TYPE_CHECKING:
    from rlm.domain.ports import BrokerPort
    from rlm.domain.types import ContextPayload, Prompt
from rlm.infrastructure.comms.protocol import (
    request_completion,
    request_completions_batched,
)
from rlm.infrastructure.logging import warn_cleanup_failure

# =============================================================================
# Request Validators (composable validation for HTTP handler)
# =============================================================================

# Prompt must be str, dict, or list
_prompt_validator: Validator[object] = Validator[object]().satisfies(
    lambda x: isinstance(x, (str, dict, list)),
    "Invalid prompt",
)

# Prompts must be a list
_prompts_validator: Validator[object] = Validator[object]().is_type(list, "Invalid prompts")

# Model must be None or string
_model_validator: Validator[object] = Validator[object]().satisfies(
    lambda x: x is None or isinstance(x, str),
    "Invalid model",
)

# Correlation ID must be None or string
_correlation_id_validator: Validator[object] = Validator[object]().satisfies(
    lambda x: x is None or isinstance(x, str),
    "Invalid correlation_id",
)


def _validate_request_fields(
    body: dict[str, Any],
    required_field: str,
    required_validator: Validator[object],
) -> dict[str, str] | None:
    """
    Validate common HTTP request fields.

    Returns error dict if validation fails, None if all valid.
    """
    # Validate required field
    result = required_validator.validate_to_result(body.get(required_field))
    if isinstance(result, Err):
        return {"error": str(result.error)}

    # Validate optional model field
    result = _model_validator.validate_to_result(body.get("model"))
    if isinstance(result, Err):
        return {"error": str(result.error)}

    # Validate optional correlation_id field
    result = _correlation_id_validator.validate_to_result(body.get("correlation_id"))
    if isinstance(result, Err):
        return {"error": str(result.error)}

    return None


def _use_host_network() -> bool:
    """
    Check if Docker should use host networking mode.

    When RLM_DOCKER_USE_HOST_NETWORK=1, the container shares the host's network
    namespace. This is useful in CI environments (like GitHub Actions) where
    `host.docker.internal` doesn't reliably resolve.

    With host networking:
    - Container uses `localhost:PORT` instead of `host.docker.internal:PORT`
    - No `--add-host` flag needed
    - `--network=host` is added to docker run
    """
    raw = (os.environ.get("RLM_DOCKER_USE_HOST_NETWORK") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


class DockerLLMProxyHandler(BaseHTTPRequestHandler):
    """
    Host-side HTTP proxy for `llm_query()` calls issued from within a container.

    The in-container execution script calls:
    - POST /llm_query
    - POST /llm_query_batched

    The handler:
    - routes requests to an injected BrokerPort (preferred) OR broker TCP address
    - records successful ChatCompletion objects into `pending_calls`
    """

    # Injected at runtime by DockerEnvironmentAdapter via a derived handler type.
    broker: BrokerPort | None = None
    broker_address: tuple[str, int] | None = None
    pending_calls: list[ChatCompletion] = []
    lock: threading.Lock = threading.Lock()
    timeout_s: float = DEFAULT_DOCKER_PROXY_HTTP_TIMEOUT_S

    # Silence noisy default logging - must match BaseHTTPRequestHandler signature
    def log_message(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        format: str,  # noqa: A002  # Must shadow builtin to match base class
        *args: object,
    ) -> None:
        pass

    def do_POST(self) -> None:
        try:
            raw_len = self.headers.get("Content-Length")
            if raw_len is None:
                self._respond(400, {"error": "Missing Content-Length"})
                return
            try:
                content_length = int(raw_len)
            except ValueError:
                self._respond(400, {"error": "Invalid Content-Length"})
                return
            if content_length <= 0:
                self._respond(400, {"error": "Missing request body"})
                return
            body_raw: Any = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError:
            self._respond(400, {"error": "Invalid JSON payload"})
            return
        except Exception as exc:
            self._respond(400, {"error": str(exc)})
            return

        if not isinstance(body_raw, dict):
            self._respond(400, {"error": "Request body must be a JSON object"})
            return

        body: dict[str, Any] = cast("dict[str, Any]", body_raw)

        if self.path == "/llm_query":
            result = self._handle_single(body)
        elif self.path == "/llm_query_batched":
            result = self._handle_batched(body)
        else:
            self._respond(404, {"error": "Not found"})
            return

        self._respond(200, result)

    def _respond(self, status: int, data: dict[str, Any]) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    # ------------------------------------------------------------------
    # Request handlers
    # ------------------------------------------------------------------

    def _handle_single(self, body: dict[str, Any]) -> dict[str, Any]:
        # Validate request fields using composable validators
        validation_error = _validate_request_fields(body, "prompt", _prompt_validator)
        if validation_error is not None:
            return validation_error

        # After validation, we know prompt matches the Prompt type
        prompt: Prompt = cast("Prompt", body["prompt"])
        model = body.get("model")
        correlation_id = body.get("correlation_id")

        try:
            broker = getattr(self, "broker", None)
            if broker is not None:
                cc = broker.complete(LLMRequest(prompt=prompt, model=model))
            else:
                addr = getattr(self, "broker_address", None)
                if addr is None:
                    return {"error": "No broker configured"}
                cc = request_completion(
                    addr,
                    prompt,
                    model=model,
                    correlation_id=correlation_id,
                    timeout_s=self.timeout_s,
                )
        except Exception as exc:
            return {"error": str(exc)}

        with self.lock:
            self.pending_calls.append(cc)

        # Important: preserve legitimate empty-string response. Don't rely on truthiness.
        return {"response": cc.response}

    def _handle_batched(self, body: dict[str, Any]) -> dict[str, Any]:
        # Validate request fields using composable validators
        validation_error = _validate_request_fields(body, "prompts", _prompts_validator)
        if validation_error is not None:
            return validation_error

        prompts = body.get("prompts", [])
        model = body.get("model")
        correlation_id = body.get("correlation_id")

        # Preferred: in-process broker with per-item error semantics (legacy-compatible).
        broker = getattr(self, "broker", None)
        if broker is not None:
            results: list[str] = []
            for p in prompts:
                # Validate each prompt using the prompt validator
                prompt_result = _prompt_validator.validate_to_result(p)
                if isinstance(prompt_result, Err):
                    results.append(f"Error: {prompt_result.error}")
                    continue
                try:
                    cc = broker.complete(LLMRequest(prompt=p, model=model))
                except Exception as exc:
                    results.append(f"Error: {exc}")
                    continue
                with self.lock:
                    self.pending_calls.append(cc)
                results.append(cc.response)
            return {"responses": results}

        # Fallback: wire-protocol client against broker_address (per-item errors preserved).
        addr = getattr(self, "broker_address", None)
        if addr is None:
            return {"error": "No broker configured"}
        try:
            wire_results = request_completions_batched(
                addr,
                prompts,
                model=model,
                correlation_id=correlation_id,
                timeout_s=self.timeout_s,
            )
        except Exception as exc:
            return {"error": str(exc)}

        wire_responses: list[str] = []
        for r in wire_results:
            cc = r.chat_completion
            if r.error is not None or cc is None:
                wire_responses.append(f"Error: {r.error or 'No completion returned'}")
                continue
            with self.lock:
                self.pending_calls.append(cc)
            wire_responses.append(cc.response)
        return {"responses": wire_responses}


def _build_exec_script(
    code: str,
    proxy_port: int,
    /,
    *,
    proxy_timeout_s: float = DEFAULT_DOCKER_PROXY_HTTP_TIMEOUT_S,
    use_host_network: bool = False,
) -> str:
    """
    Build the container-side Python script.

    The script:
    - loads persistent locals from /workspace/state.pkl
    - defines llm_query + llm_query_batched using stdlib urllib (no pip deps)
    - executes the provided code in a persistent namespace
    - persists locals back to state.pkl (dropping unpickleable values)
    - prints a final JSON object containing stdout/stderr/locals
    """
    code_b64 = base64.b64encode(code.encode()).decode()
    # When using host network mode, the container shares the host's network namespace,
    # so we use localhost. Otherwise, use Docker's special DNS name.
    proxy_host = "localhost" if use_host_network else "host.docker.internal"

    # NOTE: The string checks in tests intentionally depend on this script shape.
    return textwrap.dedent(
        f"""
import sys, io, json, base64, os, uuid, pickle
from urllib import request as _urlreq

PROXY = "http://{proxy_host}:{proxy_port}"
STATE = "/workspace/state.pkl"
RUN_CORRELATION_ID = os.environ.get("RLM_CORRELATION_ID") or str(uuid.uuid4())

def _post_json(url, payload, timeout_s):
    data = json.dumps(payload).encode("utf-8")
    req = _urlreq.Request(url, data=data, headers={{"Content-Type": "application/json"}}, method="POST")
    with _urlreq.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))

def llm_query(prompt, model=None, correlation_id=None):
    try:
        cid = correlation_id or RUN_CORRELATION_ID
        d = _post_json(
            f"{{PROXY}}/llm_query",
            {{"prompt": prompt, "model": model, "correlation_id": cid}},
            {proxy_timeout_s},
        )
        # Important: preserve a legitimate empty-string response. Don't rely on
        # truthiness; explicitly branch on `error`.
        err = d.get("error")
        if err is not None:
            return f"Error: {{err}}"
        resp = d.get("response")
        return "" if resp is None else resp
    except Exception as e:
        return f"Error: {{e}}"

def llm_query_batched(prompts, model=None, correlation_id=None):
    try:
        cid = correlation_id or RUN_CORRELATION_ID
        d = _post_json(
            f"{{PROXY}}/llm_query_batched",
            {{"prompts": prompts, "model": model, "correlation_id": cid}},
            {proxy_timeout_s},
        )
        # Important: preserve an empty responses list for `prompts=[]`.
        err = d.get("error")
        if err is not None:
            return [f"Error: {{err}}"] * len(prompts)
        resps = d.get("responses")
        return [] if resps is None else resps
    except Exception as e:
        return [f"Error: {{e}}"] * len(prompts)

def load_state():
    if os.path.exists(STATE):
        try:
            with open(STATE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {{}}
    return {{}}

def save_state(s):
    clean = {{k: v for k, v in s.items() if not k.startswith("_")}}
    for k in list(clean.keys()):
        try:
            pickle.dumps(clean[k])
        except Exception:
            del clean[k]
    with open(STATE, "wb") as f:
        pickle.dump(clean, f)

_locals = load_state()

def FINAL_VAR(name):
    name = name.strip().strip("\\"\\'")
    return str(_locals.get(name, f"Error: Variable '{{name}}' not found"))

_globals = {{
    "__builtins__": __builtins__,
    "__name__": "__main__",
    "llm_query": llm_query,
    "llm_query_batched": llm_query_batched,
    "FINAL_VAR": FINAL_VAR,
}}

code = base64.b64decode("{code_b64}").decode()
stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
old_stdout, old_stderr = sys.stdout, sys.stderr
try:
    sys.stdout, sys.stderr = stdout_buf, stderr_buf
    combined = {{**_globals, **_locals}}
    exec(code, combined, combined)
    for k, v in combined.items():
        if k not in _globals and not k.startswith("_"):
            _locals[k] = v
except Exception as e:
    # Safe error mapping: avoid dumping full stack traces into user-facing output.
    stderr_buf.write(f"{{type(e).__name__}}: {{e}}")
finally:
    sys.stdout, sys.stderr = old_stdout, old_stderr

save_state(_locals)
print(
    json.dumps(
        {{
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "locals": {{k: repr(v) for k, v in _locals.items() if not k.startswith("_")}},
        }},
        ensure_ascii=False,
    )
)
""",
    )


class DockerEnvironmentAdapter(BaseEnvironmentAdapter):
    """
    Native Docker environment adapter (Phase 05).

    Runs Python code in a per-session container with:
    - persistent state across executions (pickled state under /workspace)
    - host proxy for nested `llm_query()` and `llm_query_batched()` calls
    - best-effort cleanup of container + proxy server + temp dirs
    """

    environment_type: str = "docker"

    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        broker: BrokerPort | None = None,
        broker_address: tuple[str, int] | None = None,
        correlation_id: str | None = None,
        context_payload: ContextPayload | None = None,
        setup_code: str | None = None,
        subprocess_timeout_s: float = DEFAULT_DOCKER_SUBPROCESS_TIMEOUT_S,
        proxy_http_timeout_s: float = DEFAULT_DOCKER_PROXY_HTTP_TIMEOUT_S,
        stop_grace_s: int = DEFAULT_DOCKER_STOP_GRACE_S,
        cleanup_subprocess_timeout_s: float = DEFAULT_DOCKER_CLEANUP_SUBPROCESS_TIMEOUT_S,
        thread_join_timeout_s: float = DEFAULT_DOCKER_THREAD_JOIN_TIMEOUT_S,
    ) -> None:
        self.image = image
        self._broker = broker
        self._broker_address = broker_address
        self._correlation_id = correlation_id
        self._subprocess_timeout_s = subprocess_timeout_s
        self._proxy_http_timeout_s = proxy_http_timeout_s
        self._stop_grace_s = stop_grace_s
        self._cleanup_subprocess_timeout_s = cleanup_subprocess_timeout_s
        self._thread_join_timeout_s = thread_join_timeout_s
        self._use_host_network = _use_host_network()

        self._tmp = tempfile.TemporaryDirectory(prefix="rlm_docker_env_")
        self._host_workspace = self._tmp.name

        self._calls_lock = threading.Lock()
        self._pending_calls: list[ChatCompletion] = []

        self._proxy_server: HTTPServer | None = None
        self._proxy_thread: threading.Thread | None = None
        self._proxy_port: int = 0

        self._container_id: str | None = None

        try:
            self._start_proxy()
            self._start_container()
        except Exception:
            # Ensure partial init doesn't leak threads/dirs/containers.
            self.cleanup()
            raise

        if context_payload is not None:
            self.load_context(context_payload)
        if setup_code:
            self.execute_code(setup_code)

    # ------------------------------------------------------------------
    # EnvironmentPort
    # ------------------------------------------------------------------

    def load_context(self, context_payload: ContextPayload, /) -> None:
        """
        Load context into the container-backed REPL.

        IMPORTANT: Do *not* embed `json.dumps(context_payload)` directly inside a quoted
        Python string literal. JSON does not escape single quotes, so payloads like
        {"name": "O'Brien"} will generate invalid Python code. We instead write the
        payload to a file in the mounted workspace and load it from within the
        container.
        """
        if isinstance(context_payload, str):
            host_path = os.path.join(self._host_workspace, "context.txt")
            container_path = "/workspace/context.txt"
            with open(host_path, "w", encoding="utf-8") as f:
                f.write(context_payload)
            self.execute_code(
                f"with open({container_path!r}, 'r', encoding='utf-8') as f:\n"
                "    context = f.read()",
            )
            return

        host_path = os.path.join(self._host_workspace, "context.json")
        container_path = "/workspace/context.json"
        with open(host_path, "w", encoding="utf-8") as f:
            json.dump(context_payload, f)
        self.execute_code(
            "import json\n"
            f"with open({container_path!r}, 'r', encoding='utf-8') as f:\n"
            "    context = json.load(f)",
        )

    def execute_code(self, code: str, /) -> ReplResult:
        container_id = self._container_id
        if not container_id:
            raise RuntimeError("Docker container not running (container_id is missing)")

        start = time.perf_counter()

        with self._calls_lock:
            self._pending_calls.clear()

        script = _build_exec_script(
            code,
            self._proxy_port,
            proxy_timeout_s=self._proxy_http_timeout_s,
            use_host_network=self._use_host_network,
        )

        cmd: list[str] = ["docker", "exec", "--workdir", "/workspace"]
        if self._correlation_id is not None:
            cmd.extend(["--env", f"RLM_CORRELATION_ID={self._correlation_id}"])
        cmd.extend([container_id, "python", "-c", script])

        try:
            result = subprocess.run(  # nosec B603 - safe list-form command, no shell injection risk
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._subprocess_timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            # Best-effort: stop the container so we don't leave a runaway python process behind.
            with self._calls_lock:
                calls = list(self._pending_calls)
                self._pending_calls.clear()
            try:
                self.cleanup()
            except Exception as cleanup_exc:  # nosec B110
                warn_cleanup_failure("DockerEnvironment.execute_timeout", cleanup_exc)

            raw_stderr = exc.stderr or b""
            raw_stdout = exc.stdout or b""
            stderr = (
                raw_stderr.decode() if isinstance(raw_stderr, bytes) else raw_stderr
            ) + f"\nTimeoutExpired: docker exec exceeded {self._subprocess_timeout_s}s"
            stdout = raw_stdout.decode() if isinstance(raw_stdout, bytes) else raw_stdout
            return ReplResult(
                stdout=stdout,
                stderr=stderr,
                locals={},
                llm_calls=calls,
                execution_time=time.perf_counter() - start,
            )

        with self._calls_lock:
            calls = list(self._pending_calls)
            self._pending_calls.clear()

        elapsed = time.perf_counter() - start
        try:
            lines = result.stdout.strip().split("\n")
            data_raw: Any = json.loads(lines[-1]) if lines else {}
            data: dict[str, Any] = cast(
                "dict[str, Any]",
                data_raw if isinstance(data_raw, dict) else {},
            )
            return ReplResult(
                stdout=str(data.get("stdout", "")),
                stderr=str(data.get("stderr", "")) + (result.stderr or ""),
                locals=dict(data.get("locals", {}) or {}),
                llm_calls=calls,
                execution_time=elapsed,
            )
        except json.JSONDecodeError:
            return ReplResult(
                stdout=result.stdout,
                stderr=result.stderr or "Parse error",
                locals={},
                llm_calls=calls,
                execution_time=elapsed,
            )

    def cleanup(self) -> None:
        # Cleanup must be idempotent and tolerate partial initialization.
        container_id = getattr(self, "_container_id", None)
        proxy_server = getattr(self, "_proxy_server", None)
        proxy_thread = getattr(self, "_proxy_thread", None)
        calls_lock = getattr(self, "_calls_lock", None)
        pending_calls = getattr(self, "_pending_calls", None)
        tmp = getattr(self, "_tmp", None)

        try:
            if container_id:
                subprocess.run(  # nosec B603 B607 - safe list-form, Docker CLI is standard practice
                    [
                        "docker",
                        "stop",
                        "-t",
                        str(getattr(self, "_stop_grace_s", DEFAULT_DOCKER_STOP_GRACE_S)),
                        container_id,
                    ],
                    check=False,
                    capture_output=True,
                    timeout=getattr(
                        self,
                        "_cleanup_subprocess_timeout_s",
                        DEFAULT_DOCKER_CLEANUP_SUBPROCESS_TIMEOUT_S,
                    ),
                )
        except Exception as exc:  # nosec B110
            warn_cleanup_failure("DockerEnvironment.cleanup_container_stop", exc)
        finally:
            try:
                self._container_id = None
            except Exception as exc:  # nosec B110
                warn_cleanup_failure("DockerEnvironment.cleanup_container_id_clear", exc)

        try:
            if proxy_server is not None:
                proxy_server.shutdown()
                proxy_server.server_close()
        except Exception as exc:  # nosec B110
            warn_cleanup_failure("DockerEnvironment.cleanup_proxy_server", exc)
        finally:
            try:
                self._proxy_server = None
            except Exception as exc:  # nosec B110
                warn_cleanup_failure("DockerEnvironment.cleanup_proxy_server_clear", exc)

        try:
            if proxy_thread is not None and proxy_thread.is_alive():
                proxy_thread.join(
                    timeout=getattr(
                        self,
                        "_thread_join_timeout_s",
                        DEFAULT_DOCKER_THREAD_JOIN_TIMEOUT_S,
                    ),
                )
        except Exception as exc:  # nosec B110
            warn_cleanup_failure("DockerEnvironment.cleanup_proxy_thread", exc)
        finally:
            try:
                self._proxy_thread = None
            except Exception as exc:  # nosec B110
                warn_cleanup_failure("DockerEnvironment.cleanup_proxy_thread_clear", exc)

        try:
            if calls_lock is not None and pending_calls is not None:
                with calls_lock:
                    pending_calls.clear()
        except Exception as exc:  # nosec B110
            warn_cleanup_failure("DockerEnvironment.cleanup_pending_calls", exc)

        try:
            if tmp is not None and hasattr(tmp, "cleanup"):
                tmp.cleanup()
        except Exception as exc:  # nosec B110
            warn_cleanup_failure("DockerEnvironment.cleanup_tmp", exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _start_proxy(self) -> None:
        handler: type[DockerLLMProxyHandler] = type(
            "DockerProxyHandler",
            (DockerLLMProxyHandler,),
            {
                "broker": self._broker,
                "broker_address": self._broker_address,
                "pending_calls": self._pending_calls,
                "lock": self._calls_lock,
                "timeout_s": self._proxy_http_timeout_s,
            },
        )
        # When using host network mode, bind to all interfaces so the container
        # (which shares the host's network namespace) can reach localhost:PORT.
        bind_addr = "0.0.0.0" if self._use_host_network else "127.0.0.1"  # nosec B104
        self._proxy_server = HTTPServer((bind_addr, 0), handler)  # nosec B104 - intentional for Docker host network
        self._proxy_port = self._proxy_server.server_address[1]
        self._proxy_thread = threading.Thread(
            target=self._proxy_server.serve_forever,
            name="rlm-docker-llm-proxy",
            daemon=True,
        )
        self._proxy_thread.start()

    def _start_container(self) -> None:
        cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            "-v",
            f"{self._host_workspace}:/workspace",
        ]
        if self._use_host_network:
            # Host network mode: container shares host's network namespace.
            # No need for --add-host; container uses localhost directly.
            cmd.append("--network=host")
        else:
            # Bridge network (default): use Docker's special DNS to reach host.
            cmd.extend(["--add-host", "host.docker.internal:host-gateway"])

        cmd.extend([self.image, "tail", "-f", "/dev/null"])

        result = subprocess.run(  # nosec B603 B607 - safe list-form command, Docker CLI is standard
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=self._subprocess_timeout_s,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")
        self._container_id = result.stdout.strip()
