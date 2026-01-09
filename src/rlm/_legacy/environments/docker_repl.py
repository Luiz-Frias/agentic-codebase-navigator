"""
Docker REPL environment that runs Python code in a Docker container.

Setup:
    docker build -t rlm-sandbox -f Dockerfile.sandbox .

Or use any Python 3.11+ image with: pip install dill requests
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from rlm._legacy.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm._legacy.core.types import REPLResult, RLMChatCompletion, UsageSummary
from rlm._legacy.environments.base_env import NonIsolatedEnv
from rlm.domain.models import LLMRequest as DomainLLMRequest
from rlm.domain.ports import BrokerPort


class LLMProxyHandler(BaseHTTPRequestHandler):
    """HTTP handler for LLM requests from the container."""

    broker: BrokerPort | None = None
    lm_handler_address: tuple[str, int] | None = None
    pending_calls: list[RLMChatCompletion] = []
    lock: threading.Lock = threading.Lock()

    def log_message(self, *args):
        pass

    def do_POST(self):
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
            body = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError:
            self._respond(400, {"error": "Invalid JSON payload"})
            return
        except Exception as exc:  # noqa: BLE001 - proxy boundary
            self._respond(400, {"error": str(exc)})
            return

        if not isinstance(body, dict):
            self._respond(400, {"error": "Request body must be a JSON object"})
            return

        if self.path == "/llm_query":
            result = self._handle_single(body)
        elif self.path == "/llm_query_batched":
            result = self._handle_batched(body)
        else:
            self._respond(404, {"error": "Not found"})
            return

        self._respond(200, result)

    def _respond(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _handle_single(self, body: dict) -> dict:
        broker = getattr(self, "broker", None)
        if broker is not None:
            prompt = body.get("prompt")
            model = body.get("model")
            correlation_id = body.get("correlation_id")  # reserved for later phases
            if not isinstance(prompt, (str, dict, list)):
                return {"error": "Invalid prompt"}
            if model is not None and not isinstance(model, str):
                return {"error": "Invalid model"}
            if correlation_id is not None and not isinstance(correlation_id, str):
                return {"error": "Invalid correlation_id"}
            try:
                cc = broker.complete(DomainLLMRequest(prompt=prompt, model=model))
            except Exception as exc:  # noqa: BLE001 - proxy boundary
                return {"error": str(exc)}

            legacy_prompt: str | dict[str, Any]
            if isinstance(prompt, (str, dict)):
                legacy_prompt = prompt
            else:
                legacy_prompt = {"prompt": prompt}

            with self.lock:
                self.pending_calls.append(
                    RLMChatCompletion(
                        root_model=cc.root_model,
                        prompt=legacy_prompt,
                        response=cc.response,
                        usage_summary=UsageSummary.from_dict(cc.usage_summary.to_dict()),
                        execution_time=cc.execution_time,
                    )
                )

            return {"response": cc.response}

        if not self.lm_handler_address:
            return {"error": "No LM handler configured"}

        prompt = body.get("prompt")
        model = body.get("model")
        if not isinstance(prompt, (str, dict)):
            return {"error": "Invalid prompt"}
        if model is not None and not isinstance(model, str):
            return {"error": "Invalid model"}
        request = LMRequest(prompt=prompt, model=model)
        response = send_lm_request(self.lm_handler_address, request)

        if not response.success:
            return {"error": response.error}

        with self.lock:
            self.pending_calls.append(response.chat_completion)

        return {"response": response.chat_completion.response}

    def _handle_batched(self, body: dict) -> dict:
        broker = getattr(self, "broker", None)
        if broker is not None:
            prompts = body.get("prompts", [])
            model = body.get("model")
            correlation_id = body.get("correlation_id")  # reserved for later phases
            if not isinstance(prompts, list):
                return {"error": "Invalid prompts"}
            if model is not None and not isinstance(model, str):
                return {"error": "Invalid model"}
            if correlation_id is not None and not isinstance(correlation_id, str):
                return {"error": "Invalid correlation_id"}

            results: list[str] = []
            for p in prompts:
                if not isinstance(p, (str, dict, list)):
                    results.append("Error: Invalid prompt")
                    continue
                try:
                    cc = broker.complete(DomainLLMRequest(prompt=p, model=model))
                except Exception as exc:  # noqa: BLE001 - per-item boundary
                    results.append(f"Error: {exc}")
                    continue

                legacy_prompt: str | dict[str, Any]
                if isinstance(p, (str, dict)):
                    legacy_prompt = p
                else:
                    legacy_prompt = {"prompt": p}

                with self.lock:
                    self.pending_calls.append(
                        RLMChatCompletion(
                            root_model=cc.root_model,
                            prompt=legacy_prompt,
                            response=cc.response,
                            usage_summary=UsageSummary.from_dict(cc.usage_summary.to_dict()),
                            execution_time=cc.execution_time,
                        )
                    )
                results.append(cc.response)

            return {"responses": results}

        if not self.lm_handler_address:
            return {"error": "No LM handler configured"}

        prompts = body.get("prompts", [])
        model = body.get("model")
        if not isinstance(prompts, list):
            return {"error": "Invalid prompts"}
        if model is not None and not isinstance(model, str):
            return {"error": "Invalid model"}
        responses = send_lm_request_batched(self.lm_handler_address, prompts, model=model)

        results = []
        for resp in responses:
            if not resp.success:
                results.append(f"Error: {resp.error}")
            else:
                with self.lock:
                    self.pending_calls.append(resp.chat_completion)
                results.append(resp.chat_completion.response)

        return {"responses": results}


def _build_exec_script(code: str, proxy_port: int) -> str:
    """Build execution script for the container."""
    code_b64 = base64.b64encode(code.encode()).decode()

    return textwrap.dedent(
        f"""
import sys, io, json, base64, traceback, os, requests, uuid
try:
    import dill
except ImportError:
    import pickle as dill

PROXY = "http://host.docker.internal:{proxy_port}"
STATE = "/workspace/state.dill"
RUN_CORRELATION_ID = os.environ.get("RLM_CORRELATION_ID") or str(uuid.uuid4())

def llm_query(prompt, model=None, correlation_id=None):
    try:
        cid = correlation_id or RUN_CORRELATION_ID
        r = requests.post(
            f"{{PROXY}}/llm_query",
            json={{"prompt": prompt, "model": model, "correlation_id": cid}},
            timeout=300,
        )
        d = r.json()
        return d.get("response") or f"Error: {{d.get('error')}}"
    except Exception as e:
        return f"Error: {{e}}"

def llm_query_batched(prompts, model=None, correlation_id=None):
    try:
        cid = correlation_id or RUN_CORRELATION_ID
        r = requests.post(
            f"{{PROXY}}/llm_query_batched",
            json={{"prompts": prompts, "model": model, "correlation_id": cid}},
            timeout=300,
        )
        d = r.json()
        return d.get("responses") or [f"Error: {{d.get('error')}}"] * len(prompts)
    except Exception as e:
        return [f"Error: {{e}}"] * len(prompts)

def load_state():
    if os.path.exists(STATE):
        try:
            with open(STATE, "rb") as f:
                return dill.load(f)
        except:
            pass
    return {{}}

def save_state(s):
    clean = {{k: v for k, v in s.items() if not k.startswith("_")}}
    for k in list(clean.keys()):
        try:
            dill.dumps(clean[k])
        except:
            del clean[k]
    with open(STATE, "wb") as f:
        dill.dump(clean, f)

_locals = load_state()

def FINAL_VAR(name):
    name = name.strip().strip("\\"\\'")
    return str(_locals.get(name, f"Error: Variable '{{name}}' not found"))

_globals = {{"__builtins__": __builtins__, "__name__": "__main__", "llm_query": llm_query, "llm_query_batched": llm_query_batched, "FINAL_VAR": FINAL_VAR}}

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
except:
    traceback.print_exc(file=stderr_buf)
finally:
    sys.stdout, sys.stderr = old_stdout, old_stderr

save_state(_locals)
print(json.dumps({{"stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue(), "locals": {{k: repr(v) for k, v in _locals.items() if not k.startswith("_")}}}}, ensure_ascii=False))
"""
    )


class DockerREPL(NonIsolatedEnv):
    """
    Docker REPL - runs Python in a Docker container with LLM support.

    Requires: Docker with a Python 3.11+ image (default: python:3.11-slim).
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        lm_handler_address: tuple[str, int] | None = None,
        broker: BrokerPort | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        *,
        subprocess_timeout_s: float = 300,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image = image
        self.lm_handler_address = lm_handler_address
        self._broker = broker
        self.subprocess_timeout_s = subprocess_timeout_s
        self.container_id: str | None = None
        self.proxy_server: HTTPServer | None = None
        self.proxy_thread: threading.Thread | None = None
        self.proxy_port: int = 0
        self.temp_dir = tempfile.mkdtemp(prefix="docker_repl_")
        self.pending_calls: list[RLMChatCompletion] = []
        self._calls_lock = threading.Lock()

        try:
            self.setup()
        except Exception:
            # Ensure we don't leak a proxy thread or temp dir if docker startup fails.
            self.cleanup()
            raise

        if context_payload:
            self.load_context(context_payload)
        if setup_code:
            self.execute_code(setup_code)

    def setup(self):
        """Start the proxy server and Docker container."""
        try:
            # Start LLM proxy server
            handler = type(
                "Handler",
                (LLMProxyHandler,),
                {
                    "broker": self._broker,
                    "lm_handler_address": self.lm_handler_address,
                    "pending_calls": self.pending_calls,
                    "lock": self._calls_lock,
                },
            )
            self.proxy_server = HTTPServer(("127.0.0.1", 0), handler)
            self.proxy_port = self.proxy_server.server_address[1]
            self.proxy_thread = threading.Thread(
                target=self.proxy_server.serve_forever, daemon=True
            )
            self.proxy_thread.start()

            # Start Docker container
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--rm",
                    "-v",
                    f"{self.temp_dir}:/workspace",
                    "--add-host",
                    "host.docker.internal:host-gateway",
                    self.image,
                    "tail",
                    "-f",
                    "/dev/null",
                ],
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout_s,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start container: {result.stderr}")

            self.container_id = result.stdout.strip()

            # Install dependencies
            subprocess.run(
                [
                    "docker",
                    "exec",
                    self.container_id,
                    "pip",
                    "install",
                    "-q",
                    "dill",
                    "requests",
                ],
                capture_output=True,
                timeout=self.subprocess_timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"Failed to start container: docker command timed out after {self.subprocess_timeout_s}s"
            ) from e
        except Exception:
            self.cleanup()
            raise

    def load_context(self, context_payload: dict | list | str):
        """
        Load context into the container-backed REPL.

        IMPORTANT: Do *not* embed `json.dumps(context_payload)` directly inside a quoted
        Python string literal. JSON does not escape single quotes, so payloads like
        {"name": "O'Brien"} will generate invalid Python code. We instead write the
        payload to a file in the mounted workspace and load it from within the
        container.
        """
        if isinstance(context_payload, str):
            host_context_path = os.path.join(self.temp_dir, "context.txt")
            container_context_path = "/workspace/context.txt"
            with open(host_context_path, "w", encoding="utf-8") as f:
                f.write(context_payload)
            self.execute_code(
                f"with open({container_context_path!r}, 'r', encoding='utf-8') as f:\n"
                "    context = f.read()"
            )
            return

        host_context_path = os.path.join(self.temp_dir, "context.json")
        container_context_path = "/workspace/context.json"
        with open(host_context_path, "w", encoding="utf-8") as f:
            json.dump(context_payload, f)
        self.execute_code(
            "import json\n"
            f"with open({container_context_path!r}, 'r', encoding='utf-8') as f:\n"
            "    context = json.load(f)"
        )

    def execute_code(self, code: str) -> REPLResult:
        if not self.container_id:
            raise RuntimeError("Docker container not running (container_id is missing)")
        start = time.perf_counter()

        with self._calls_lock:
            self.pending_calls.clear()

        script = _build_exec_script(code, self.proxy_port)
        try:
            result = subprocess.run(
                ["docker", "exec", self.container_id, "python", "-c", script],
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            # Best-effort: stop the container so we don't leave a runaway python process behind.
            with self._calls_lock:
                calls = self.pending_calls.copy()
                self.pending_calls.clear()

            try:
                self.cleanup()
            except Exception:
                pass

            stderr = (
                e.stderr or ""
            ) + f"\nTimeoutExpired: docker exec exceeded {self.subprocess_timeout_s}s"
            stdout = e.stdout or ""
            return REPLResult(
                stdout=stdout,
                stderr=stderr,
                locals={},
                execution_time=time.perf_counter() - start,
                rlm_calls=calls,
            )

        with self._calls_lock:
            calls = self.pending_calls.copy()
            self.pending_calls.clear()

        try:
            lines = result.stdout.strip().split("\n")
            data = json.loads(lines[-1]) if lines else {}
            return REPLResult(
                stdout=data.get("stdout", ""),
                stderr=data.get("stderr", "") + result.stderr,
                locals=data.get("locals", {}),
                execution_time=time.perf_counter() - start,
                rlm_calls=calls,
            )
        except json.JSONDecodeError:
            return REPLResult(
                stdout=result.stdout,
                stderr=result.stderr or "Parse error",
                locals={},
                execution_time=time.perf_counter() - start,
                rlm_calls=calls,
            )

    def cleanup(self):
        # Cleanup must be idempotent and tolerate partial initialization.
        container_id = getattr(self, "container_id", None)
        proxy_server = getattr(self, "proxy_server", None)
        proxy_thread = getattr(self, "proxy_thread", None)
        temp_dir = getattr(self, "temp_dir", None)

        try:
            if container_id:
                # Don't block indefinitely on docker stop.
                subprocess.run(
                    ["docker", "stop", "-t", "2", container_id],
                    capture_output=True,
                    timeout=5,
                )
        except Exception:
            pass
        finally:
            try:
                self.container_id = None
            except Exception:
                pass

        try:
            if proxy_server is not None:
                proxy_server.shutdown()
                proxy_server.server_close()
        except Exception:
            pass
        finally:
            try:
                self.proxy_server = None
            except Exception:
                pass

        try:
            if proxy_thread is not None and proxy_thread.is_alive():
                proxy_thread.join(timeout=2)
        except Exception:
            pass
        finally:
            try:
                self.proxy_thread = None
            except Exception:
                pass

        try:
            if temp_dir and os.path.exists(temp_dir):
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()
        return False

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            # Best-effort cleanup; never raise from finalizer.
            pass
