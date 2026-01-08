"""
LMHandler - Routes LLM requests from the RLM process and environment subprocesses.

Uses a multi-threaded socket server. Protocol: 4-byte length prefix + JSON payload.
"""

from __future__ import annotations

import asyncio
import time
from socketserver import StreamRequestHandler, ThreadingTCPServer
from threading import Thread
from typing import Any

from rlm._legacy.clients.base_lm import BaseLM
from rlm._legacy.core.comms_utils import LMRequest, LMResponse, socket_recv, socket_send
from rlm._legacy.core.types import RLMChatCompletion, UsageSummary


class LMRequestHandler(StreamRequestHandler):
    """Socket handler for LLM completion requests."""

    def handle(self):
        try:
            request_data = socket_recv(self.connection)
            if not isinstance(request_data, dict):
                response = LMResponse.error_response("Request must be a JSON object")
                socket_send(self.connection, response.to_dict())
                return

            request = LMRequest.from_dict(request_data)
            handler: LMHandler = self.server.lm_handler  # type: ignore[attr-defined]

            if request.is_batched:
                # Batched request: process multiple prompts.
                response = self._handle_batched(request, handler)
            elif request.prompt:
                # Single request: process one prompt
                response = self._handle_single(request, handler)
            else:
                response = LMResponse.error_response("Missing 'prompt' or 'prompts' in request.")

            socket_send(self.connection, response.to_dict())

        except Exception as e:
            response = LMResponse.error_response(str(e))
            socket_send(self.connection, response.to_dict())

    def _handle_single(self, request: LMRequest, handler: LMHandler) -> LMResponse:
        """Handle a single prompt request."""
        client = handler.get_client(request.model)

        start_time = time.perf_counter()
        content = client.completion(request.prompt)
        end_time = time.perf_counter()

        usage_summary = client.get_last_usage()
        return LMResponse.success_response(
            chat_completion=RLMChatCompletion(
                root_model=request.model or client.model_name,
                prompt=request.prompt,
                response=content,
                usage_summary=usage_summary,
                execution_time=end_time - start_time,
            )
        )

    def _handle_batched(self, request: LMRequest, handler: LMHandler) -> LMResponse:
        """Handle a batched prompts request.

        NOTE: `BaseLM.get_last_usage()` is defined as "usage for the last call". Many
        implementations store this as shared mutable state, which makes it unsafe
        to call `acompletion()` concurrently and then try to attribute per-prompt
        usage after the fact.

        We therefore execute batched prompts sequentially to preserve correct
        per-prompt usage accounting. If/when provider adapters expose a safe
        per-call usage API, this can be revisited for concurrency.
        """
        client = handler.get_client(request.model)

        start_time = time.perf_counter()

        async def run_all_with_usage():
            outputs: list[tuple[str, UsageSummary]] = []
            for prompt in request.prompts:
                content = await client.acompletion(prompt)
                # Defensive copy so later calls can't mutate earlier per-call usage objects.
                usage = UsageSummary.from_dict(client.get_last_usage().to_dict())
                outputs.append((content, usage))
            return outputs

        results_with_usage = asyncio.run(run_all_with_usage())
        end_time = time.perf_counter()

        total_time = end_time - start_time

        chat_completions = [
            RLMChatCompletion(
                root_model=request.model or client.model_name,
                prompt=prompt,
                response=content,
                usage_summary=usage_summary,
                execution_time=total_time / len(request.prompts),  # approximate per-prompt time
            )
            for prompt, (content, usage_summary) in zip(
                request.prompts, results_with_usage, strict=True
            )
        ]

        return LMResponse.batched_success_response(chat_completions=chat_completions)


class ThreadingLMServer(ThreadingTCPServer):
    """Multi-threaded TCP server for LM requests."""

    daemon_threads = True
    allow_reuse_address = True


class LMHandler:
    """
    Handles all LM calls from the RLM main process and environment subprocesses.

    Uses a multi-threaded socket server for concurrent requests.
    Protocol: 4-byte big-endian length prefix + JSON payload.
    """

    def __init__(
        self,
        client: BaseLM,
        host: str = "127.0.0.1",
        port: int = 0,  # auto-assign available port
    ):
        self.default_client = client
        self.clients: dict[str, BaseLM] = {}
        self.host = host
        self._server: ThreadingLMServer | None = None
        self._thread: Thread | None = None
        self._port = port

        self.register_client(client.model_name, client)

    def register_client(self, model_name: str, client: BaseLM) -> None:
        """Register a client for a specific model name."""
        self.clients[model_name] = client

    def get_client(self, model: str | None = None) -> BaseLM:
        """Get client by model name, or return default."""
        if model and model in self.clients:
            return self.clients[model]
        return self.default_client

    @property
    def port(self) -> int:
        """Get the actual port (useful when auto-assigned)."""
        if self._server:
            return self._server.server_address[1]
        return self._port

    @property
    def address(self) -> tuple[str, int]:
        """Get (host, port) tuple for connecting."""
        return (self.host, self.port)

    def start(self) -> tuple[str, int]:
        """Start the socket server in a background thread. Returns (host, port)."""
        if self._server is not None:
            return self.address

        self._server = ThreadingLMServer((self.host, self._port), LMRequestHandler)
        self._server.lm_handler = self  # type: ignore[attr-defined]

        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        return self.address

    def stop(self):
        """Stop the socket server."""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None

    def completion(self, prompt: str | dict[str, Any], model: str | None = None) -> str:
        """Direct completion call (for main process use)."""
        return self.get_client(model).completion(prompt)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def get_usage_summary(self) -> UsageSummary:
        """Get the usage summary for all clients, merged into a single dict."""
        merged = {}
        for client in self.clients.values():
            client_summary = client.get_usage_summary()
            merged.update(client_summary.model_usage_summaries)
        return UsageSummary(model_usage_summaries=merged)
