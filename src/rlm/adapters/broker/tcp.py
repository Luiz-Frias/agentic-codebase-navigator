from __future__ import annotations

import asyncio
import concurrent.futures
import time
from socketserver import StreamRequestHandler, ThreadingTCPServer
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any, Final, TypeVar

from rlm.adapters.base import BaseBrokerAdapter
from rlm.domain.errors import LLMError, ValidationError
from rlm.domain.models import (
    BatchedLLMRequest,
    ChatCompletion,
    LLMRequest,
    ModelSpec,
    ModelUsageSummary,
    UsageSummary,
    build_routing_rules,
)
from rlm.domain.models.usage import merge_usage_summaries
from rlm.domain.policies.timeouts import (
    DEFAULT_BROKER_ASYNC_LOOP_START_TIMEOUT_S,
    DEFAULT_BROKER_THREAD_JOIN_TIMEOUT_S,
    BrokerTimeouts,
    CancellationPolicy,
)
from rlm.infrastructure.comms.codec import (
    DEFAULT_MAX_MESSAGE_BYTES,
    recv_frame,
    send_frame,
)
from rlm.infrastructure.comms.messages import WireRequest, WireResponse, WireResult
from rlm.infrastructure.comms.protocol import try_parse_request
from rlm.infrastructure.logging import warn_cleanup_failure

if TYPE_CHECKING:
    from collections.abc import Coroutine, Sequence

    from rlm.domain.ports import LLMPort
    from rlm.domain.types import Prompt

_T = TypeVar("_T")


def _safe_error_message(exc: BaseException, /) -> str:
    """
    Convert internal exceptions to client-safe error strings.

    Important: do not leak stack traces or repr() of large/sensitive payloads.
    """
    match exc:
        case LLMError():
            return str(exc)
        case ValidationError():
            return str(exc)
        case ValueError() | TypeError():
            return str(exc)
        case TimeoutError():
            return "Request timed out"
        case ConnectionError() | OSError():
            return "Connection error"
        case _:
            return "Internal broker error"


class _AsyncLoopThread:
    """
    Single shared asyncio loop used for batched request execution.

    This avoids calling `asyncio.run()` inside `socketserver` handler threads.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Thread | None = None
        self._ready = Event()

    def start(self) -> None:
        if self._thread is not None:
            return

        # Allow restart after stop().
        self._ready.clear()

        def _runner() -> None:
            loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            self._ready.set()
            loop.run_forever()

            # Best-effort cleanup on stop.
            try:
                pending: set[asyncio.Task[Any]] = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            finally:
                loop.close()

        self._thread = Thread(target=_runner, name="rlm-tcp-broker-async-loop", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=DEFAULT_BROKER_ASYNC_LOOP_START_TIMEOUT_S)
        if self._loop is None:
            raise RuntimeError("Async loop thread failed to start")

    def stop(self) -> None:
        if self._thread is None or self._loop is None:
            self._thread = None
            self._loop = None
            return

        loop: asyncio.AbstractEventLoop = self._loop
        loop.call_soon_threadsafe(loop.stop)
        self._thread.join(timeout=DEFAULT_BROKER_THREAD_JOIN_TIMEOUT_S)
        self._thread = None
        self._loop = None

    def run(
        self,
        coro: Coroutine[Any, Any, _T],
        /,
        *,
        timeout_s: float | None = None,
        cancellation: CancellationPolicy | None = None,
    ) -> _T:
        if self._loop is None:
            raise RuntimeError("Async loop not started")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        if timeout_s is None:
            return fut.result()

        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            # Best-effort cancellation: ask the loop to cancel the coroutine.
            fut.cancel()

            # Give cancellation a small grace period to propagate and clean up children.
            grace = cancellation.grace_timeout_s if cancellation is not None else 0.0
            if grace > 0:
                try:
                    fut.result(timeout=grace)
                except Exception as exc:  # nosec B110
                    warn_cleanup_failure("TcpBrokerAdapter.cancellation_grace", exc)
            raise TimeoutError("Batched request timed out") from None


class _TcpBrokerServer(ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class _TcpBrokerRequestHandler(StreamRequestHandler):
    def handle(self) -> None:
        broker: TcpBrokerAdapter = self.server.broker  # type: ignore[attr-defined]

        try:
            raw: dict[str, Any] | None = recv_frame(
                self.connection,
                max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES,
            )
            if raw is None:
                return

            request: WireRequest | None
            parse_error: WireResponse | None
            request, parse_error = try_parse_request(raw)
            if parse_error is not None:
                send_frame(self.connection, parse_error.to_dict())
                return
            assert request is not None

            response: WireResponse = broker._handle_wire_request(request)
            send_frame(self.connection, response.to_dict())
        except Exception as exc:
            send_frame(
                self.connection,
                WireResponse(
                    correlation_id=None,
                    error=_safe_error_message(exc),
                    results=None,
                ).to_dict(),
            )


class TcpBrokerAdapter(BaseBrokerAdapter):
    """
    A TCP broker that:
    - exposes the infra wire protocol over a length-prefixed JSON socket server
    - routes requests to registered `LLMPort` implementations by model name
    - supports batched requests via an asyncio TaskGroup (runs on a dedicated loop thread)
    """

    _DEFAULT_HOST: Final[str] = "127.0.0.1"

    def __init__(
        self,
        default_llm: LLMPort,
        *,
        host: str = _DEFAULT_HOST,
        port: int = 0,
        timeouts: BrokerTimeouts | None = None,
        cancellation: CancellationPolicy | None = None,
    ) -> None:
        self._host = host
        self._port = port

        self._default_llm = default_llm
        self._llms: dict[str, LLMPort] = {default_llm.model_name: default_llm}
        self._routing_rules = build_routing_rules(
            [ModelSpec(name=default_llm.model_name, is_default=True)],
        )

        self._timeouts = timeouts or BrokerTimeouts()
        self._cancellation = cancellation or CancellationPolicy()

        self._server: _TcpBrokerServer | None = None
        self._thread: Thread | None = None
        self._async_loop = _AsyncLoopThread()

        # Per-broker usage totals (tracks only calls routed *through this broker*).
        self._usage_lock = Lock()
        self._usage = UsageSummary(model_usage_summaries={})

    # ---------------------------------------------------------------------
    # BrokerPort
    # ---------------------------------------------------------------------

    def register_llm(self, model_name: str, llm: LLMPort, /) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValidationError("Broker.register_llm requires a non-empty model_name")
        if model_name != llm.model_name:
            raise ValidationError(
                f"Broker.register_llm model_name {model_name!r} must match llm.model_name {llm.model_name!r}",
            )

        self._llms[model_name] = llm
        # Rebuild routing rules deterministically (avoid accidental ambiguity).
        default: str = self._default_llm.model_name
        specs: list[ModelSpec] = [ModelSpec(name=default, is_default=True)]
        for name in sorted(self._llms):
            if name == default:
                continue
            specs.append(ModelSpec(name=name))
        self._routing_rules = build_routing_rules(specs)

    def start(self) -> tuple[str, int]:
        if self._server is not None:
            return self.address

        self._async_loop.start()

        self._server = _TcpBrokerServer((self._host, self._port), _TcpBrokerRequestHandler)
        self._server.broker = self  # type: ignore[attr-defined]
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.address

    def stop(self) -> None:
        if self._server is None:
            self._async_loop.stop()
            return

        thread: Thread | None = self._thread
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        if thread is not None:
            thread.join(timeout=DEFAULT_BROKER_THREAD_JOIN_TIMEOUT_S)
        self._thread = None

        self._async_loop.stop()

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        llm: LLMPort = self._select_llm(request.model)
        start: float = time.perf_counter()
        # `request.model` is used for *routing*; once selected, call the chosen
        # adapter with its own model name to keep `root_model` and usage consistent.
        cc: ChatCompletion = llm.complete(LLMRequest(prompt=request.prompt, model=llm.model_name))
        end: float = time.perf_counter()

        # Preserve adapter timings when the LLM impl doesn't set execution_time.
        if cc.execution_time == 0.0:
            cc = ChatCompletion(
                root_model=cc.root_model,
                prompt=cc.prompt,
                response=cc.response,
                usage_summary=cc.usage_summary,
                execution_time=end - start,
            )

        self._record_usage(cc.usage_summary)
        return cc

    def complete_batched(self, request: BatchedLLMRequest, /) -> list[ChatCompletion]:
        llm: LLMPort = self._select_llm(request.model)

        async def _run() -> list[ChatCompletion | Exception]:
            return await _acomplete_prompts_batched(llm, request.prompts, llm.model_name)

        try:
            out: list[ChatCompletion | Exception] = self._async_loop.run(
                _run(),
                timeout_s=self._timeouts.batched_completion_timeout_s,
                cancellation=self._cancellation,
            )
        except TimeoutError as exc:
            raise LLMError(str(exc)) from None

        results: list[ChatCompletion] = []
        # Record usage for all successful items, even if some items failed.
        for item in out:
            if isinstance(item, Exception):
                raise LLMError(_safe_error_message(item))
            self._record_usage(item.usage_summary)
            results.append(item)
        return results

    def get_usage_summary(self) -> UsageSummary:
        # Snapshot + clone (avoid external mutation and keep keys deterministic).
        with self._usage_lock:
            return merge_usage_summaries([self._usage])

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None:
            return (self._host, self._port)
        return (self._host, self._server.server_address[1])

    def _select_llm(self, model: str | None, /) -> LLMPort:
        resolved: str = self._routing_rules.resolve(model)
        return self._llms[resolved]

    def _record_usage(self, usage: UsageSummary, /) -> None:
        """
        Merge a per-call usage summary into the broker's totals.

        This intentionally tracks *only* calls routed through this broker instance,
        independent of any internal totals maintained by LLM adapters.
        """
        with self._usage_lock:
            for model, mus in usage.model_usage_summaries.items():
                current: ModelUsageSummary | None = self._usage.model_usage_summaries.get(model)
                if current is None:
                    self._usage.model_usage_summaries[model] = ModelUsageSummary(
                        total_calls=mus.total_calls,
                        total_input_tokens=mus.total_input_tokens,
                        total_output_tokens=mus.total_output_tokens,
                    )
                else:
                    current.total_calls += mus.total_calls
                    current.total_input_tokens += mus.total_input_tokens
                    current.total_output_tokens += mus.total_output_tokens

    def _handle_wire_request(self, request: WireRequest, /) -> WireResponse:
        try:
            if request.prompt is not None:
                try:
                    cc: ChatCompletion = self.complete(
                        LLMRequest(prompt=request.prompt, model=request.model),
                    )
                    result: WireResult = WireResult(error=None, chat_completion=cc)
                except Exception as exc:
                    result = WireResult(error=_safe_error_message(exc), chat_completion=None)

                return WireResponse(
                    correlation_id=request.correlation_id,
                    error=None,
                    results=[result],
                )

            if request.prompts is None:
                return WireResponse(
                    correlation_id=request.correlation_id,
                    error="WireRequest missing prompts",
                    results=None,
                )

            llm: LLMPort = self._select_llm(request.model)
            # Capture for closure type narrowing - request.prompts is already validated
            prompts: Sequence[Prompt] = request.prompts  # type: ignore[assignment]  # Wire protocol validates

            async def _run() -> list[WireResult]:
                out: list[ChatCompletion | Exception] = await _acomplete_prompts_batched(
                    llm,
                    prompts,
                    llm.model_name,
                )
                results: list[WireResult] = []
                for item in out:
                    if isinstance(item, Exception):
                        results.append(
                            WireResult(error=_safe_error_message(item), chat_completion=None),
                        )
                    else:
                        self._record_usage(item.usage_summary)
                        results.append(WireResult(error=None, chat_completion=item))
                return results

            try:
                results: list[WireResult] = self._async_loop.run(
                    _run(),
                    timeout_s=self._timeouts.batched_completion_timeout_s,
                    cancellation=self._cancellation,
                )
            except TimeoutError as exc:
                return WireResponse(
                    correlation_id=request.correlation_id,
                    error=_safe_error_message(exc),
                    results=None,
                )
            return WireResponse(correlation_id=request.correlation_id, error=None, results=results)
        except Exception as exc:
            return WireResponse(
                correlation_id=request.correlation_id,
                error=_safe_error_message(exc),
                results=None,
            )


async def _acomplete_prompts_batched(
    llm: LLMPort,
    prompts: Sequence[Prompt],
    model: str | None,
) -> list[ChatCompletion | Exception]:
    """
    Run multiple `llm.acomplete(...)` calls concurrently and preserve ordering.

    This is used by both the direct `complete_batched` API and the wire-protocol
    handler to keep concurrency logic consistent.
    """
    out: list[ChatCompletion | Exception] = [Exception("uninitialized")] * len(prompts)

    async def _one(i: int, prompt: Prompt) -> None:
        try:
            out[i] = await llm.acomplete(LLMRequest(prompt=prompt, model=model))
        except Exception as exc:
            out[i] = exc

    async with asyncio.TaskGroup() as tg:
        for i, p in enumerate(prompts):
            tg.create_task(_one(i, p))

    return out
