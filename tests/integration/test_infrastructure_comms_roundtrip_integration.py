from __future__ import annotations

from socketserver import StreamRequestHandler, ThreadingTCPServer
from threading import Thread
from typing import ClassVar

import pytest

from rlm.domain.models import ChatCompletion, ModelUsageSummary, UsageSummary
from rlm.infrastructure.comms.codec import (
    DEFAULT_MAX_MESSAGE_BYTES,
    recv_frame,
    send_frame,
)
from rlm.infrastructure.comms.messages import WireResponse, WireResult
from rlm.infrastructure.comms.protocol import (
    request_completion,
    request_completions_batched,
    try_parse_request,
)


def _fake_chat_completion(*, prompt: object, response: str, model: str = "m") -> ChatCompletion:
    usage = UsageSummary(model_usage_summaries={model: ModelUsageSummary(1, 2, 3)})
    return ChatCompletion(
        root_model=model,
        prompt=prompt,
        response=response,
        usage_summary=usage,
        execution_time=0.01,
    )


class _WireProtocolHandler(StreamRequestHandler):
    """Minimal in-process broker stub that speaks the infra wire protocol.

    This is intentionally tiny: it validates/decodes WireRequest via `try_parse_request`
    and returns a WireResponse using the same framing/codec as production.
    """

    MODEL: ClassVar[str] = "stub"

    def handle(self) -> None:
        raw = recv_frame(self.connection, max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES)
        if raw is None:
            return

        req, err = try_parse_request(raw)
        if err is not None:
            send_frame(self.connection, err.to_dict())
            return

        assert req is not None
        if req.prompt is not None:
            cc = _fake_chat_completion(prompt=req.prompt, response="FINAL(ok)", model=self.MODEL)
            resp = WireResponse(
                correlation_id=req.correlation_id,
                error=None,
                results=[WireResult(error=None, chat_completion=cc)],
            )
            send_frame(self.connection, resp.to_dict())
            return

        assert req.prompts is not None
        results: list[WireResult] = []
        for i, p in enumerate(req.prompts):
            if i == 1:
                results.append(WireResult(error="oops", chat_completion=None))
            else:
                results.append(
                    WireResult(
                        error=None,
                        chat_completion=_fake_chat_completion(
                            prompt=p,
                            response=f"r{i + 1}",
                            model=self.MODEL,
                        ),
                    ),
                )

        send_frame(
            self.connection,
            WireResponse(correlation_id=req.correlation_id, error=None, results=results).to_dict(),
        )


@pytest.mark.integration
def test_wire_protocol_roundtrip_single_request() -> None:
    server = ThreadingTCPServer(("127.0.0.1", 0), _WireProtocolHandler)
    server.daemon_threads = True
    server.allow_reuse_address = True

    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        addr = server.server_address
        cc = request_completion(addr, "hello", correlation_id="cid")
        assert cc.response == "FINAL(ok)"
        assert cc.root_model == _WireProtocolHandler.MODEL
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.integration
def test_wire_protocol_roundtrip_batched_request_preserves_order_and_errors() -> None:
    server = ThreadingTCPServer(("127.0.0.1", 0), _WireProtocolHandler)
    server.daemon_threads = True
    server.allow_reuse_address = True

    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        addr = server.server_address
        results = request_completions_batched(addr, ["p1", "p2", "p3"], correlation_id="cid")
        assert [r.error for r in results] == [None, "oops", None]
        assert (
            results[0].chat_completion is not None and results[0].chat_completion.response == "r1"
        )
        assert (
            results[2].chat_completion is not None and results[2].chat_completion.response == "r3"
        )
    finally:
        server.shutdown()
        server.server_close()
