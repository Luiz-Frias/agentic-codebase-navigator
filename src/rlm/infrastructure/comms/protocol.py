from __future__ import annotations

import json
from typing import Any

from rlm.domain.errors import BrokerError
from rlm.domain.models import ChatCompletion
from rlm.domain.policies.timeouts import DEFAULT_BROKER_CLIENT_TIMEOUT_S
from rlm.domain.types import Prompt
from rlm.infrastructure.comms.codec import DEFAULT_MAX_MESSAGE_BYTES, request_response
from rlm.infrastructure.comms.messages import WireRequest, WireResponse, WireResult


def _safe_error_message(exc: BaseException, /) -> str:
    """
    Convert internal exceptions into a client-safe error string.

    Important: do not leak stack traces or repr() of large/sensitive payloads.
    """
    match exc:
        case json.JSONDecodeError():
            return "Invalid JSON payload"
        case OSError():
            # Includes socket.timeout and connection-level failures.
            return "Connection error"
        case TimeoutError():
            return "Request timed out"
        case ConnectionError():
            return "Connection error"
        case ValueError() | TypeError():
            # These are expected validation errors; keep the message.
            return str(exc)
        case _:
            return "Internal broker error"


def try_parse_request(message: dict[str, Any], /) -> tuple[WireRequest | None, WireResponse | None]:
    """
    Parse a decoded JSON request into a WireRequest, or produce a safe WireResponse error.

    This is intended for broker servers to avoid crashing on malformed client payloads.
    """
    correlation_id = message.get("correlation_id")
    cid = correlation_id if isinstance(correlation_id, str) else None
    try:
        return WireRequest.from_dict(message), None
    except Exception as exc:  # noqa: BLE001 - boundary: convert to safe error response
        return None, WireResponse(correlation_id=cid, error=_safe_error_message(exc), results=None)


def parse_response(message: dict[str, Any], /) -> WireResponse:
    """Parse a decoded JSON response into a WireResponse (strict)."""
    return WireResponse.from_dict(message)


def send_request(
    address: tuple[str, int],
    request: WireRequest,
    /,
    *,
    timeout_s: float = DEFAULT_BROKER_CLIENT_TIMEOUT_S,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> WireResponse:
    """
    Client helper: send a WireRequest and parse the WireResponse.

    Raises:
        BrokerError: if the server responds with a request-level error.
    """
    try:
        raw = request_response(
            address,
            request.to_dict(),
            timeout_s=timeout_s,
            max_message_bytes=max_message_bytes,
        )
    except Exception as exc:  # noqa: BLE001 - boundary: map transport/protocol errors
        raise BrokerError(_safe_error_message(exc)) from None

    try:
        response = parse_response(raw)
    except Exception as exc:  # noqa: BLE001 - boundary: server bug / unexpected payload
        raise BrokerError(_safe_error_message(exc)) from None
    if response.error is not None:
        raise BrokerError(response.error)
    return response


def request_completion(
    address: tuple[str, int],
    prompt: Prompt,
    /,
    *,
    model: str | None = None,
    correlation_id: str | None = None,
    timeout_s: float = DEFAULT_BROKER_CLIENT_TIMEOUT_S,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> ChatCompletion:
    """
    Convenience client: request a single completion and return the ChatCompletion.

    Raises:
        BrokerError: for request-level errors or per-item errors.
    """
    req = WireRequest(correlation_id=correlation_id, prompt=prompt, model=model)
    resp = send_request(
        address,
        req,
        timeout_s=timeout_s,
        max_message_bytes=max_message_bytes,
    )
    if resp.results is None or len(resp.results) != 1:
        raise BrokerError("Invalid broker response: expected exactly 1 result")

    result = resp.results[0]
    if result.error is not None:
        raise BrokerError(result.error)
    if result.chat_completion is None:
        raise BrokerError("Invalid broker response: missing chat_completion")
    return result.chat_completion


def request_completions_batched(
    address: tuple[str, int],
    prompts: list[Prompt],
    /,
    *,
    model: str | None = None,
    correlation_id: str | None = None,
    timeout_s: float = DEFAULT_BROKER_CLIENT_TIMEOUT_S,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> list[WireResult]:
    """
    Convenience client: request a batched completion.

    Returns:
        A per-prompt list of WireResult, preserving ordering.

    Raises:
        BrokerError: for request-level errors (invalid request/transport).
    """
    if not prompts:
        return []
    req = WireRequest(correlation_id=correlation_id, prompts=prompts, model=model)
    resp = send_request(
        address,
        req,
        timeout_s=timeout_s,
        max_message_bytes=max_message_bytes,
    )
    if resp.results is None:
        raise BrokerError("Invalid broker response: missing results")
    if len(resp.results) != len(prompts):
        raise BrokerError(
            f"Invalid broker response: expected {len(prompts)} results, got {len(resp.results)}"
        )
    return resp.results
