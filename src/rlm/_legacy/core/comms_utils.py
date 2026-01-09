"""
Communication utilities for RLM socket protocol.

Protocol: 4-byte big-endian length prefix + JSON payload.
Used for communication between LMHandler and environment subprocesses.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any

from rlm._legacy.core.types import RLMChatCompletion, UsageSummary
from rlm.domain.policies.timeouts import DEFAULT_BROKER_CLIENT_TIMEOUT_S
from rlm.infrastructure.comms.codec import (
    DEFAULT_MAX_MESSAGE_BYTES,
)
from rlm.infrastructure.comms.codec import (
    recv_frame as _recv_frame,
)
from rlm.infrastructure.comms.codec import (
    request_response as _request_response,
)
from rlm.infrastructure.comms.codec import (
    send_frame as _send_frame,
)

# =============================================================================
# Message Dataclasses
# =============================================================================


@dataclass
class LMRequest:
    """Request message sent to the LM Handler.

    Supports both single prompt (prompt field) and batched prompts (prompts field).
    """

    prompt: str | dict[str, Any] | None = None
    prompts: list[str | dict[str, Any]] | None = None
    model: str | None = None

    @property
    def is_batched(self) -> bool:
        """Check if this is a batched request."""
        return self.prompts is not None and len(self.prompts) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        d: dict[str, Any] = {}
        if self.prompt is not None:
            d["prompt"] = self.prompt
        if self.prompts is not None:
            d["prompts"] = self.prompts
        if self.model is not None:
            d["model"] = self.model
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LMRequest:
        """Create from dict."""
        return cls(
            prompt=data.get("prompt"),
            prompts=data.get("prompts"),
            model=data.get("model"),
        )


@dataclass
class LMResponse:
    """Response message from the LM Handler.

    Supports both single response (chat_completion) and batched responses (chat_completions).
    """

    error: str | None = None
    chat_completion: RLMChatCompletion | None = None
    chat_completions: list[RLMChatCompletion] | None = None

    @property
    def success(self) -> bool:
        """Check if response was successful."""
        return self.error is None

    @property
    def is_batched(self) -> bool:
        """Check if this is a batched response."""
        return self.chat_completions is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        if self.error is not None:
            return {
                "error": self.error,
                "chat_completion": None,
                "chat_completions": None,
            }
        if self.chat_completions is not None:
            return {
                "chat_completions": [c.to_dict() for c in self.chat_completions],
                "chat_completion": None,
                "error": None,
            }
        if self.chat_completion is not None:
            return {
                "chat_completion": self.chat_completion.to_dict(),
                "chat_completions": None,
                "error": None,
            }
        return {
            "error": "No chat completion or error provided.",
            "chat_completion": None,
            "chat_completions": None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LMResponse:
        """Create from dict."""
        chat_completions = None
        if data.get("chat_completions"):
            chat_completions = [RLMChatCompletion.from_dict(c) for c in data["chat_completions"]]

        chat_completion = None
        if data.get("chat_completion"):
            chat_completion = RLMChatCompletion.from_dict(data["chat_completion"])

        return cls(
            error=data.get("error"),
            chat_completion=chat_completion,
            chat_completions=chat_completions,
        )

    @classmethod
    def success_response(cls, chat_completion: RLMChatCompletion) -> LMResponse:
        """Create a successful single response."""
        return cls(chat_completion=chat_completion)

    @classmethod
    def batched_success_response(cls, chat_completions: list[RLMChatCompletion]) -> LMResponse:
        """Create a successful batched response."""
        return cls(chat_completions=chat_completions)

    @classmethod
    def error_response(cls, error: str) -> LMResponse:
        """Create an error response."""
        return cls(error=error)


# =============================================================================
# Socket Protocol Helpers
# =============================================================================


def socket_send(sock: socket.socket, data: dict[str, Any]) -> None:
    """Send a length-prefixed JSON message over socket.

    Protocol: 4-byte big-endian length prefix + UTF-8 JSON payload.
    """
    _send_frame(sock, data)


def socket_recv(sock: socket.socket) -> dict[str, Any]:
    """Receive a length-prefixed JSON message from socket.

    Protocol: 4-byte big-endian length prefix + UTF-8 JSON payload.
    Returns empty dict if connection closed before length received.

    Raises:
        ConnectionError: If connection closes mid-message.
    """
    message = _recv_frame(sock, max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES)
    if message is None:
        # Legacy behavior: empty dict indicates the peer closed before length prefix.
        return {}
    return message


def socket_request(
    address: tuple[str, int], data: dict[str, Any], timeout: float = DEFAULT_BROKER_CLIENT_TIMEOUT_S
) -> dict[str, Any]:
    """Send a request and receive a response over a new socket connection.

    Opens a new TCP connection, sends the request, waits for response, then closes.

    Args:
        address: (host, port) tuple to connect to.
        data: Dictionary to send as JSON.
        timeout: Socket timeout in seconds.

    Returns:
        Response dictionary.
    """
    try:
        return _request_response(
            address,
            data,
            timeout_s=float(timeout),
            max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES,
        )
    except ConnectionError as exc:
        # Preserve legacy semantics: treat "closed before response frame" as empty dict,
        # but do not swallow mid-payload truncation errors.
        if str(exc) == "Connection closed before response frame":
            return {}
        raise


# =============================================================================
# Typed Request Helpers
# =============================================================================


def send_lm_request(
    address: tuple[str, int],
    request: LMRequest,
    timeout: float = DEFAULT_BROKER_CLIENT_TIMEOUT_S,
) -> LMResponse:
    """Send an LM request and return typed response.

    Args:
        address: (host, port) tuple of LM Handler server.
        request: LMRequest to send.
        timeout: Socket timeout in seconds.

    Returns:
        LMResponse with content or error.
    """
    try:
        response_data = socket_request(address, request.to_dict(), timeout)
        if not isinstance(response_data, dict):
            return LMResponse.error_response("Invalid response (expected JSON object)")

        # Compat: support both legacy LMHandler responses (chat_completion/chat_completions)
        # and new wire protocol responses (error/results).
        if "results" in response_data:
            # WireResponse: {"correlation_id": ..., "error": str|None, "results": [...]|None}
            if response_data.get("error"):
                return LMResponse.error_response(str(response_data.get("error")))
            results = response_data.get("results") or []
            if not isinstance(results, list) or len(results) != 1:
                return LMResponse.error_response("Invalid broker response: expected 1 result")
            item = results[0]
            if not isinstance(item, dict):
                return LMResponse.error_response(
                    "Invalid broker response: result must be an object"
                )
            if item.get("error"):
                return LMResponse.error_response(str(item.get("error")))
            cc = item.get("chat_completion")
            if not isinstance(cc, dict):
                return LMResponse.error_response("Invalid broker response: missing chat_completion")
            # Map domain ChatCompletion dict -> legacy RLMChatCompletion
            from rlm.domain.models import ChatCompletion as DomainChatCompletion

            domain_cc = DomainChatCompletion.from_dict(cc)
            legacy_cc = RLMChatCompletion(
                root_model=domain_cc.root_model,
                prompt=domain_cc.prompt,  # type: ignore[arg-type]
                response=domain_cc.response,
                usage_summary=UsageSummary.from_dict(domain_cc.usage_summary.to_dict()),
                execution_time=domain_cc.execution_time,
            )
            return LMResponse.success_response(legacy_cc)

        return LMResponse.from_dict(response_data)
    except Exception as e:
        return LMResponse.error_response(f"Request failed: {e}")


def send_lm_request_batched(
    address: tuple[str, int],
    prompts: list[str | dict[str, Any]],
    model: str | None = None,
    timeout: float = DEFAULT_BROKER_CLIENT_TIMEOUT_S,
) -> list[LMResponse]:
    """Send a batched LM request and return a list of typed responses.

    Args:
        address: (host, port) tuple of LM Handler server.
        prompts: List of prompts to send.
        model: Optional model name to use.
        timeout: Socket timeout in seconds.

    Returns:
        List of LMResponse objects, one per prompt, in the same order.
    """
    try:
        request = LMRequest(prompts=prompts, model=model)
        response_data = socket_request(address, request.to_dict(), timeout)
        if not isinstance(response_data, dict):
            return [LMResponse.error_response("Invalid response (expected JSON object)")] * len(
                prompts
            )

        # Compat: wire protocol batched responses.
        if "results" in response_data:
            if response_data.get("error"):
                return [LMResponse.error_response(str(response_data.get("error")))] * len(prompts)
            results = response_data.get("results")
            if not isinstance(results, list) or len(results) != len(prompts):
                return [
                    LMResponse.error_response("Invalid broker response: bad results length")
                ] * len(prompts)

            out: list[LMResponse] = []
            from rlm.domain.models import ChatCompletion as DomainChatCompletion

            for item in results:
                if not isinstance(item, dict):
                    out.append(LMResponse.error_response("Invalid broker result item"))
                    continue
                if item.get("error"):
                    out.append(LMResponse.error_response(str(item.get("error"))))
                    continue
                cc = item.get("chat_completion")
                if not isinstance(cc, dict):
                    out.append(
                        LMResponse.error_response(
                            "Invalid broker response: missing chat_completion"
                        )
                    )
                    continue
                domain_cc = DomainChatCompletion.from_dict(cc)
                legacy_cc = RLMChatCompletion(
                    root_model=domain_cc.root_model,
                    prompt=domain_cc.prompt,  # type: ignore[arg-type]
                    response=domain_cc.response,
                    usage_summary=UsageSummary.from_dict(domain_cc.usage_summary.to_dict()),
                    execution_time=domain_cc.execution_time,
                )
                out.append(LMResponse.success_response(legacy_cc))
            return out

        response = LMResponse.from_dict(response_data)

        if not response.success:
            # Return error responses for all prompts
            return [LMResponse.error_response(response.error)] * len(prompts)

        if response.chat_completions is None:
            return [LMResponse.error_response("No completions returned")] * len(prompts)

        # Convert batched response to list of individual responses
        return [
            LMResponse.success_response(chat_completion)
            for chat_completion in response.chat_completions
        ]
    except Exception as e:
        return [LMResponse.error_response(f"Request failed: {e}")] * len(prompts)
