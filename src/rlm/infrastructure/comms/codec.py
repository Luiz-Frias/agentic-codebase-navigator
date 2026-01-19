from __future__ import annotations

import json
import socket
import struct
from typing import Any

from rlm.domain.policies.timeouts import DEFAULT_BROKER_CLIENT_TIMEOUT_S

DEFAULT_MAX_MESSAGE_BYTES = 10_000_000  # 10MB safety cap
_FRAME_LEN_STRUCT = struct.Struct(">I")  # 4-byte big-endian unsigned int
_FRAME_LEN_SIZE = _FRAME_LEN_STRUCT.size  # 4 bytes
_JSON_ENCODER = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))


def encode_frame(message: dict[str, Any], /) -> bytes:
    """Encode a JSON object into a length-prefixed frame."""
    payload = _JSON_ENCODER.encode(message).encode("utf-8")
    return _FRAME_LEN_STRUCT.pack(len(payload)) + payload


def _recv_exact(sock: socket.socket, n: int, /) -> bytes:
    """Receive exactly n bytes or raise ConnectionError if the peer closes early."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed before message complete")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(
    sock: socket.socket,
    /,
    *,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> dict[str, Any] | None:
    """
    Receive a single length-prefixed JSON object from a socket.

    Returns:
        - dict: parsed JSON object
        - None: connection closed cleanly before any length prefix was read

    """
    raw_len = bytearray()
    while len(raw_len) < _FRAME_LEN_SIZE:
        chunk = sock.recv(_FRAME_LEN_SIZE - len(raw_len))
        if not chunk:
            # Connection closed before we received a full length prefix.
            return None
        raw_len.extend(chunk)

    length = _FRAME_LEN_STRUCT.unpack(raw_len)[0]
    if length > max_message_bytes:
        raise ValueError(f"Frame too large: {length} bytes (max {max_message_bytes})")

    payload = _recv_exact(sock, length)
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Frame JSON must be an object")
    return value


def send_frame(sock: socket.socket, message: dict[str, Any], /) -> None:
    """Send a single length-prefixed JSON object to a socket."""
    sock.sendall(encode_frame(message))


def request_response(
    address: tuple[str, int],
    message: dict[str, Any],
    /,
    *,
    timeout_s: float = DEFAULT_BROKER_CLIENT_TIMEOUT_S,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> dict[str, Any]:
    """Open a TCP connection, send a single request frame, then read one response frame."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        sock.connect(address)
        send_frame(sock, message)
        response = recv_frame(sock, max_message_bytes=max_message_bytes)
        if response is None:
            raise ConnectionError("Connection closed before response frame")
        return response
