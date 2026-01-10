from __future__ import annotations

import json
import struct

import pytest

from rlm._legacy.core.comms_utils import socket_recv


class _FragmentingSocket:
    """
    Minimal socket-like object that returns data in pre-chunked fragments.

    This lets us deterministically simulate TCP fragmentation where `recv(n)` may
    return fewer than `n` bytes (including for the 4-byte length prefix).
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = [bytearray(c) for c in chunks]

    def recv(self, n: int) -> bytes:  # noqa: D401 - matches socket API
        while self._chunks:
            buf = self._chunks[0]
            if not buf:
                self._chunks.pop(0)
                continue

            out = bytes(buf[:n])
            del buf[:n]
            if not buf:
                self._chunks.pop(0)
            return out

        return b""


@pytest.mark.unit
def test_socket_recv_handles_partial_length_prefix_reads() -> None:
    msg = {"ok": True, "value": 123}
    payload = json.dumps(msg).encode("utf-8")
    prefix = struct.pack(">I", len(payload))

    # Fragment both the prefix and payload. Old code would raise struct.error on
    # the first recv(4) returning only 2 bytes.
    sock = _FragmentingSocket(
        chunks=[
            prefix[:2],
            prefix[2:],
            payload[:1],
            payload[1:5],
            payload[5:],
        ]
    )

    assert socket_recv(sock) == msg


@pytest.mark.unit
def test_socket_recv_returns_empty_dict_if_connection_closes_mid_length_prefix() -> None:
    msg = {"x": "y"}
    payload = json.dumps(msg).encode("utf-8")
    prefix = struct.pack(">I", len(payload))

    sock = _FragmentingSocket(chunks=[prefix[:2]])  # closes before full 4-byte prefix
    assert socket_recv(sock) == {}


@pytest.mark.unit
def test_socket_recv_raises_if_connection_closes_mid_payload() -> None:
    msg = {"hello": "world"}
    payload = json.dumps(msg).encode("utf-8")
    prefix = struct.pack(">I", len(payload))

    sock = _FragmentingSocket(chunks=[prefix, payload[:3]])  # closes before payload complete
    with pytest.raises(ConnectionError, match="before message complete"):
        socket_recv(sock)
