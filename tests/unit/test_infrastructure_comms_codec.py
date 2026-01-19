from __future__ import annotations

import json
import struct

import pytest

from rlm.infrastructure.comms.codec import recv_frame


class _FragmentingSocket:
    """
    Minimal socket-like object that returns data in pre-chunked fragments.

    This deterministically simulates TCP fragmentation where `recv(n)` may return
    fewer than `n` bytes, including for the 4-byte length prefix.
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = [bytearray(c) for c in chunks]

    def recv(self, n: int) -> bytes:
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
def test_recv_frame_handles_partial_length_prefix_reads() -> None:
    msg = {"ok": True, "value": 123}
    payload = json.dumps(msg).encode("utf-8")
    prefix = struct.pack(">I", len(payload))

    sock = _FragmentingSocket(
        chunks=[
            prefix[:2],
            prefix[2:],
            payload[:1],
            payload[1:5],
            payload[5:],
        ],
    )

    assert recv_frame(sock) == msg


@pytest.mark.unit
def test_recv_frame_returns_none_if_connection_closes_mid_length_prefix() -> None:
    msg = {"x": "y"}
    payload = json.dumps(msg).encode("utf-8")
    prefix = struct.pack(">I", len(payload))

    sock = _FragmentingSocket(chunks=[prefix[:2]])  # closes before full 4-byte prefix
    assert recv_frame(sock) is None


@pytest.mark.unit
def test_recv_frame_raises_if_connection_closes_mid_payload() -> None:
    msg = {"hello": "world"}
    payload = json.dumps(msg).encode("utf-8")
    prefix = struct.pack(">I", len(payload))

    sock = _FragmentingSocket(chunks=[prefix, payload[:3]])  # closes before payload complete
    with pytest.raises(ConnectionError, match="before message complete"):
        recv_frame(sock)


@pytest.mark.unit
def test_recv_frame_rejects_non_object_json() -> None:
    payload = json.dumps(["not", "an", "object"]).encode("utf-8")
    prefix = struct.pack(">I", len(payload))

    sock = _FragmentingSocket(chunks=[prefix, payload])
    with pytest.raises(TypeError, match="must be an object"):
        recv_frame(sock)


@pytest.mark.unit
def test_recv_frame_rejects_overly_large_frames() -> None:
    msg = {"x": "y"}
    payload = json.dumps(msg).encode("utf-8")

    # Lie in the length prefix so the decoder rejects before allocating.
    prefix = struct.pack(">I", 999)
    sock = _FragmentingSocket(chunks=[prefix, payload])

    with pytest.raises(ValueError, match="Frame too large"):
        recv_frame(sock, max_message_bytes=10)
