from __future__ import annotations

import random
import struct

import pytest

from rlm.infrastructure.comms.codec import recv_frame


class _FragmentingSocket:
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
def test_recv_frame_fuzz_does_not_hang_or_crash_unexpectedly() -> None:
    rng = random.Random(0)

    # A small, deterministic fuzz loop that focuses on "invalid-but-plausible" frames:
    # - valid length prefix
    # - random payload bytes
    # - random fragmentation patterns
    for _ in range(250):
        payload_len = rng.randint(0, 64)
        payload = bytes(rng.getrandbits(8) for _ in range(payload_len))
        prefix = struct.pack(">I", len(payload))

        # Randomly fragment prefix+payload into small chunks.
        stream = prefix + payload
        chunks: list[bytes] = []
        i = 0
        while i < len(stream):
            step = rng.randint(1, 5)
            chunks.append(stream[i : i + step])
            i += step

        sock = _FragmentingSocket(chunks=chunks)
        try:
            msg = recv_frame(sock, max_message_bytes=1024)
            assert isinstance(msg, dict)
        except (ConnectionError, ValueError, TypeError):
            # Any decode/validation failure is acceptable as long as we don't
            # hang or allocate unbounded memory.
            pass
