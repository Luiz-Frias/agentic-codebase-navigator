from __future__ import annotations

import socket
import threading

import pytest

from rlm.infrastructure.comms.codec import request_response


@pytest.mark.unit
def test_request_response_raises_if_server_closes_without_response_frame() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    host, port = server.getsockname()

    def _serve_and_close() -> None:
        try:
            conn, _addr = server.accept()
        except OSError:
            return
        with conn:
            # Read a little (client sends a length-prefixed request) then close.
            try:
                conn.recv(1024)
            except OSError:
                pass
        server.close()

    t = threading.Thread(target=_serve_and_close, daemon=True)
    t.start()

    with pytest.raises(ConnectionError, match="before response frame"):
        request_response((host, port), {"hello": "world"}, timeout_s=0.2)

    t.join(timeout=1)
