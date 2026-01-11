"""
Infrastructure comms (wire protocol).

This package defines the **wire DTOs** and **codec helpers** used by transport
adapters (e.g., the TCP broker).

Design goals:
- dependency-free (stdlib + `rlm.domain` only)
- strict input validation
- stable JSON shapes (length-prefixed frames at the transport level)
"""

from __future__ import annotations

from rlm.infrastructure.comms.messages import WireRequest, WireResponse, WireResult

__all__ = ["WireRequest", "WireResponse", "WireResult"]
