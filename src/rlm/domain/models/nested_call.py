from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NestedCallResponse:
    handled: bool
    response: str | None = None

    @staticmethod
    def not_handled() -> NestedCallResponse:
        return NestedCallResponse(handled=False, response=None)

    @staticmethod
    def handled_response(response: str) -> NestedCallResponse:
        return NestedCallResponse(handled=True, response=response)
