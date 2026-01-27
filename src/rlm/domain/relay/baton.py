from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeVar, cast, get_origin
from uuid import uuid4

from rlm.domain.errors import ValidationError
from rlm.domain.models.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rlm.domain.relay.budget import TokenBudget

T = TypeVar("T")


class _TypeAdapterProtocol(Protocol):
    def __init__(self, python_type: type[object]) -> None: ...

    def validate_python(self, value: object, /) -> object: ...


_PYDANTIC_CHECKED: bool = False
_TYPE_ADAPTER: type[_TypeAdapterProtocol] | None = None


def _get_pydantic_type_adapter() -> type[_TypeAdapterProtocol] | None:
    global _PYDANTIC_CHECKED, _TYPE_ADAPTER  # noqa: PLW0603

    if not _PYDANTIC_CHECKED:
        try:
            from pydantic import TypeAdapter  # noqa: PLC0415 - optional dep

            _TYPE_ADAPTER = cast("type[_TypeAdapterProtocol]", TypeAdapter)
        except ImportError:
            _TYPE_ADAPTER = None
        _PYDANTIC_CHECKED = True

    return _TYPE_ADAPTER


def has_pydantic() -> bool:
    return _get_pydantic_type_adapter() is not None


def _validate_payload[T](expected_type: type[T], payload: object) -> Result[T, ValidationError]:
    type_adapter_cls = _get_pydantic_type_adapter()
    if type_adapter_cls is None:
        return Err(
            ValidationError("Pydantic is required for baton validation. Install rlm[pydantic].")
        )

    origin = get_origin(expected_type)
    if origin is not None and isinstance(payload, origin):
        return Ok(cast("T", payload))
    if isinstance(expected_type, type) and isinstance(payload, expected_type):
        return Ok(cast("T", payload))  # type: ignore[redundant-cast]

    try:
        adapter = type_adapter_cls(expected_type)
        validated = adapter.validate_python(payload)
        return Ok(cast("T", validated))
    except Exception as exc:  # noqa: BLE001 - surface as domain ValidationError
        return Err(ValidationError(f"Baton payload failed validation: {exc}"))


@dataclass(frozen=True, slots=True)
class BatonMetadata:
    trace_id: str
    created_at: float | None = None
    budget: TokenBudget | None = None
    tokens_consumed: int = 0

    @classmethod
    def create(
        cls,
        *,
        trace_id: str | None = None,
        created_at: float | None = None,
        budget: TokenBudget | None = None,
        tokens_consumed: int = 0,
    ) -> BatonMetadata:
        return cls(
            trace_id=trace_id or uuid4().hex,
            created_at=created_at,
            budget=budget,
            tokens_consumed=tokens_consumed,
        )


@dataclass(frozen=True, slots=True)
class BatonTraceEvent:
    state_name: str
    timestamp: float | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class Baton[T]:
    payload: T
    metadata: BatonMetadata
    trace: tuple[BatonTraceEvent, ...] = field(default_factory=tuple)

    @classmethod
    def create(
        cls,
        payload: object,
        expected_type: type[T],
        /,
        *,
        metadata: BatonMetadata | None = None,
        trace: Sequence[BatonTraceEvent] | None = None,
    ) -> Result[Baton[T], ValidationError]:
        validation_result = _validate_payload(expected_type, payload)
        if isinstance(validation_result, Err):
            return validation_result

        validated = validation_result.unwrap()
        baton = cls(
            payload=validated,
            metadata=metadata or BatonMetadata.create(),
            trace=tuple(trace or ()),
        )
        return Ok(baton)

    def with_trace_event(self, event: BatonTraceEvent) -> Baton[T]:
        return Baton(payload=self.payload, metadata=self.metadata, trace=(*self.trace, event))
