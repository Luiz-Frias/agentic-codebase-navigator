from __future__ import annotations

import pytest

from rlm.domain.errors import ValidationError
from rlm.domain.models.result import Err, Ok
from rlm.domain.relay import Baton, BatonMetadata, BatonTraceEvent, has_pydantic
import rlm.domain.relay.baton as baton_module


@pytest.mark.unit
def test_baton_create_returns_error_when_pydantic_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baton_module, "_get_pydantic_type_adapter", lambda: None)

    result = Baton.create({"value": "data"}, dict[str, object])

    assert isinstance(result, Err)
    assert isinstance(result.error, ValidationError)
    assert "pydantic" in str(result.error).lower()


@pytest.mark.unit
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_baton_create_respects_metadata_and_trace() -> None:
    metadata = BatonMetadata(trace_id="trace-123", created_at=1.0)
    trace = (BatonTraceEvent(state_name="start", timestamp=1.5),)

    result = Baton.create({"value": "data"}, object, metadata=metadata, trace=trace)

    assert isinstance(result, Ok)
    baton = result.value
    assert baton.metadata == metadata
    assert baton.trace == trace


@pytest.mark.unit
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_baton_create_validates_payload() -> None:
    from pydantic import BaseModel

    class Payload(BaseModel):
        name: str

    ok_result = Baton.create({"name": "hello"}, Payload)
    assert isinstance(ok_result, Ok)
    assert isinstance(ok_result.value.payload, Payload)

    err_result = Baton.create({"name": 123}, Payload)
    assert isinstance(err_result, Err)
    assert isinstance(err_result.error, ValidationError)
