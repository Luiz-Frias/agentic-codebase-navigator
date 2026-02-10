from __future__ import annotations

import pytest

from rlm.domain.errors import ValidationError
from rlm.domain.relay import StateSpec


@pytest.mark.unit
def test_state_spec_rejects_blank_name() -> None:
    with pytest.raises(ValidationError):
        StateSpec(name=" ", input_type=str, output_type=str)


@pytest.mark.unit
def test_state_spec_rejects_non_types() -> None:
    with pytest.raises(ValidationError):
        StateSpec(name="ok", input_type="str", output_type=str)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        StateSpec(name="ok", input_type=str, output_type="str")  # type: ignore[arg-type]


@pytest.mark.unit
def test_state_spec_allows_valid_types() -> None:
    spec = StateSpec(name="ok", input_type=str, output_type=int)
    assert spec.name == "ok"
    assert spec.input_type is str
    assert spec.output_type is int
