from __future__ import annotations

import pytest

from rlm.domain.errors import ValidationError
from rlm.domain.models.model_spec import ModelSpec, build_routing_rules


@pytest.mark.unit
def test_model_spec_validations() -> None:
    with pytest.raises(ValidationError, match="name must be a non-empty string"):
        ModelSpec(name="")  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="aliases must be a tuple"):
        ModelSpec(name="m", aliases=["a"])  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="aliases must contain only non-empty strings"):
        ModelSpec(name="m", aliases=(" ",))


@pytest.mark.unit
def test_model_routing_rules_validations_and_resolution() -> None:
    with pytest.raises(ValidationError, match="must be a non-empty tuple"):
        build_routing_rules([])

    with pytest.raises(ValidationError, match="must contain only ModelSpec"):
        build_routing_rules([ModelSpec(name="m", is_default=True), "nope"])  # type: ignore[list-item]

    with pytest.raises(ValidationError, match="Duplicate model name"):
        build_routing_rules(
            [
                ModelSpec(name="m1", aliases=("m2",), is_default=True),
                ModelSpec(name="m2"),
            ]
        )

    with pytest.raises(ValidationError, match="ambiguous"):
        build_routing_rules(
            [
                ModelSpec(name="m1", aliases=("x",), is_default=True),
                ModelSpec(name="m2", aliases=("x",), is_default=False),
            ]
        )

    with pytest.raises(ValidationError, match="Exactly one ModelSpec"):
        build_routing_rules(
            [
                ModelSpec(name="m1", is_default=True),
                ModelSpec(name="m2", is_default=True),
            ]
        )

    with pytest.raises(ValidationError, match="requires exactly one default"):
        build_routing_rules([ModelSpec(name="m1", is_default=False)])

    with pytest.raises(ValidationError, match="fallback_model must be a non-empty string"):
        build_routing_rules([ModelSpec(name="m1", is_default=True)], fallback_model=" ")  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="fallback_model .* is not in allowed models"):
        build_routing_rules([ModelSpec(name="m1", is_default=True)], fallback_model="m2")

    rules = build_routing_rules(
        [ModelSpec(name="m1", aliases=("alias",), is_default=True)], fallback_model="m1"
    )
    assert rules.default_model == "m1"
    assert rules.resolve(None) == "m1"
    assert rules.resolve("") == "m1"
    assert rules.resolve("alias") == "m1"
    assert rules.resolve("m1") == "m1"

    with pytest.raises(ValidationError, match="Requested model must be a string"):
        rules.resolve(123)  # type: ignore[arg-type]
