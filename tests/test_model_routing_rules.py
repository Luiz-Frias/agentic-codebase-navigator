from __future__ import annotations

import pytest

from rlm.domain.errors import ValidationError
from rlm.domain.models import ModelSpec, build_routing_rules


@pytest.mark.unit
def test_model_routing_rules_resolves_default_for_none_or_blank_and_allows_aliases() -> None:
    rules = build_routing_rules(
        [
            ModelSpec(name="root", aliases=("default", "main"), is_default=True),
            ModelSpec(name="sub"),
        ]
    )

    assert rules.resolve(None) == "root"
    assert rules.resolve("") == "root"
    assert rules.resolve("   ") == "root"
    assert rules.resolve("default") == "root"
    assert rules.resolve("main") == "root"
    assert rules.resolve("sub") == "sub"


@pytest.mark.unit
def test_model_routing_rules_unknown_model_raises_validation_error_with_allowed_models() -> None:
    rules = build_routing_rules(
        [
            ModelSpec(name="root", is_default=True),
            ModelSpec(name="sub"),
        ]
    )

    with pytest.raises(ValidationError, match="Unknown model"):
        rules.resolve("nope")


@pytest.mark.unit
def test_model_routing_rules_can_fallback_when_configured() -> None:
    rules = build_routing_rules(
        [
            ModelSpec(name="root", is_default=True),
            ModelSpec(name="sub"),
        ],
        fallback_model="root",
    )

    assert rules.resolve("nope") == "root"


@pytest.mark.unit
def test_model_routing_rules_rejects_ambiguous_aliases() -> None:
    with pytest.raises(ValidationError, match="ambiguous"):
        build_routing_rules(
            [
                ModelSpec(name="a", aliases=("x",), is_default=True),
                ModelSpec(name="b", aliases=("x",)),
            ]
        )


@pytest.mark.unit
def test_model_routing_rules_requires_exactly_one_default() -> None:
    with pytest.raises(ValidationError, match="default"):
        build_routing_rules([ModelSpec(name="a"), ModelSpec(name="b")])

    with pytest.raises(ValidationError, match="default"):
        build_routing_rules(
            [ModelSpec(name="a", is_default=True), ModelSpec(name="b", is_default=True)]
        )
