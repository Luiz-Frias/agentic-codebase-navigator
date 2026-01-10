from __future__ import annotations

import pytest

from rlm.infrastructure.ids import Uuid4IdGenerator


@pytest.mark.unit
def test_uuid4_id_generator_returns_unique_strings() -> None:
    gen = Uuid4IdGenerator()
    a = gen.new_id()
    b = gen.new_id()
    assert isinstance(a, str) and isinstance(b, str)
    assert a != b


@pytest.mark.unit
def test_uuid4_id_generator_prefix() -> None:
    gen = Uuid4IdGenerator(prefix="run")
    value = gen.new_id()
    assert value.startswith("run_")
