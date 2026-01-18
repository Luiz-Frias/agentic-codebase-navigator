from __future__ import annotations

import typing

import pytest

from rlm.adapters.tools.native import NativeToolAdapter, _python_type_to_json_schema


@pytest.mark.unit
def test_python_type_to_json_schema_handles_optional_and_union() -> None:
    assert _python_type_to_json_schema(typing.Optional[int]) == {"type": "integer"}  # noqa: UP045
    assert _python_type_to_json_schema(int | None) == {"type": "integer"}
    assert _python_type_to_json_schema(typing.Union[int, str]) == {  # noqa: UP007
        "anyOf": [{"type": "integer"}, {"type": "string"}],
    }


@pytest.mark.unit
async def test_native_tool_adapter_aexecute_runs_sync_once() -> None:
    calls = {"count": 0}

    def add_one(value: int) -> int:
        calls["count"] += 1
        return value + 1

    adapter = NativeToolAdapter(add_one)
    result = await adapter.aexecute(value=1)

    assert result == 2
    assert calls["count"] == 1


@pytest.mark.unit
def test_native_tool_adapter_execute_rejects_coroutine_result() -> None:
    async def inner() -> int:
        return 1

    def returns_coroutine() -> object:
        return inner()

    adapter = NativeToolAdapter(returns_coroutine)
    with pytest.raises(TypeError, match="returned a coroutine"):
        adapter.execute()


@pytest.mark.unit
async def test_native_tool_adapter_aexecute_rejects_coroutine_result() -> None:
    async def inner() -> int:
        return 1

    def returns_coroutine() -> object:
        return inner()

    adapter = NativeToolAdapter(returns_coroutine)
    with pytest.raises(TypeError, match="returned a coroutine"):
        await adapter.aexecute()
