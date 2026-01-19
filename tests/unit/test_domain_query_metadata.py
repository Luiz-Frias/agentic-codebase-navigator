from __future__ import annotations

import pytest

from rlm.domain.models.query_metadata import QueryMetadata


@pytest.mark.unit
def test_query_metadata_from_context_str() -> None:
    md = QueryMetadata.from_context("hello")
    assert md.context_type == "str"
    assert md.context_lengths == [5]
    assert md.context_total_length == 5


@pytest.mark.unit
def test_query_metadata_from_context_dict_json_and_fallback_repr() -> None:
    md0 = QueryMetadata.from_context({"a": "hello"})
    assert md0.context_type == "dict"
    assert md0.context_lengths == [5]
    assert md0.context_total_length == 5

    # Normal json.dumps path (default=str).
    md = QueryMetadata.from_context({"a": {"x": 1}})
    assert md.context_type == "dict"
    assert md.context_total_length > 0

    # Force json.dumps failure via a circular reference.
    circular: list[object] = []
    circular.append(circular)
    md2 = QueryMetadata.from_context({"a": circular})
    assert md2.context_type == "dict"
    assert md2.context_total_length > 0


@pytest.mark.unit
def test_query_metadata_from_context_list_empty() -> None:
    md = QueryMetadata.from_context([])
    assert md.context_type == "list"
    assert md.context_lengths == [0]
    assert md.context_total_length == 0


@pytest.mark.unit
def test_query_metadata_from_context_message_list_with_content() -> None:
    md = QueryMetadata.from_context(
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "bb"}],
    )
    assert md.context_type == "list"
    assert md.context_lengths == [1, 2]
    assert md.context_total_length == 3


@pytest.mark.unit
def test_query_metadata_from_context_list_of_dict_without_content_uses_json_or_repr() -> None:
    md = QueryMetadata.from_context([{"x": 1}, {"y": 2}])
    assert md.context_type == "list"
    assert md.context_total_length > 0

    circular: list[object] = []
    circular.append(circular)
    md2 = QueryMetadata.from_context([{"x": circular}])
    assert md2.context_type == "list"
    assert md2.context_total_length > 0


@pytest.mark.unit
def test_query_metadata_from_context_list_of_strings() -> None:
    md = QueryMetadata.from_context(["a", "bb"])
    assert md.context_type == "list"
    assert md.context_lengths == [1, 2]
    assert md.context_total_length == 3


@pytest.mark.unit
def test_query_metadata_from_context_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match="Invalid context type"):
        QueryMetadata.from_context(123)  # type: ignore[arg-type]
