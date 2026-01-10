from __future__ import annotations

import json

import pytest

from rlm.domain.models.query_metadata import QueryMetadata


@pytest.mark.unit
def test_query_metadata_str() -> None:
    md = QueryMetadata.from_context("abc")
    assert md.context_type == "str"
    assert md.context_lengths == [3]
    assert md.context_total_length == 3


@pytest.mark.unit
def test_query_metadata_dict_matches_legacy() -> None:
    from rlm._legacy.core.types import QueryMetadata as LegacyQueryMetadata

    payload = {"a": "hi", "b": {"x": 1}}
    md = QueryMetadata.from_context(payload)
    legacy = LegacyQueryMetadata(payload)

    assert md.context_type == "dict"
    assert md.context_lengths == [2, len(json.dumps({"x": 1}, default=str))]
    assert md.context_total_length == sum(md.context_lengths)
    assert md.context_lengths == legacy.context_lengths
    assert md.context_total_length == legacy.context_total_length
    assert md.context_type == legacy.context_type


@pytest.mark.unit
def test_query_metadata_empty_list() -> None:
    md = QueryMetadata.from_context([])
    assert md.context_type == "list"
    assert md.context_lengths == [0]
    assert md.context_total_length == 0


@pytest.mark.unit
def test_query_metadata_list_of_messages_content_lengths() -> None:
    payload = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    md = QueryMetadata.from_context(payload)
    assert md.context_type == "list"
    assert md.context_lengths == [2, 2]
    assert md.context_total_length == 4


@pytest.mark.unit
def test_query_metadata_list_of_dicts_without_content_matches_legacy() -> None:
    from rlm._legacy.core.types import QueryMetadata as LegacyQueryMetadata

    payload = [{"x": 1}, {"y": "abc"}]
    md = QueryMetadata.from_context(payload)
    legacy = LegacyQueryMetadata(payload)

    assert md.context_type == "list"
    assert md.context_lengths == [
        len(json.dumps({"x": 1}, default=str)),
        len(json.dumps({"y": "abc"}, default=str)),
    ]
    assert md.context_total_length == sum(md.context_lengths)
    assert md.context_lengths == legacy.context_lengths
    assert md.context_total_length == legacy.context_total_length
    assert md.context_type == legacy.context_type


@pytest.mark.unit
def test_query_metadata_list_of_strings_matches_legacy() -> None:
    from rlm._legacy.core.types import QueryMetadata as LegacyQueryMetadata

    payload = ["a", "bb"]
    md = QueryMetadata.from_context(payload)
    legacy = LegacyQueryMetadata(payload)

    assert md.context_type == "list"
    assert md.context_lengths == [1, 2]
    assert md.context_total_length == 3
    assert md.context_lengths == legacy.context_lengths
    assert md.context_total_length == legacy.context_total_length
    assert md.context_type == legacy.context_type


@pytest.mark.unit
def test_query_metadata_invalid_type_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Invalid context type"):
        QueryMetadata.from_context(123)  # type: ignore[arg-type]
