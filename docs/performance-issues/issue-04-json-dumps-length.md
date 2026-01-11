# perf: Multiple json.dumps() calls for length measurement

## Summary

The `QueryMetadata.from_context()` method calls `json.dumps()` multiple times just to measure the serialized length of context items, causing redundant serialization overhead.

## Location

`src/rlm/domain/models/query_metadata.py` lines 30-98

## Problem

```python
# Lines 48-55 (dict context)
for key, value in context.items():
    try:
        lengths.append(len(json.dumps(chunk, default=str)))  # Full serialization just for length
    except Exception:
        lengths.append(0)

# Lines 63-96 (list context) - similar pattern
lengths.append(len(json.dumps(chunk, default=str)))
```

Each context item is fully JSON-serialized just to count characters, then discarded. For large contexts with many items, this creates significant overhead.

## Impact

- **CPU**: O(n) serialization work for n context items
- **Memory**: Temporary JSON strings allocated and immediately discarded
- **Latency**: Adds overhead to every orchestrator initialization

## Suggested Fixes

1. **Lazy Length Estimation**: Estimate length without full serialization
2. **Cache Serialized**: If serialization is needed later, cache the result
3. **Sampling**: For large contexts, sample items instead of measuring all
4. **Remove Feature**: If length metadata isn't critical, remove this computation

## Example Fix

```python
def _estimate_length(value: Any) -> int:
    """Estimate serialized length without full JSON encoding."""
    if isinstance(value, str):
        return len(value) + 2  # quotes
    if isinstance(value, (int, float, bool, type(None))):
        return len(str(value))
    if isinstance(value, (list, tuple)):
        return 2 + sum(_estimate_length(v) for v in value) + len(value)  # brackets + commas
    if isinstance(value, dict):
        return 2 + sum(len(k) + 4 + _estimate_length(v) for k, v in value.items())
    return 20  # fallback estimate
```

## Benchmarks

See `tests/performance/test_speed_serialization.py::test_query_metadata_from_context_*`

## Severity

**High** - Affects initialization performance for all runs with context
