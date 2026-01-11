# perf: Recursive serialization without depth limit risks stack overflow

## Summary

The `serialize_value()` function recursively processes nested structures without a depth limit, risking stack overflow on deeply nested data.

## Location

`src/rlm/domain/models/serialization.py` lines 7-22

## Problem

```python
def serialize_value(value: Any) -> Any:
    # ...
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]  # Recursive call
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}  # Recursive call
    # ...
```

For deeply nested structures (depth > ~500-1000 depending on Python's recursion limit), this will raise `RecursionError`.

## Impact

- **Crash Risk**: Deep data structures cause stack overflow
- **Memory**: Each recursion level allocates stack frame
- **Performance**: No memoization means repeated objects serialized multiple times

## Suggested Fixes

1. **Depth Limit**: Add max_depth parameter with default (e.g., 100)
2. **Iterative Approach**: Rewrite using explicit stack instead of recursion
3. **Memoization**: Cache already-serialized objects to avoid re-processing
4. **Truncation**: Truncate deeply nested structures with placeholder

## Example Fix

```python
def serialize_value(value: Any, *, max_depth: int = 100, _depth: int = 0) -> Any:
    if _depth >= max_depth:
        return f"<nested depth {_depth} exceeded>"
    # ... rest of implementation with _depth + 1 passed to recursive calls
```

## Benchmarks

See `tests/performance/test_speed_serialization.py::test_serialize_value_recursion_depth`

## Severity

**Critical** - Can crash the application on valid but deeply nested input
