# perf: Regex with DOTALL on large response text

## Summary

The code block extraction regex uses `re.DOTALL` flag which can be slower on large response texts, especially with multiple potential match sites.

## Location

`src/rlm/domain/services/parsing.py` lines 21-22

## Problem

```python
def find_code_blocks(text: str) -> list[str]:
    pattern = r"```repl\s*\n(.*?)\n```"
    return [m.group(1).strip() for m in re.finditer(pattern, text, re.DOTALL)]
```

With `re.DOTALL`, the `.` matches newlines, making the `.*?` pattern potentially slower as it must consider more possibilities. For LLM responses with many backtick sequences, this can cause performance degradation.

## Impact

- **CPU**: Regex engine does more backtracking with DOTALL
- **Scaling**: Performance degrades with response length
- **Latency**: Adds overhead to every iteration

## Benchmarks

Current performance (from tests):
- 1KB response: ~0.5ms per extraction
- 10KB response: ~2ms per extraction
- 100KB response: ~15ms per extraction

## Suggested Fixes

1. **Specific Pattern**: Use `[^\`]` instead of `.` to avoid matching backticks
2. **Two-Phase Parse**: First find ``` markers, then extract content
3. **Compiled Regex**: Pre-compile the pattern for reuse
4. **Streaming Parse**: Process response incrementally instead of full regex

## Example Fix

```python
# Pre-compiled pattern without DOTALL
_CODE_BLOCK_PATTERN = re.compile(r"```repl\s*\n([^`]*(?:`(?!``)[^`]*)*)\n```")

def find_code_blocks(text: str) -> list[str]:
    return [m.group(1).strip() for m in _CODE_BLOCK_PATTERN.finditer(text)]
```

## Benchmarks

See `tests/performance/test_speed_parsing.py::test_find_code_blocks_scales_with_text_size`

## Severity

**High** - Affects every iteration for large LLM responses
