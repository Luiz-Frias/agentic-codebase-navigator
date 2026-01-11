# Performance Hotspot Issues

This directory contains GitHub issue templates for the identified performance hotspots.
These can be created manually or via the `gh` CLI when authenticated.

## Quick Create Script

Once you have `gh` authenticated, run:

```bash
# Create all issues
for f in docs/performance-issues/issue-*.md; do
  title=$(head -1 "$f" | sed 's/^# //')
  body=$(tail -n +3 "$f")
  gh issue create --title "$title" --body "$body" --label "performance"
done
```

## Issues

### Critical (High Impact)

1. **issue-01-message-history-growth.md** - O(n²) token usage from message history accumulation
2. **issue-02-sequential-code-execution.md** - Sequential code block execution blocks parallelization
3. **issue-03-recursive-serialization.md** - Recursive serialization without depth limit

### High (Significant Impact)

4. **issue-04-json-dumps-length.md** - Multiple json.dumps() for length measurement
5. **issue-05-global-execution-lock.md** - Global process execution lock limits concurrency
6. **issue-06-regex-dotall-large-text.md** - Regex with DOTALL on large response text
