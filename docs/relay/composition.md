# Composition

Relay pipelines are built by composing `StateSpec` instances with a small DSL.
The resulting `Pipeline` can be validated and executed with sync/async executors.

## Sequential (`>>`)

```python
pipeline = research >> analyze >> summarize
```

## Parallel (`|`) and Join

Use `|` to create parallel branches and `.join()` to configure merge behavior.

```python
pipeline = (analyze | fact_check).join(mode="all") >> synthesize
```

Join modes:

- `all`: waits for all branches, aggregates outputs into a dict keyed by state name.
- `race`: returns the first completed branch output.

For `join(mode="all")`, the join target must accept `dict` as input.

## Conditional Routing (`when` / `otherwise`)

```python
pipeline = start.when(lambda baton: baton.payload.score > 0.8) >> high_confidence
pipeline = pipeline.otherwise(low_confidence)
```

Guards receive the output baton from the originating state. If no guard matches,
the default branch (`otherwise`) is used when provided.
