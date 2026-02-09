# Validation

Relay validates pipelines at definition time to surface errors early.

```python
pipeline.validate()
```

## Checks Performed

- **Type compatibility**: each edge must connect compatible output/input types.
- **Join rules**:
  - `join(mode="all")` aggregates into a `dict`, so the target input type must be `dict`.
  - `join(mode="race")` requires each branch output to be compatible with the target input type.
- **Reachability**: every state must be reachable from the entry state.
- **Terminal states**: at least one terminal state is required.
- **Cycles**: cycles are rejected unless explicitly allowed.

## Allowing Cycles

If you have an intentional loop, annotate the pipeline using `allow_cycles`:

```python
from rlm.domain.relay.validation import allow_cycles

pipeline = allow_cycles(max_iterations=3)(pipeline)
pipeline.validate()
```

When cycles are allowed, `max_iterations` must be a positive integer.
