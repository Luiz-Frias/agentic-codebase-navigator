# Relay Overview

Relay is a type-safe pipeline library for orchestrating agent workflows in RLM. It emphasizes
explicit baton passing between states, definition-time validation, and a “library-not-framework”
execution model.

## Philosophy

- **Library, not framework**: Relay yields steps; you own the execution loop.
- **Typed batons**: Payloads crossing state boundaries are validated at runtime (Pydantic).
- **Definition-time checks**: Cycles, reachability, and type mismatches are caught early.
- **Composable**: Pipelines can be wrapped as states and reused across workflows.

## Quick Start

```python
from pydantic import BaseModel

from rlm.adapters.relay import FunctionStateExecutor, SyncPipelineExecutor
from rlm.domain.relay import Baton, StateSpec


class Query(BaseModel):
    text: str


class Answer(BaseModel):
    text: str


def _answer(query: Query) -> Answer:
    return Answer(text=f"Echo: {query.text}")


start = StateSpec(
    name="answer",
    input_type=Query,
    output_type=Answer,
    executor=FunctionStateExecutor(_answer),
)

pipeline = start
initial = Baton.create(Query(text="hello"), Query).unwrap()

executor = SyncPipelineExecutor(pipeline, initial)
last = None
for step in executor:
    result = step.state.executor.execute(step.state, step.baton)
    executor.advance(result)
    if result.is_ok():
        last = result.value

print(last.payload.text if last else "no result")
```

## When to Use Relay

- You want strict baton typing between steps.
- You need deterministic, debuggable orchestration with validation up front.
- You want to combine RLM orchestrators, direct LLM calls, and pure functions in one graph.
