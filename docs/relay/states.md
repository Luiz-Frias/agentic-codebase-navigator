# States and Executors

Relay states are described by `StateSpec[InputT, OutputT]`, which pairs a name,
input/output types, and an executor that implements `StateExecutorPort`.

```python
from rlm.domain.relay import StateSpec

state = StateSpec(
    name="step",
    input_type=InputType,
    output_type=OutputType,
    executor=some_executor,
)
```

## Built-in Executors

### FunctionStateExecutor

Runs a pure Python function and wraps the output as a validated baton.

```python
from rlm.adapters.relay import FunctionStateExecutor

state = StateSpec(
    name="normalize",
    input_type=str,
    output_type=str,
    executor=FunctionStateExecutor(lambda text: text.strip()),
)
```

### AsyncStateExecutor

Runs an async callable (outside any running event loop) and returns a baton.

```python
from rlm.adapters.relay import AsyncStateExecutor

async def enrich(text: str) -> str:
    return text.upper()

state = StateSpec(
    name="enrich",
    input_type=str,
    output_type=str,
    executor=AsyncStateExecutor(enrich),
)
```

### LLMStateExecutor

Executes a single LLM call via an `LLMPort`. You can provide a request builder
to map the input payload into an `LLMRequest`.

```python
from rlm.adapters.relay import LLMStateExecutor
from rlm.domain.models import LLMRequest

state = StateSpec(
    name="draft",
    input_type=str,
    output_type=ChatCompletion,
    executor=LLMStateExecutor(
        llm=llm_adapter,
        request_builder=lambda text: LLMRequest(prompt=text),
    ),
)
```

### RLMStateExecutor

Runs a full `RLMOrchestrator` (code or tools mode) as a pipeline state.

```python
from rlm.adapters.relay import RLMStateExecutor

state = StateSpec(
    name="agent",
    input_type=str,
    output_type=ChatCompletion,
    executor=RLMStateExecutor(orchestrator=orch, max_iterations=5),
)
```

### PipelineStateExecutor

Wraps an existing pipeline as a state, allowing composition of pipelines.

```python
from rlm.adapters.relay.states import SyncPipelineStateExecutor

nested_state = pipeline.as_state(
    name="nested",
    input_type=str,
    output_type=str,
    executor=SyncPipelineStateExecutor(pipeline),
)
```
