# Extending RLM with Custom Policies

RLM provides extension protocols that allow external applications to customize
the orchestrator's behavior without modifying the core library. This is useful
for building debugging tools, implementing custom stopping criteria, managing
context budgets, and enabling recursive orchestration.

## Overview

Three extension protocols are available:

| Protocol | Purpose | Default Implementation |
|----------|---------|----------------------|
| `StoppingPolicy` | Control when iteration loops terminate | `DefaultStoppingPolicy` |
| `ContextCompressor` | Compress nested call results | `NoOpContextCompressor` |
| `NestedCallPolicy` | Control nested orchestrator spawning | `SimpleNestedCallPolicy` |

All protocols use Python's `@runtime_checkable` Protocol pattern, meaning you
can implement them with any class that has the required methods (duck typing).
No inheritance is required.

## StoppingPolicy

Controls when the orchestrator's iteration loop should terminate.

### Protocol Definition

```python
from typing import Any
from rlm.domain.models import ChatCompletion

class StoppingPolicy(Protocol):
    def should_stop(self, context: dict[str, Any]) -> bool:
        """Return True to stop the iteration loop early."""
        ...

    def on_iteration_complete(
        self,
        context: dict[str, Any],
        result: ChatCompletion
    ) -> None:
        """Called after each iteration for state tracking."""
        ...
```

### Context Dictionary

The `context` dictionary contains orchestrator state:

- `iteration`: Current iteration number (0-indexed)
- `max_iterations`: Maximum iterations allowed
- `depth`: Current recursion depth
- `agent_mode`: Either "code" or "tools"
- `correlation_id`: Unique run identifier

### Example: EIG-Gated Stopping

```python
from dataclasses import dataclass, field
from typing import Any
from rlm import RLM, ChatCompletion

@dataclass
class EIGStoppingPolicy:
    """Stop when Expected Information Gain falls below threshold."""

    eig_threshold: float = 0.1
    window_size: int = 3
    recent_responses: list[str] = field(default_factory=list)

    def should_stop(self, context: dict[str, Any]) -> bool:
        if len(self.recent_responses) < self.window_size:
            return False

        # Simple heuristic: stop if recent responses are too similar
        recent = self.recent_responses[-self.window_size:]
        avg_length = sum(len(r) for r in recent) / len(recent)
        variance = sum((len(r) - avg_length) ** 2 for r in recent) / len(recent)

        # Low variance suggests diminishing returns
        return variance < self.eig_threshold

    def on_iteration_complete(
        self, context: dict[str, Any], result: ChatCompletion
    ) -> None:
        self.recent_responses.append(result.response)


# Usage
rlm = RLM(
    llm=my_llm,
    stopping_policy=EIGStoppingPolicy(eig_threshold=0.05),
)
result = rlm.completion("Solve this complex problem...")
```

## ContextCompressor

Compresses results from nested calls before returning them to the parent
orchestrator. Useful for managing token budgets in deep recursion.

### Protocol Definition

```python
class ContextCompressor(Protocol):
    def compress(self, result: str, max_tokens: int | None = None) -> str:
        """Compress nested call result before returning to parent."""
        ...
```

### Example: Truncation Compressor

```python
from dataclasses import dataclass

@dataclass
class TruncationCompressor:
    """Truncate results to max_tokens with ellipsis."""

    default_max_tokens: int = 500

    def compress(self, result: str, max_tokens: int | None = None) -> str:
        limit = max_tokens or self.default_max_tokens
        if len(result) <= limit:
            return result
        return result[:limit] + "..."


# Usage
rlm = RLM(
    llm=my_llm,
    context_compressor=TruncationCompressor(default_max_tokens=200),
)
```

### Example: LLM-Based Summarization

```python
from dataclasses import dataclass
from rlm.domain.ports import LLMPort

@dataclass
class SummarizingCompressor:
    """Use an LLM to summarize long results."""

    summarizer_llm: LLMPort
    max_tokens: int = 500

    def compress(self, result: str, max_tokens: int | None = None) -> str:
        limit = max_tokens or self.max_tokens
        if len(result) <= limit:
            return result

        # Use LLM to summarize
        from rlm.domain.models import LLMRequest
        summary_req = LLMRequest(
            prompt=f"Summarize this in under {limit} characters:\n\n{result}"
        )
        completion = self.summarizer_llm.complete(summary_req)
        return completion.response[:limit]
```

## NestedCallPolicy

Controls whether nested `llm_query()` calls from generated code should spawn
sub-orchestrators or use simple LLM calls.

### Protocol Definition

```python
from rlm.domain.agent_ports import NestedConfig

class NestedCallPolicy(Protocol):
    def should_orchestrate(self, prompt: str, depth: int) -> bool:
        """Return True to spawn a nested orchestrator for this call."""
        ...

    def get_nested_config(self) -> NestedConfig:
        """Return configuration for nested orchestrator."""
        ...
```

### NestedConfig TypedDict

```python
class NestedConfig(TypedDict, total=False):
    agent_mode: Literal["code", "tools"]
    tools: list[ToolDefinition]
    max_iterations: int
    max_depth: int
```

### Example: Depth-Based Orchestration

```python
from dataclasses import dataclass
from rlm.domain.agent_ports import NestedConfig

@dataclass
class DepthBasedNestedPolicy:
    """Spawn orchestrators only above certain depth."""

    min_depth_for_orchestration: int = 1
    nested_max_iterations: int = 10

    def should_orchestrate(self, prompt: str, depth: int) -> bool:
        # Only orchestrate nested calls at depth >= threshold
        return depth >= self.min_depth_for_orchestration

    def get_nested_config(self) -> NestedConfig:
        return NestedConfig(
            agent_mode="code",
            max_iterations=self.nested_max_iterations,
            max_depth=2,
        )


# Usage
rlm = RLM(
    llm=my_llm,
    nested_call_policy=DepthBasedNestedPolicy(min_depth_for_orchestration=1),
)
```

## Combining Policies

All three policies can be used together:

```python
from rlm import RLM

rlm = RLM(
    llm=my_llm,
    tools=[my_tool],
    agent_mode="tools",
    stopping_policy=EIGStoppingPolicy(eig_threshold=0.05),
    context_compressor=TruncationCompressor(default_max_tokens=200),
    nested_call_policy=DepthBasedNestedPolicy(min_depth_for_orchestration=2),
)

result = rlm.completion("Complex multi-step task...")
```

## Default Implementations

RLM provides sensible defaults:

### DefaultStoppingPolicy

Stops when `iteration >= max_iterations`. No custom logic.

```python
from rlm import DefaultStoppingPolicy

policy = DefaultStoppingPolicy()
assert policy.should_stop({"iteration": 30, "max_iterations": 30})  # True
assert not policy.should_stop({"iteration": 29, "max_iterations": 30})  # False
```

### NoOpContextCompressor

Passthrough - returns results unchanged.

```python
from rlm import NoOpContextCompressor

compressor = NoOpContextCompressor()
assert compressor.compress("long text") == "long text"
```

### SimpleNestedCallPolicy

Never spawns nested orchestrators - all nested calls use simple LLM completion.

```python
from rlm import SimpleNestedCallPolicy

policy = SimpleNestedCallPolicy()
assert not policy.should_orchestrate("any prompt", 0)  # Always False
assert policy.get_nested_config() == {}  # Empty config
```

## Importing Protocols

All protocols and default implementations are exported from the main package:

```python
from rlm import (
    # Protocols (for type hints and isinstance checks)
    StoppingPolicy,
    ContextCompressor,
    NestedCallPolicy,
    NestedConfig,
    # Default implementations
    DefaultStoppingPolicy,
    NoOpContextCompressor,
    SimpleNestedCallPolicy,
)
```

## Testing Custom Policies

Protocols are `@runtime_checkable`, so you can verify implementations:

```python
from rlm import StoppingPolicy, ContextCompressor, NestedCallPolicy

# Verify your custom class satisfies the protocol
assert isinstance(my_custom_stopping_policy, StoppingPolicy)
assert isinstance(my_custom_compressor, ContextCompressor)
assert isinstance(my_custom_nested_policy, NestedCallPolicy)
```

## Use Cases

### Debugging Applications

Custom `StoppingPolicy` can emit events for visualization:

```python
@dataclass
class DebugStoppingPolicy:
    event_callback: Callable[[str, dict], None]

    def should_stop(self, context: dict[str, Any]) -> bool:
        self.event_callback("iteration_check", context)
        return context.get("iteration", 0) >= context.get("max_iterations", 30)

    def on_iteration_complete(
        self, context: dict[str, Any], result: ChatCompletion
    ) -> None:
        self.event_callback("iteration_complete", {
            "iteration": context.get("iteration"),
            "response": result.response,
        })
```

### MDP Exploration

Custom policies can implement entropy-based stopping:

```python
@dataclass
class EntropyStoppingPolicy:
    entropy_threshold: float = 0.5

    def should_stop(self, context: dict[str, Any]) -> bool:
        # Stop when action entropy falls below threshold
        entropy = context.get("action_entropy", 1.0)
        return entropy < self.entropy_threshold

    def on_iteration_complete(
        self, context: dict[str, Any], result: ChatCompletion
    ) -> None:
        # External app updates context with computed entropy
        pass
```

### Context Budget Management

Use `ContextCompressor` to stay within token limits:

```python
@dataclass
class BudgetCompressor:
    total_budget: int = 4000
    used_tokens: int = 0

    def compress(self, result: str, max_tokens: int | None = None) -> str:
        remaining = self.total_budget - self.used_tokens
        if len(result) <= remaining:
            self.used_tokens += len(result)
            return result
        truncated = result[:remaining]
        self.used_tokens = self.total_budget
        return truncated + "...[budget exceeded]"
```
