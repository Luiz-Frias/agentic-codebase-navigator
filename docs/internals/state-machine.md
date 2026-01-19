# State Machine Architecture

RLM v1.2.0 introduces a declarative state machine architecture for orchestration, replacing nested loops with explicit state transitions, guards, and typed event dispatch.

## Why State Machines?

The original orchestrator used nested `while True` loops with complex condition checks:

```python
# Before: Nested loops with implicit control flow
while iteration < max_iterations:
    response = llm.complete(...)
    if has_final_answer(response):
        break
    code_blocks = parse_code_blocks(response)
    if code_blocks:
        for block in code_blocks:
            result = env.execute(block)
            # ... more nested logic
```

This approach had several issues:
- **C901 complexity violations** — Cyclomatic complexity exceeded linter thresholds
- **Implicit control flow** — Difficult to trace which conditions led to which outcomes
- **Hard to extend** — Adding new states or transitions required modifying deeply nested code
- **Testing challenges** — Unit testing specific transitions required mocking entire loops

The state machine pattern solves these by making control flow **explicit and declarative**.

## Core Concepts

### StateMachine[S, E, C]

The generic `StateMachine` class is parameterized by three types:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `S` | State type (typically an Enum) | `CodeModeState` |
| `E` | Event type (union of event dataclasses) | `CodeModeEvent` |
| `C` | Context type (mutable dataclass) | `CodeModeContext` |

```python
from rlm.domain.models.state_machine import StateMachine

machine: StateMachine[MyState, MyEvent, MyContext] = StateMachine()
```

### States

States are discrete phases of execution. Each state represents a distinct operational mode:

```python
from enum import Enum, auto

class CodeModeState(Enum):
    INIT = auto()           # Initialize, validate config
    SHALLOW_CALL = auto()   # Single LLM call (depth exceeded)
    PROMPTING = auto()      # Calling the LLM
    EXECUTING = auto()      # Running code blocks
    DONE = auto()           # Terminal state
```

### Events

Events are immutable dataclasses that trigger transitions. They carry relevant data:

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class CodeBlocksFound:
    """Code blocks extracted from LLM response."""
    blocks: list[str]

@dataclass(slots=True, frozen=True)
class FinalAnswerFound:
    """Final answer extracted from response."""
    answer: str
```

### Context

Context is a mutable dataclass that accumulates state across transitions:

```python
@dataclass
class CodeModeContext:
    iteration: int = 0
    max_iterations: int = 30
    message_history: list[dict] = field(default_factory=list)
    final_answer: str | None = None
    # ... more fields
```

### Transitions

Transitions connect states via events, with optional guards and actions:

```python
machine.transition(
    from_state=CodeModeState.PROMPTING,
    event_type=FinalAnswerFound,
    to_state=CodeModeState.DONE,
    guard=lambda e, c: len(e.answer) > 0,  # Optional predicate
    action=store_final_answer,              # Optional side effect
)
```

### Event Sources

Event sources are callables that observe state and context, perform side effects (LLM calls, code execution), and return events:

```python
def event_source(state: S, ctx: C) -> E | None:
    """Generate next event based on current state."""
    if state == CodeModeState.PROMPTING:
        response = llm.complete(...)
        if has_final_answer(response):
            return FinalAnswerFound(answer=extract_answer(response))
        return CodeBlocksFound(blocks=parse_blocks(response))
    return None  # No more events
```

## Code Mode State Machine

### State Flow Diagram

```
                    ┌──────────────────────────────────────────────┐
                    │                                              │
                    │  ┌──────┐  DepthExceeded  ┌─────────────┐   │
                    │  │ INIT │ ───────────────▶│ SHALLOW_CALL│   │
                    │  └──────┘                 └─────────────┘   │
                    │      │                           │          │
                    │      │ LLMResponse               │ LLMResp  │
                    │      ▼                           │          │
                    │  ┌──────────┐                    │          │
                    │  │ PROMPTING│◀───────────────────┼──────────┤
                    │  └──────────┘                    │          │
                    │      │       │                   │          │
                    │      │       │ FinalAnswer       │          │
                    │      │       ▼                   ▼          │
                    │      │  ┌────────────────────────────┐      │
                    │      │  │            DONE            │      │
                    │      │  └────────────────────────────┘      │
                    │      │                   ▲                  │
                    │      │ CodeBlocks        │ MaxIterations    │
                    │      ▼                   │ or FinalAnswer   │
                    │  ┌──────────┐            │                  │
                    │  │EXECUTING │────────────┘                  │
                    │  └──────────┘                               │
                    │      │  ▲                                   │
                    │      │  │ CodeExecuted                      │
                    │      └──┘                                   │
                    │                                              │
                    └──────────────────────────────────────────────┘
```

### States

| State | Description | Entry Condition |
|-------|-------------|-----------------|
| `INIT` | Validate config, check depth | Initial state |
| `SHALLOW_CALL` | Single LLM call without iteration | `depth >= max_depth` |
| `PROMPTING` | Call LLM with current prompt | Normal flow from INIT |
| `EXECUTING` | Execute parsed code blocks | Code blocks found |
| `DONE` | Terminal — return result | Final answer or max iterations |

### Events

| Event | Triggers Transition |
|-------|-------------------|
| `DepthExceeded` | INIT → SHALLOW_CALL |
| `LLMResponseReceived` | INIT → PROMPTING, SHALLOW_CALL → DONE |
| `CodeBlocksFound` | PROMPTING → EXECUTING |
| `FinalAnswerFound` | PROMPTING → DONE, EXECUTING → DONE |
| `CodeExecuted` | EXECUTING → PROMPTING |
| `MaxIterationsReached` | EXECUTING → DONE |

### Usage

```python
from rlm.domain.services.code_mode_machine import build_code_mode_machine
from rlm.domain.models.orchestration_types import CodeModeContext, CodeModeState

# Build the machine (do this once at startup)
machine = build_code_mode_machine()

# Create context for this run
ctx = CodeModeContext(
    prompt="Calculate the factorial of 10",
    max_iterations=30,
    depth=0,
)

# Run with an event source
final_state, final_ctx = machine.run(
    initial_state=CodeModeState.INIT,
    context=ctx,
    event_source=my_event_source,
)

# Check results
if final_ctx.final_answer:
    print(f"Result: {final_ctx.final_answer}")
```

## Tools Mode State Machine

### State Flow Diagram

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    │  ┌──────┐  LLMResponse  ┌──────────┐   │
                    │  │ INIT │ ─────────────▶│ PROMPTING│   │
                    │  └──────┘               └──────────┘   │
                    │                             │    │     │
                    │                   ToolCalls │    │     │
                    │                             ▼    │     │
                    │                     ┌────────────┐│    │
                    │                     │ EXECUTING  ││    │
                    │                     │   TOOLS    ││    │
                    │                     └────────────┘│    │
                    │                       │    │      │    │
                    │            ToolsExec  │    │      │    │
                    │            ┌──────────┘    │      │    │
                    │            │               │      │    │
                    │            ▼               │      │    │
                    │        ┌───────┐           │NoTool│    │
                    │        │       │ PolicyStop│Calls │    │
                    │        │ DONE  │◀──────────┴──────┘    │
                    │        │       │                       │
                    │        └───────┘                       │
                    │                                         │
                    └─────────────────────────────────────────┘
```

### States

| State | Description |
|-------|-------------|
| `INIT` | Build conversation, validate tools |
| `PROMPTING` | Call LLM with tools available |
| `EXECUTING_TOOLS` | Execute tool calls |
| `DONE` | Terminal — return result |

### Events

| Event | Description |
|-------|-------------|
| `ToolCallsFound` | LLM requested tool invocations |
| `ToolsExecuted` | All tool calls completed |
| `NoToolCalls` | LLM returned text without tool calls (final answer) |
| `PolicyStop` | Custom `StoppingPolicy` requested termination |
| `MaxIterationsReached` | Iteration limit hit |

### Policy Stop Tracking (v1.2.0)

When a `StoppingPolicy` requests early termination, the context is marked:

```python
@dataclass
class ToolsModeContext:
    policy_stop: bool = False  # Set True when policy stops execution
    last_response: str | None = None
```

The orchestrator checks `policy_stop` first for reliable termination detection:

```python
if final_ctx.policy_stop:
    finish_reason = "policy_stop"
```

## Writing Custom State Machines

### Step 1: Define States

```python
from enum import Enum, auto

class MyState(Enum):
    START = auto()
    PROCESSING = auto()
    FINISHED = auto()
```

### Step 2: Define Events

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class ProcessingComplete:
    result: str

@dataclass(slots=True, frozen=True)
class ErrorOccurred:
    error: str
```

### Step 3: Define Context

```python
from dataclasses import dataclass, field

@dataclass
class MyContext:
    input_data: str
    results: list[str] = field(default_factory=list)
    error: str | None = None
```

### Step 4: Build the Machine

```python
from rlm.domain.models.state_machine import StateMachine

def build_my_machine() -> StateMachine[MyState, MyEvent, MyContext]:
    machine: StateMachine[MyState, MyEvent, MyContext] = StateMachine()

    # Register states
    machine.state(MyState.START)
    machine.state(MyState.PROCESSING)
    machine.state(MyState.FINISHED)

    # Mark terminal states
    machine.terminal(MyState.FINISHED)

    # Define transitions
    machine.transition(MyState.START, ProcessingComplete, MyState.PROCESSING)
    machine.transition(MyState.PROCESSING, ProcessingComplete, MyState.FINISHED)
    machine.transition(MyState.PROCESSING, ErrorOccurred, MyState.FINISHED)

    return machine
```

### Step 5: Create Event Source

```python
def my_event_source(state: MyState, ctx: MyContext) -> MyEvent | None:
    if state == MyState.START:
        # Do some work
        return ProcessingComplete(result="started")

    if state == MyState.PROCESSING:
        try:
            result = process(ctx.input_data)
            ctx.results.append(result)
            return ProcessingComplete(result=result)
        except Exception as e:
            ctx.error = str(e)
            return ErrorOccurred(error=str(e))

    return None
```

### Step 6: Run

```python
machine = build_my_machine()
ctx = MyContext(input_data="hello")
final_state, final_ctx = machine.run(MyState.START, ctx, my_event_source)
```

## Async Support

The `StateMachine` supports both sync and async execution:

```python
# Synchronous
final_state, ctx = machine.run(initial_state, ctx, sync_event_source)

# Asynchronous
final_state, ctx = await machine.arun(initial_state, ctx, async_event_source)
```

Async event sources have the signature:

```python
async def async_event_source(state: S, ctx: C) -> E | None:
    response = await llm.acomplete(...)
    return SomeEvent(data=response)
```

## Guards and Actions

### Guards

Guards are predicates that determine if a transition should fire:

```python
def has_remaining_budget(event: MyEvent, ctx: MyContext) -> bool:
    return ctx.token_budget > 0

machine.transition(
    MyState.PROCESSING,
    ProcessingComplete,
    MyState.PROCESSING,
    guard=has_remaining_budget,  # Only continue if budget remains
)
```

### Actions

Actions are side effects executed during transitions:

```python
def log_transition(event: MyEvent, ctx: MyContext) -> None:
    print(f"Transitioning with event: {event}")

machine.transition(
    MyState.START,
    ProcessingComplete,
    MyState.PROCESSING,
    action=log_transition,
)
```

### Execution Order

When a transition fires:

1. `on_exit` callback of the **old** state (if registered)
2. `action` of the transition (if provided)
3. `on_enter` callback of the **new** state (if registered)

```python
machine.state(
    MyState.PROCESSING,
    on_enter=lambda ctx: print("Entering PROCESSING"),
    on_exit=lambda ctx: print("Exiting PROCESSING"),
)
```

## Thread Safety

- **Configuration phase**: NOT thread-safe — build machines at startup
- **Execution phase**: Thread-safe if context mutations are thread-safe
- Each `run()` call operates on its own context instance

## See Also

- [Extension Protocols](../extending/extension-protocols.md) — Custom stopping policies
- [Architecture](../architecture.md) — Overall system design
- [Troubleshooting](../troubleshooting.md) — Common issues
