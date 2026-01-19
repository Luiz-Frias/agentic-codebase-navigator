# Pydantic Integration

RLM supports optional Pydantic integration for enhanced type validation and JSON Schema generation. This is useful for structured outputs, tool definitions, and complex type hierarchies.

## Installation

Pydantic is an optional dependency:

```bash
# Install RLM with Pydantic support
pip install "agentic-codebase-navigator[pydantic]"

# Or add to existing installation
pip install pydantic>=2.0.0
```

## Why Pydantic?

RLM's JSON Schema mapper has two implementation paths:

| Path | When Used | Advantages |
|------|-----------|------------|
| **Manual** | Pydantic not installed or `prefer_pydantic=False` | Zero dependencies, predictable output |
| **TypeAdapter** | Pydantic installed and `prefer_pydantic=True` | Battle-tested, handles edge cases |

### Differences in Schema Output

```python
from typing import Optional

# Manual implementation: Optional unwrapped
mapper_manual = JsonSchemaMapper(prefer_pydantic=False)
mapper_manual.map(Optional[int])
# {"type": "integer"}

# Pydantic TypeAdapter: Explicit anyOf
mapper_pydantic = JsonSchemaMapper(prefer_pydantic=True)
mapper_pydantic.map(Optional[int])
# {"anyOf": [{"type": "integer"}, {"type": "null"}]}
```

Choose based on your needs:
- **Manual**: Simpler schemas, backward compatible with LLM providers
- **Pydantic**: More explicit, better for complex validation

## JsonSchemaMapper

The `JsonSchemaMapper` class converts Python types to JSON Schema:

```python
from rlm.domain.models.json_schema_mapper import JsonSchemaMapper

mapper = JsonSchemaMapper(prefer_pydantic=True)

# Basic types
mapper.map(str)           # {"type": "string"}
mapper.map(int)           # {"type": "integer"}
mapper.map(float)         # {"type": "number"}
mapper.map(bool)          # {"type": "boolean"}

# Container types
mapper.map(list[int])     # {"type": "array", "items": {"type": "integer"}}
mapper.map(dict[str, int]) # {"type": "object", "additionalProperties": {"type": "integer"}}

# Union types
mapper.map(int | str)     # {"anyOf": [{"type": "integer"}, {"type": "string"}]}
```

### Dataclass Support

Dataclasses are automatically converted to object schemas:

```python
from dataclasses import dataclass

@dataclass
class UserProfile:
    name: str
    age: int
    email: str | None = None

mapper.map(UserProfile)
# {
#     "type": "object",
#     "properties": {
#         "name": {"type": "string"},
#         "age": {"type": "integer"},
#         "email": {"type": "string"}
#     },
#     "required": ["name", "age"]
# }
```

### Pydantic Model Support

Pydantic models use their native `model_json_schema()`:

```python
from pydantic import BaseModel

class WeatherReport(BaseModel):
    city: str
    temperature: float
    conditions: str
    humidity: int

mapper.map(WeatherReport)
# Uses WeatherReport.model_json_schema() internally
```

## Tool Definitions with Pydantic

When defining tools, Pydantic models provide type-safe schemas:

```python
from pydantic import BaseModel, Field
from rlm.adapters.tools import tool, ToolRegistry

class SearchParams(BaseModel):
    """Parameters for web search."""
    query: str = Field(description="Search query")
    max_results: int = Field(default=10, ge=1, le=100)
    include_snippets: bool = Field(default=True)

@tool
def web_search(params: SearchParams) -> list[dict]:
    """Search the web for information."""
    # Implementation here
    return [{"title": "Result", "url": "https://example.com"}]

registry = ToolRegistry()
registry.register(web_search)
```

The tool's JSON Schema will include Field metadata (descriptions, constraints).

## Structured Outputs

Use Pydantic for structured LLM outputs:

```python
from pydantic import BaseModel
from rlm.adapters.tools import pydantic_to_schema

class ExtractedEntity(BaseModel):
    name: str
    entity_type: str
    confidence: float

# Convert to JSON Schema for LLM
schema = pydantic_to_schema(ExtractedEntity)

# Use in tool definition or response_format
result = rlm.completion(
    "Extract entities from: 'Apple released the iPhone in 2007'",
    response_format=schema,
)
```

## Dual-Path Validation

RLM uses a dual-path validation strategy for tool call responses:

```
LLM Response
    │
    ▼
┌─────────────────────┐
│ Try Pydantic        │
│ TypeAdapter.validate│
└─────────────────────┘
    │
    │ Success? ──────────▶ Return typed result
    │
    ▼ Failure
┌─────────────────────┐
│ Fallback to         │
│ SafeAccessor        │
└─────────────────────┘
    │
    ▼
Return dict with manual extraction
```

This ensures graceful degradation when:
- Pydantic is not installed
- The LLM returns malformed data
- Schema validation fails

## SafeAccessor Fallback

When Pydantic validation fails, `SafeAccessor` provides duck-typed navigation:

```python
from rlm.domain.models.safe_accessor import SafeAccessor

# Parse LLM response safely
response = {"tool_calls": [{"name": "search", "arguments": {"q": "test"}}]}
accessor = SafeAccessor(response)

# Navigate with defaults
tool_name = accessor["tool_calls"][0]["name"].unwrap_or("unknown")
query = accessor["tool_calls"][0]["arguments"]["q"].unwrap_or("")
```

## Configuration

### Per-Mapper Configuration

```python
# Always use manual implementation
mapper = JsonSchemaMapper(prefer_pydantic=False)

# Prefer Pydantic when available
mapper = JsonSchemaMapper(prefer_pydantic=True)
```

### Checking Pydantic Availability

```python
from rlm.domain.models.json_schema_mapper import has_pydantic

if has_pydantic():
    print("Pydantic is available")
else:
    print("Using manual schema generation")
```

## Best Practices

### 1. Use Pydantic for Complex Types

```python
# Good: Complex nested structures
class APIResponse(BaseModel):
    status: Literal["success", "error"]
    data: list[DataItem] | None
    errors: list[ErrorDetail] = []
```

### 2. Use Manual for Simple Types

```python
# Good: Simple tool parameters
@tool
def calculate(expression: str) -> float:
    """Evaluate a math expression."""
    return eval(expression)  # Schema: {"expression": {"type": "string"}}
```

### 3. Always Handle Validation Errors

```python
from pydantic import ValidationError

try:
    result = MyModel.model_validate(llm_response)
except ValidationError as e:
    # Log validation errors, use fallback
    logger.warning(f"Validation failed: {e}")
    result = extract_manually(llm_response)
```

### 4. Test Both Paths

```python
import pytest

@pytest.mark.parametrize("prefer_pydantic", [True, False])
def test_schema_generation(prefer_pydantic):
    mapper = JsonSchemaMapper(prefer_pydantic=prefer_pydantic)
    schema = mapper.map(MyType)
    assert schema["type"] == "object"
```

## Troubleshooting

### "Pydantic not found" Warnings

If you see warnings about Pydantic not being found:

```bash
pip install "agentic-codebase-navigator[pydantic]"
```

### Schema Differences Between Paths

If you need consistent schemas across environments:

```python
# Force manual implementation for consistency
mapper = JsonSchemaMapper(prefer_pydantic=False)
```

### Circular Reference Errors

Pydantic handles recursive types, but the manual implementation may not:

```python
# Use Pydantic for recursive structures
class TreeNode(BaseModel):
    value: int
    children: list["TreeNode"] = []

TreeNode.model_rebuild()  # Resolve forward references
```

## See Also

- [Extension Protocols](./extension-protocols.md) — Custom policies
- [Tool Calling](../README.md#tool-calling-agent-mode) — Basic tool usage
- [API Reference](../api-reference.md) — Full API documentation
