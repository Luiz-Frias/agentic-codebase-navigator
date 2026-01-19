"""
Native tool adapter for Python callables.

Wraps plain Python functions as ToolPort implementations, automatically
extracting schema from type hints and docstrings.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import types
import typing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast, get_args, get_origin, get_type_hints

from rlm.adapters.base import BaseToolAdapter
from rlm.domain.agent_ports import ToolDefinition

if TYPE_CHECKING:
    from collections.abc import Callable

# dict[K, V] has exactly 2 type args: key type and value type
_DICT_TYPE_ARGS_COUNT = 2


def _python_type_to_json_schema(python_type: type) -> dict[str, Any]:
    """Convert a Python type to JSON Schema."""
    # Handle None/NoneType
    if python_type is type(None):
        return {"type": "null"}

    # Basic type mappings
    type_map: dict[type, dict[str, Any]] = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }

    if python_type in type_map:
        return type_map[python_type]

    # Handle Optional (Union with None) and parameterized collections
    origin = get_origin(python_type)
    args = get_args(python_type)

    if origin is list and args:
        item_type = cast("type", args[0])
        return {"type": "array", "items": _python_type_to_json_schema(item_type)}

    if origin is dict and len(args) >= _DICT_TYPE_ARGS_COUNT:
        value_type = cast("type", args[1])
        return {
            "type": "object",
            "additionalProperties": _python_type_to_json_schema(value_type),
        }

    # Union types (including Optional)
    if origin in (types.UnionType, typing.Union):
        non_none = [cast("type", a) for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[X] -> X (nullable handled implicitly by most LLMs)
            return _python_type_to_json_schema(non_none[0])
        union_types = [cast("type", a) for a in args]
        return {"anyOf": [_python_type_to_json_schema(a) for a in union_types]}

    # Fallback to string for complex types
    return {"type": "string"}


def _parse_docstring_params(docstring: str | None) -> dict[str, str]:
    """
    Parse parameter descriptions from docstring.

    Supports Google, NumPy, and Sphinx style docstrings.
    """
    if not docstring:
        return {}

    params: dict[str, str] = {}

    # Google style: Args:\n    param_name: description
    google_pattern = r"Args?:\s*\n((?:\s+\w+.*\n?)+)"
    google_match = re.search(google_pattern, docstring)
    if google_match:
        args_section = google_match.group(1)
        for match in re.finditer(
            r"(\w+)\s*(?:\([^)]*\))?:\s*(.+?)(?=\n\s+\w+|\n\n|$)",
            args_section,
            re.DOTALL,
        ):
            params[match.group(1)] = match.group(2).strip()

    # NumPy style: Parameters\n----------\nparam_name : type\n    description
    numpy_pattern = r"Parameters?\s*\n-+\s*\n((?:.+\n?)+?)(?=\n\w|\n-|$)"
    numpy_match = re.search(numpy_pattern, docstring)
    if numpy_match and not params:
        params_section = numpy_match.group(1)
        for match in re.finditer(
            r"(\w+)\s*:\s*\w+.*?\n\s+(.+?)(?=\n\w+\s*:|\n\n|$)",
            params_section,
            re.DOTALL,
        ):
            params[match.group(1)] = match.group(2).strip()

    # Sphinx style: :param param_name: description
    sphinx_pattern = r":param\s+(\w+):\s*(.+?)(?=:param|:return|:raises|$)"
    for match in re.finditer(sphinx_pattern, docstring, re.DOTALL):
        if match.group(1) not in params:
            params[match.group(1)] = match.group(2).strip()

    return params


@dataclass(slots=True)
class NativeToolAdapter(BaseToolAdapter):
    """
    Wraps a Python callable as a ToolPort.

    Automatically extracts tool schema from:
    - Function name (or custom name)
    - Docstring (first line as description)
    - Type hints (for parameter schemas)
    - Docstring parameter descriptions

    Example:
        def get_weather(city: str, unit: str = "celsius") -> str:
            '''Get the current weather for a city.

    Args:
                city: The city name to look up
                unit: Temperature unit (celsius or fahrenheit)
            '''
            return f"Weather in {city}: 72{unit[0].upper()}"

        tool = NativeToolAdapter(get_weather)
        # tool.definition will have the proper JSON schema

    """

    func: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    _definition: ToolDefinition | None = field(default=None, init=False, repr=False)

    @property
    def definition(self) -> ToolDefinition:
        """Generate tool definition from function introspection."""
        if self._definition is not None:
            return self._definition

        # Get function metadata
        func_name = self.name or self.func.__name__
        docstring = inspect.getdoc(self.func)

        # Extract description from docstring first line
        if self.description:
            func_description = self.description
        elif docstring:
            func_description = docstring.split("\n")[0].strip()
        else:
            func_description = f"Call the {func_name} function"

        # Parse docstring for parameter descriptions
        param_docs = _parse_docstring_params(docstring)

        # Get type hints
        hints: dict[str, type[Any]]
        try:
            hints = get_type_hints(self.func)
        except Exception:
            hints = {}

        # Get function signature
        sig = inspect.signature(self.func)

        # Build parameters schema
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            # Skip *args, **kwargs, and 'self'/'cls'
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if param_name in ("self", "cls"):
                continue

            # Get type from hints
            param_type = hints.get(param_name, str)
            param_schema = _python_type_to_json_schema(param_type)

            # Add description if available
            if param_name in param_docs:
                param_schema["description"] = param_docs[param_name]

            properties[param_name] = param_schema

            # Track required parameters (those without defaults)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        parameters_schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters_schema["required"] = required

        self._definition = ToolDefinition(
            name=func_name,
            description=func_description,
            parameters=parameters_schema,
        )
        return self._definition

    def execute(self, **kwargs: Any) -> Any:
        """Execute the wrapped function synchronously."""
        result = self.func(**kwargs)
        if asyncio.iscoroutine(result):
            result.close()
            raise TypeError(
                "Tool function returned a coroutine in sync execution. "
                "Declare the tool as async or call it via aexecute().",
            )
        return result

    async def aexecute(self, **kwargs: Any) -> Any:
        """Execute the wrapped function asynchronously."""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**kwargs)

        result = await asyncio.to_thread(self.func, **kwargs)
        if asyncio.iscoroutine(result):
            result.close()
            raise TypeError(
                "Tool function returned a coroutine from a sync implementation. "
                "Declare the tool as async instead.",
            )
        return result
