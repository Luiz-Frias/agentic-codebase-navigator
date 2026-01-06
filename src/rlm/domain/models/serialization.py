from __future__ import annotations

from types import ModuleType
from typing import Any


def serialize_value(value: Any) -> Any:
    """Convert a Python value into a JSON-serializable representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, ModuleType):
        return f"<module '{value.__name__}'>"
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    if callable(value):
        return f"<{type(value).__name__} '{getattr(value, '__name__', repr(value))}'>"
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"
