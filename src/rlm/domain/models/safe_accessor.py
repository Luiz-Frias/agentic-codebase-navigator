"""
Safe accessor for SDK objects and dict representations.

This module provides a SafeAccessor abstraction that unifies attribute access
(SDK objects) and key access (dicts) into a single interface. It handles the
"duck-typing with fallback" pattern common in SDK boundary code.

Type-driven boundary pattern:
- Input: `object` (accepts any SDK response or dict at boundary)
- Typed accessors: Return `Result[T, AccessError]` - caller gets typed data or explicit error
- Legacy accessor: `get()` returns `object` with default - use typed accessors for new code

Example:
    # Works with either SDK objects or dicts
    accessor = SafeAccessor(response)

    # Type-safe access with Result (preferred in domain layer):
    match accessor.get_str("content"):
        case Ok(content):
            process(content)  # content is str
        case Err(e):
            handle_missing(e)

    # Legacy access (for boundary/adapter code):
    choices = accessor.get("choices")  # Returns object

Design notes:
- Try attribute access first (SDK objects are the common case)
- Fall back to dict/list access for JSON-decoded responses
- Typed accessors return Result for explicit error handling
- Legacy get() returns default (None) for backward compatibility

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlm.domain.errors import ValidationError
from rlm.domain.models.result import Err, Ok

if TYPE_CHECKING:
    from rlm.domain.models.result import Result


class AccessError(ValidationError):
    """Error accessing a value from an SDK response or dict."""


@dataclass(slots=True)
class SafeAccessor[T]:
    """
    Unified accessor for SDK objects and dict representations.

    Provides both typed accessors (returning Result) and legacy accessor
    (returning object with default) for different use cases.

    Thread safety:
    - Thread-safe for read-only access
    - Does not mutate the wrapped object

    """

    _obj: T

    # =========================================================================
    # Internal: Raw access returning object
    # =========================================================================

    def _raw_get(self, key: str | int, /) -> object:
        """
        Internal raw access - tries attr then dict/list, returns None if missing.

        This is the foundation for both typed and legacy accessors.

        Access strategy:
        - For dicts with string keys: try dict access first (avoids method shadowing)
        - For SDK objects: try attribute access first, then dict/list access
        """
        if self._obj is None:
            return None

        # For plain dicts (not subclasses) with string keys, prefer dict access
        # to avoid method shadowing (e.g., dict["items"] vs dict.items method).
        # Dict subclasses (SDK responses, custom classes) still prefer attribute access.
        if type(self._obj) is dict and isinstance(key, str):
            if key in self._obj:
                return self._obj[key]
            return None

        # For string keys on non-dict objects, try attribute access first (SDK objects)
        if isinstance(key, str):
            try:
                # getattr returns Any (SDK boundary) - we capture and return as object
                result: object = getattr(self._obj, key)  # pyright: ignore[reportAny] - SDK boundary
            except (AttributeError, TypeError):
                pass
            else:
                return result

        # Try dict/list access (for int keys or as fallback)
        try:
            return self._obj[key]  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            pass

        return None

    # =========================================================================
    # Typed Accessors: Return Result[T, AccessError]
    # Use these in domain/application layers for type-safe access
    # =========================================================================

    def get_str(self, key: str | int, /) -> Result[str, AccessError]:
        """
        Get a string value, returning Result for type-safe handling.

        Args:
            key: Attribute name (str) or index (int)

        Returns:
            Ok(str) if value exists and is a string
            Err(AccessError) if missing or wrong type

        """
        value = self._raw_get(key)
        if value is None:
            return Err(AccessError(f"Key '{key}' not found or is None"))
        if isinstance(value, str):
            return Ok(value)
        return Err(AccessError(f"Key '{key}' expected str, got {type(value).__name__}"))

    def get_str_or(self, key: str | int, default: str, /) -> str:
        """
        Get a string value with fallback default.

        Returns the string if found and valid, otherwise returns default.
        """
        value = self._raw_get(key)
        if isinstance(value, str):
            return value
        return default

    def get_int(self, key: str | int, /) -> Result[int, AccessError]:
        """
        Get an integer value, returning Result for type-safe handling.

        Args:
            key: Attribute name (str) or index (int)

        Returns:
            Ok(int) if value exists and is an int
            Err(AccessError) if missing or wrong type

        """
        value = self._raw_get(key)
        if value is None:
            return Err(AccessError(f"Key '{key}' not found or is None"))
        if isinstance(value, bool):
            # bool is subclass of int, but we don't want to treat True/False as ints
            return Err(AccessError(f"Key '{key}' expected int, got bool"))
        if isinstance(value, int):
            return Ok(value)
        return Err(AccessError(f"Key '{key}' expected int, got {type(value).__name__}"))

    def get_int_or(self, key: str | int, default: int, /) -> int:
        """
        Get an integer value with fallback default.

        Returns the int if found and valid, otherwise returns default.
        """
        value = self._raw_get(key)
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        return default

    def get_float(self, key: str | int, /) -> Result[float, AccessError]:
        """
        Get a float value, returning Result for type-safe handling.

        Accepts both int and float (int is widened to float).

        Args:
            key: Attribute name (str) or index (int)

        Returns:
            Ok(float) if value exists and is numeric
            Err(AccessError) if missing or wrong type

        """
        value = self._raw_get(key)
        if value is None:
            return Err(AccessError(f"Key '{key}' not found or is None"))
        if isinstance(value, bool):
            return Err(AccessError(f"Key '{key}' expected float, got bool"))
        if isinstance(value, (int, float)):
            return Ok(float(value))
        return Err(AccessError(f"Key '{key}' expected float, got {type(value).__name__}"))

    def get_float_or(self, key: str | int, default: float, /) -> float:
        """
        Get a float value with fallback default.

        Returns the float if found and valid (int widened), otherwise returns default.
        """
        value = self._raw_get(key)
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        return default

    def get_bool(self, key: str | int, /) -> Result[bool, AccessError]:
        """
        Get a boolean value, returning Result for type-safe handling.

        Args:
            key: Attribute name (str) or index (int)

        Returns:
            Ok(bool) if value exists and is a bool
            Err(AccessError) if missing or wrong type

        """
        value = self._raw_get(key)
        if value is None:
            return Err(AccessError(f"Key '{key}' not found or is None"))
        if isinstance(value, bool):
            return Ok(value)
        return Err(AccessError(f"Key '{key}' expected bool, got {type(value).__name__}"))

    def get_bool_or(self, key: str | int, default: bool, /) -> bool:  # noqa: FBT001
        """
        Get a boolean value with fallback default.

        Returns the bool if found and valid, otherwise returns default.
        """
        value = self._raw_get(key)
        if isinstance(value, bool):
            return value
        return default

    def get_list(self, key: str | int, /) -> Result[list[object], AccessError]:
        """
        Get a list value, returning Result for type-safe handling.

        Args:
            key: Attribute name (str) or index (int)

        Returns:
            Ok(list) if value exists and is a list
            Err(AccessError) if missing or wrong type

        """
        value = self._raw_get(key)
        if value is None:
            return Err(AccessError(f"Key '{key}' not found or is None"))
        if isinstance(value, list):
            return Ok(value)
        return Err(AccessError(f"Key '{key}' expected list, got {type(value).__name__}"))

    def get_list_or(self, key: str | int, /) -> list[object]:
        """
        Get a list value with empty list fallback.

        Returns the list if found and valid, otherwise returns empty list.
        """
        value = self._raw_get(key)
        if isinstance(value, list):
            return value
        return []

    def get_dict(self, key: str | int, /) -> Result[dict[str, object], AccessError]:
        """
        Get a dict value, returning Result for type-safe handling.

        Args:
            key: Attribute name (str) or index (int)

        Returns:
            Ok(dict) if value exists and is a dict
            Err(AccessError) if missing or wrong type

        """
        value = self._raw_get(key)
        if value is None:
            return Err(AccessError(f"Key '{key}' not found or is None"))
        if isinstance(value, dict):
            return Ok(value)
        return Err(AccessError(f"Key '{key}' expected dict, got {type(value).__name__}"))

    def get_dict_or(self, key: str | int, /) -> dict[str, object]:
        """
        Get a dict value with empty dict fallback.

        Returns the dict if found and valid, otherwise returns empty dict.
        """
        value = self._raw_get(key)
        if isinstance(value, dict):
            return value
        return {}

    # =========================================================================
    # Legacy Accessor: Returns object (for boundary/adapter code)
    # =========================================================================

    def get(self, key: str | int, /, *, default: object = None) -> object:
        """
        Retrieve a value by attribute name or key (legacy accessor).

        Access strategy (in order):
        1. Try attribute access (getattr) - handles SDK objects
        2. Try dict/list access (__getitem__) - handles dicts and lists
        3. Return default

        Note: For new code in domain/application layers, prefer typed accessors
        (get_str, get_int, etc.) which return Result[T, AccessError].

        Args:
            key: Attribute name (str) or index (int)
            default: Value to return if access fails (default: None)

        Returns:
            The accessed value, or default if not found

        """
        value = self._raw_get(key)
        return value if value is not None else default

    def get_nested(self, *keys: str | int, default: object = None) -> object:
        """
        Navigate nested structures by attribute names or indices.

        Traverses through nested objects/dicts/lists using the provided keys.
        Returns default if any step in the path fails.

        Args:
            *keys: Sequence of attribute names or indices to traverse
            default: Value to return if any access fails (default: None)

        Returns:
            The final accessed value, or default if path fails

        Example:
            # Equivalent to: response.choices[0].message.content
            accessor.get_nested("choices", 0, "message", "content")

        """
        if not keys:
            return self._obj if self._obj is not None else default

        current: object = self._obj
        for key in keys:
            if current is None:
                return default

            # Create accessor for current level
            accessor: SafeAccessor[object] = SafeAccessor(current)
            current = accessor._raw_get(key)

            if current is None:
                return default

        return current

    def child(self, *keys: str | int) -> SafeAccessor[object]:
        """
        Get a SafeAccessor wrapping a nested value.

        Enables fluent chaining:
            accessor.child("choices", 0).child("message").get("content")

        Args:
            *keys: Path to the nested value

        Returns:
            New SafeAccessor wrapping the nested value (or None if not found)

        """
        value = self.get_nested(*keys) if keys else self._obj
        return SafeAccessor(value)
