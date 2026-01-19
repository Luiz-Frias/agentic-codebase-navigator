"""
Type-safe dispatch based on runtime types.

This module provides a registry-based approach to type dispatch, replacing
scattered isinstance() chains with a declarative, testable pattern.

Example:
    mapper = TypeMapper[Any, str]()
    mapper.register(str, lambda x: f"string:{x}")
    mapper.register(int, lambda x: f"int:{x}")
    mapper.default(lambda x: repr(x))

    result = mapper.map("hello")  # "string:hello"
    result = mapper.map(42)       # "int:42"
    result = mapper.map(3.14)     # "3.14" (default)

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

In = TypeVar("In")
Out = TypeVar("Out")


@dataclass
class TypeMapper[In, Out]:
    """
    Registry-based type dispatch.

    Replaces scattered isinstance() chains with a declarative registry.
    Each handler is simple, testable, and single-responsibility.

    Design notes:
    - Handlers are checked in registration order (first match wins)
    - Subclass instances match parent class handlers (isinstance semantics)
    - Default handler is used only when no specific handler matches
    - Raises TypeError if no handler matches and no default is set

    Thread safety:
    - NOT thread-safe during registration (build mappers at startup)
    - Thread-safe during map() calls (read-only after construction)
    """

    _handlers: dict[type, Callable[[In], Out]] = field(default_factory=dict)
    _default_handler: Callable[[In], Out] | None = None

    def register(self, type_: type, handler: Callable[[In], Out]) -> TypeMapper[In, Out]:
        """
        Register a handler for a specific type.

        Args:
            type_: The type to handle (exact type or base class)
            handler: Function to transform values of this type

        Returns:
            Self for method chaining

        Note:
            Handlers are checked in registration order. Register more specific
            types (subclasses) before more general types (base classes).

        """
        self._handlers[type_] = handler
        return self

    def default(self, handler: Callable[[In], Out]) -> TypeMapper[In, Out]:
        """
        Set the fallback handler for unregistered types.

        Args:
            handler: Function to transform values with no matching handler

        Returns:
            Self for method chaining

        """
        self._default_handler = handler
        return self

    def map(self, value: In) -> Out:
        """
        Dispatch to the appropriate handler based on value's runtime type.

        Args:
            value: The value to transform

        Returns:
            Result of calling the matching handler

        Raises:
            TypeError: If no handler matches and no default is set

        """
        for type_, handler in self._handlers.items():
            if isinstance(value, type_):
                return handler(value)

        if self._default_handler is not None:
            return self._default_handler(value)

        raise TypeError(f"No handler registered for {type(value).__name__}")

    def can_handle(self, value: In) -> bool:
        """
        Check if a handler exists for this value's type.

        Args:
            value: The value to check

        Returns:
            True if map() would succeed, False if it would raise TypeError

        """
        if self._default_handler is not None:
            return True

        return any(isinstance(value, type_) for type_ in self._handlers)

    def has_registered_handler(self, value: In) -> bool:
        """
        Check if there's a registered handler (not default) for this value's type.

        Args:
            value: The value to check

        Returns:
            True if a specific type handler is registered for this value

        Note:
            Unlike can_handle(), this does NOT consider the default handler.
            Use this when you need to check if a value has explicit type support
            vs. falling through to a generic default.

        """
        return any(isinstance(value, type_) for type_ in self._handlers)
