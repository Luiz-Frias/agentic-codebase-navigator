"""
TDD tests for TypeMapper abstraction.

These tests define the contract BEFORE implementation.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


class TestTypeMapperBasicDispatch:
    """Core dispatch behavior - the 95% case."""

    def test_maps_string_to_registered_handler(self) -> None:
        """Given a string, dispatch to the string handler."""
        from rlm.domain.models.type_mapping import TypeMapper

        mapper: TypeMapper[object, str] = TypeMapper()
        mapper.register(str, lambda x: f"string:{x}")

        result = mapper.map("hello")

        assert result == "string:hello"

    def test_maps_int_to_registered_handler(self) -> None:
        """Given an int, dispatch to the int handler."""
        from rlm.domain.models.type_mapping import TypeMapper

        mapper: TypeMapper[object, str] = TypeMapper()
        mapper.register(int, lambda x: f"int:{x}")

        result = mapper.map(42)

        assert result == "int:42"

    def test_maps_list_to_registered_handler(self) -> None:
        """Given a list, dispatch to the list handler."""
        from rlm.domain.models.type_mapping import TypeMapper

        mapper: TypeMapper[object, str] = TypeMapper()
        mapper.register(list, lambda x: f"list:{len(x)}")

        result = mapper.map([1, 2, 3])

        assert result == "list:3"

    def test_maps_dict_to_registered_handler(self) -> None:
        """Given a dict, dispatch to the dict handler."""
        from rlm.domain.models.type_mapping import TypeMapper

        mapper: TypeMapper[object, str] = TypeMapper()
        mapper.register(dict, lambda x: f"dict:{len(x)}")

        result = mapper.map({"a": 1, "b": 2})

        assert result == "dict:2"

    def test_first_matching_handler_wins(self) -> None:
        """When multiple handlers could match, first registered wins."""
        from rlm.domain.models.type_mapping import TypeMapper

        mapper: TypeMapper[object, str] = TypeMapper()
        mapper.register(str, lambda x: "string")
        mapper.register(object, lambda x: "object")  # str is also object

        result = mapper.map("hello")

        # str was registered first, should win
        assert result == "string"


class TestTypeMapperDefaultHandler:
    """Default/fallback behavior for unregistered types."""

    def test_default_handler_for_unregistered_type(self) -> None:
        """Use default handler when no specific handler matches."""
        from rlm.domain.models.type_mapping import TypeMapper

        mapper: TypeMapper[object, str] = TypeMapper()
        mapper.register(str, lambda x: f"string:{x}")
        mapper.default(lambda x: f"default:{type(x).__name__}")

        result = mapper.map(42)  # int not registered

        assert result == "default:int"

    def test_raises_type_error_without_default(self) -> None:
        """Raise TypeError when no handler matches and no default set."""
        from rlm.domain.models.type_mapping import TypeMapper

        mapper: TypeMapper[object, str] = TypeMapper()
        mapper.register(str, lambda x: f"string:{x}")

        with pytest.raises(TypeError, match="No handler registered for int"):
            mapper.map(42)

    def test_default_does_not_override_specific_handler(self) -> None:
        """Default only used when no specific handler matches."""
        from rlm.domain.models.type_mapping import TypeMapper

        mapper: TypeMapper[object, str] = TypeMapper()
        mapper.register(str, lambda x: "specific")
        mapper.default(lambda x: "default")

        result = mapper.map("hello")

        assert result == "specific"


class TestTypeMapperChaining:
    """Fluent API for building mappers."""

    def test_register_returns_self_for_chaining(self) -> None:
        """register() returns self to allow method chaining."""
        from rlm.domain.models.type_mapping import TypeMapper

        mapper: TypeMapper[object, str] = TypeMapper()
        result = mapper.register(str, lambda x: "s")

        assert result is mapper

    def test_default_returns_self_for_chaining(self) -> None:
        """default() returns self to allow method chaining."""
        from rlm.domain.models.type_mapping import TypeMapper

        mapper: TypeMapper[object, str] = TypeMapper()
        result = mapper.default(lambda x: "d")

        assert result is mapper

    def test_fluent_builder_pattern(self) -> None:
        """Build a complete mapper with method chaining."""
        from rlm.domain.models.type_mapping import TypeMapper

        mapper = (
            TypeMapper[object, str]()
            .register(str, lambda x: "string")
            .register(int, lambda x: "int")
            .register(list, lambda x: "list")
            .default(lambda x: "unknown")
        )

        assert mapper.map("hello") == "string"
        assert mapper.map(42) == "int"
        assert mapper.map([1, 2]) == "list"
        assert mapper.map(3.14) == "unknown"


class TestTypeMapperCanHandle:
    """Introspection - check if a value can be handled."""

    def test_can_handle_registered_type(self) -> None:
        """Returns True for types with registered handlers."""
        from rlm.domain.models.type_mapping import TypeMapper

        mapper: TypeMapper[object, str] = TypeMapper()
        mapper.register(str, lambda x: "s")

        assert mapper.can_handle("hello") is True

    def test_cannot_handle_unregistered_type_without_default(self) -> None:
        """Returns False for unregistered types without default."""
        from rlm.domain.models.type_mapping import TypeMapper

        mapper: TypeMapper[object, str] = TypeMapper()
        mapper.register(str, lambda x: "s")

        assert mapper.can_handle(42) is False

    def test_can_handle_any_type_with_default(self) -> None:
        """Returns True for any type when default is set."""
        from rlm.domain.models.type_mapping import TypeMapper

        mapper: TypeMapper[object, str] = TypeMapper()
        mapper.default(lambda x: "d")

        assert mapper.can_handle(42) is True
        assert mapper.can_handle("hello") is True
        assert mapper.can_handle([1, 2, 3]) is True


class TestTypeMapperRealWorldUseCases:
    """Tests that mirror actual codebase patterns we're replacing."""

    def test_context_type_inference_pattern(self) -> None:
        """Pattern from query_metadata.py - infer context type and size."""
        from rlm.domain.models.type_mapping import TypeMapper

        # This replaces the C901 complexity in query_metadata.py
        ContextInfo = tuple[str, int]  # (type_name, size_estimate)

        context_mapper: TypeMapper[object, ContextInfo] = TypeMapper()
        context_mapper.register(str, lambda x: ("str", len(x)))
        context_mapper.register(dict, lambda x: ("dict", len(json.dumps(x, default=str))))
        context_mapper.register(list, lambda x: ("list", len(json.dumps(x, default=str))))
        context_mapper.default(lambda x: ("unknown", len(repr(x))))

        assert context_mapper.map("hello world") == ("str", 11)
        assert context_mapper.map({"key": "value"})[0] == "dict"
        assert context_mapper.map([1, 2, 3])[0] == "list"
        assert context_mapper.map(42)[0] == "unknown"

    def test_serialization_pattern(self) -> None:
        """Pattern from serialization.py - serialize_value."""
        from dataclasses import dataclass

        from rlm.domain.models.type_mapping import TypeMapper

        @dataclass
        class Point:
            x: int
            y: int

        # This replaces the isinstance chain in serialize_value
        serialize_mapper: TypeMapper[object, object] = TypeMapper()
        serialize_mapper.register(type(None), lambda x: None)
        serialize_mapper.register(bool, lambda x: x)
        serialize_mapper.register(int, lambda x: x)
        serialize_mapper.register(float, lambda x: x)
        serialize_mapper.register(str, lambda x: x)
        serialize_mapper.register(list, lambda x: [serialize_mapper.map(i) for i in x])
        serialize_mapper.register(
            dict,
            lambda x: {k: serialize_mapper.map(v) for k, v in x.items()},
        )
        serialize_mapper.default(lambda x: repr(x))

        assert serialize_mapper.map(None) is None
        assert serialize_mapper.map(True) is True
        assert serialize_mapper.map(42) == 42
        assert serialize_mapper.map("hello") == "hello"
        assert serialize_mapper.map([1, 2, 3]) == [1, 2, 3]
        assert serialize_mapper.map({"a": 1}) == {"a": 1}
        # Local class repr includes qualified name, just check it ends with expected pattern
        point_repr = serialize_mapper.map(Point(1, 2))
        assert isinstance(point_repr, str)
        assert point_repr.endswith("Point(x=1, y=2)")

    def test_wire_message_dispatch_pattern(self) -> None:
        """
        Pattern from messages.py - dispatch on message type field.

        Note: TypeMapper dispatches on runtime type of VALUE, not type identity.
        For mapping type objects to schemas, use a simple dict instead:
            schema_dict = {str: {"type": "string"}, int: {"type": "integer"}}
        """
        from dataclasses import dataclass

        from rlm.domain.models.type_mapping import TypeMapper

        @dataclass
        class RequestMessage:
            prompt: str

        @dataclass
        class ResponseMessage:
            text: str

        @dataclass
        class ErrorMessage:
            error: str

        # Dispatch based on message instance type
        message_mapper: TypeMapper[object, str] = TypeMapper()
        message_mapper.register(RequestMessage, lambda m: f"REQUEST: {m.prompt}")
        message_mapper.register(ResponseMessage, lambda m: f"RESPONSE: {m.text}")
        message_mapper.register(ErrorMessage, lambda m: f"ERROR: {m.error}")

        assert message_mapper.map(RequestMessage("hello")) == "REQUEST: hello"
        assert message_mapper.map(ResponseMessage("world")) == "RESPONSE: world"
        assert message_mapper.map(ErrorMessage("oops")) == "ERROR: oops"


class TestTypeMapperSubclassHandling:
    """Subclass/inheritance behavior."""

    def test_subclass_matches_parent_handler(self) -> None:
        """Subclass instances match parent class handlers."""
        from rlm.domain.models.type_mapping import TypeMapper

        class Animal:
            pass

        class Dog(Animal):
            pass

        mapper: TypeMapper[object, str] = TypeMapper()
        mapper.register(Animal, lambda x: "animal")

        # Dog is a subclass of Animal, should match
        assert mapper.map(Dog()) == "animal"

    def test_specific_subclass_handler_takes_priority(self) -> None:
        """More specific handler registered first takes priority."""
        from rlm.domain.models.type_mapping import TypeMapper

        class Animal:
            pass

        class Dog(Animal):
            pass

        mapper: TypeMapper[object, str] = TypeMapper()
        mapper.register(Dog, lambda x: "dog")  # More specific, registered first
        mapper.register(Animal, lambda x: "animal")

        assert mapper.map(Dog()) == "dog"
        assert mapper.map(Animal()) == "animal"
