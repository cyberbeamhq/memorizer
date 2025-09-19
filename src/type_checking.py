"""
type_checking.py
Runtime type checking utilities for the Memorizer framework.
Provides type validation and runtime type checking capabilities.
"""

import inspect
import logging
from functools import wraps
from typing import Any, Type, TypeVar, Union, get_args, get_origin, get_type_hints

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TypeCheckError(Exception):
    """Exception raised when type checking fails."""

    pass


def validate_type(value: Any, expected_type: Type[T], name: str = "value") -> T:
    """
    Validate that a value matches the expected type.

    Args:
        value: The value to validate
        expected_type: The expected type
        name: Name of the parameter for error messages

    Returns:
        The validated value

    Raises:
        TypeCheckError: If the value doesn't match the expected type
    """
    if expected_type is Any:
        return value

    # Handle None values
    if value is None:
        if expected_type is type(None):
            return value
        # Check if None is allowed (Union with None)
        origin = get_origin(expected_type)
        if origin is Union:
            args = get_args(expected_type)
            if type(None) in args:
                return value
        raise TypeCheckError(f"{name} cannot be None")

    # Handle basic types
    if isinstance(value, expected_type):
        return value

    # Handle Union types
    origin = get_origin(expected_type)
    if origin is Union:
        args = get_args(expected_type)
        for arg in args:
            try:
                return validate_type(value, arg, name)
            except TypeCheckError:
                continue
        raise TypeCheckError(f"{name} must be one of {args}, got {type(value)}")

    # Handle List types
    if origin is list:
        if not isinstance(value, list):
            raise TypeCheckError(f"{name} must be a list, got {type(value)}")
        args = get_args(expected_type)
        if args:
            element_type = args[0]
            for i, item in enumerate(value):
                validate_type(item, element_type, f"{name}[{i}]")
        return value

    # Handle Dict types
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeCheckError(f"{name} must be a dict, got {type(value)}")
        args = get_args(expected_type)
        if len(args) >= 2:
            key_type, value_type = args[0], args[1]
            for key, val in value.items():
                validate_type(key, key_type, f"{name}.key")
                validate_type(val, value_type, f"{name}.{key}")
        return value

    # Handle Optional types
    if expected_type is type(None):
        if value is not None:
            raise TypeCheckError(f"{name} must be None, got {type(value)}")
        return value

    # Handle generic types
    if hasattr(expected_type, "__origin__"):
        origin = expected_type.__origin__
        if origin is Union:
            args = expected_type.__args__
            for arg in args:
                try:
                    return validate_type(value, arg, name)
                except TypeCheckError:
                    continue
            raise TypeCheckError(f"{name} must be one of {args}, got {type(value)}")

    # If we get here, the type doesn't match
    raise TypeCheckError(f"{name} must be {expected_type}, got {type(value)}")


def type_check(func):
    """
    Decorator to add runtime type checking to functions.

    This decorator validates that function arguments and return values
    match their type hints at runtime.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints
        hints = get_type_hints(func)

        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Validate arguments
        for param_name, value in bound_args.arguments.items():
            if param_name in hints:
                expected_type = hints[param_name]
                try:
                    validate_type(value, expected_type, param_name)
                except TypeCheckError as e:
                    logger.error(
                        f"Type check failed for {func.__name__}.{param_name}: {e}"
                    )
                    raise

        # Call the function
        result = func(*args, **kwargs)

        # Validate return value
        if "return" in hints:
            expected_return_type = hints["return"]
            try:
                validate_type(result, expected_return_type, "return value")
            except TypeCheckError as e:
                logger.error(f"Type check failed for {func.__name__} return value: {e}")
                raise

        return result

    return wrapper


def type_check_class(cls):
    """
    Class decorator to add runtime type checking to all methods.
    """
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith("_"):  # Skip private methods
            setattr(cls, name, type_check(method))
    return cls


def validate_dict_structure(
    data: dict, expected_structure: dict, name: str = "data"
) -> dict:
    """
    Validate that a dictionary has the expected structure.

    Args:
        data: The dictionary to validate
        expected_structure: Dictionary with expected keys and types
        name: Name of the parameter for error messages

    Returns:
        The validated dictionary

    Raises:
        TypeCheckError: If the structure doesn't match
    """
    if not isinstance(data, dict):
        raise TypeCheckError(f"{name} must be a dictionary, got {type(data)}")

    # Check required keys
    for key, expected_type in expected_structure.items():
        if key not in data:
            raise TypeCheckError(f"{name} is missing required key '{key}'")

        try:
            validate_type(data[key], expected_type, f"{name}.{key}")
        except TypeCheckError as e:
            raise TypeCheckError(f"Validation failed for {name}.{key}: {e}")

    return data


def validate_list_items(
    items: list, expected_type: Type[T], name: str = "items"
) -> list:
    """
    Validate that all items in a list match the expected type.

    Args:
        items: The list to validate
        expected_type: The expected type for each item
        name: Name of the parameter for error messages

    Returns:
        The validated list

    Raises:
        TypeCheckError: If any item doesn't match the expected type
    """
    if not isinstance(items, list):
        raise TypeCheckError(f"{name} must be a list, got {type(items)}")

    for i, item in enumerate(items):
        try:
            validate_type(item, expected_type, f"{name}[{i}]")
        except TypeCheckError as e:
            raise TypeCheckError(f"Validation failed for {name}[{i}]: {e}")

    return items


def is_optional_type(type_hint: Type) -> bool:
    """
    Check if a type hint represents an optional type (Union with None).

    Args:
        type_hint: The type hint to check

    Returns:
        True if the type is optional, False otherwise
    """
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        return type(None) in args
    return False


def get_optional_type(type_hint: Type) -> Type:
    """
    Get the non-None type from an optional type hint.

    Args:
        type_hint: The optional type hint

    Returns:
        The non-None type, or the original type if not optional

    Raises:
        ValueError: If the type hint is not optional
    """
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return non_none_args[0]
        elif len(non_none_args) > 1:
            return Union[tuple(non_none_args)]

    raise ValueError(f"Type hint {type_hint} is not optional")


def runtime_type_check_enabled() -> bool:
    """
    Check if runtime type checking is enabled.

    Returns:
        True if runtime type checking is enabled, False otherwise
    """
    return os.getenv("ENABLE_RUNTIME_TYPE_CHECKING", "false").lower() == "true"


# Import os for environment variable access
import os


# Conditional type checking decorator
def conditional_type_check(func):
    """
    Decorator that only applies type checking if enabled via environment variable.
    """
    if runtime_type_check_enabled():
        return type_check(func)
    return func


def conditional_type_check_class(cls):
    """
    Class decorator that only applies type checking if enabled via environment variable.
    """
    if runtime_type_check_enabled():
        return type_check_class(cls)
    return cls
