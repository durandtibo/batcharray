r"""Define the public interface."""

from __future__ import annotations

__all__ = ["recursive_apply", "register_transformers"]

from typing import TYPE_CHECKING, Any

from batcharray.recursive2.registry import TransformerRegistry, get_default_registry

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from batcharray.recursive2.transformer import BaseTransformer


def recursive_apply(
    data: Any, func: Callable[[Any], Any], registry: TransformerRegistry | None = None
) -> Any:
    """Recursively apply a function to all items in nested data.

    This is the main public interface that maintains compatibility
    with the original implementation.

    Args:
        data: Input data (can be nested)
        func: Function to apply to each leaf value
        registry: Registry to resolve transformers for nested data.

    Returns:
        Transformed data with same structure as input

    Example usage:

    ```pycon
    >>> from batcharray.recursive2 import recursive_apply
    >>> recursive_apply({"a": 1, "b": "abc"}, str)
    {'a': '1', 'b': 'abc'}
    >>> recursive_apply([1, [2, 3], {"x": 4}], lambda x: x * 2)
    [2, [4, 6], {'x': 8}]

    ```
    """
    if registry is None:
        registry = get_default_registry()
    return registry.transform(data, func)


def register_transformers(
    mapping: Mapping[type, BaseTransformer[Any]],
    exist_ok: bool = False,
) -> None:
    """Register custom transformers to the default global registry.

    This allows users to add support for custom types without modifying
    global state directly.

    Args:
        mapping: Dictionary mapping types to transformer instances
        exist_ok: If False, raises error if any type already registered

    Example usage:

    ```pycon
    >>> from batcharray.recursive2 import register_transformers
    >>> from batcharray.recursive2.transformer import BaseTransformer
    >>> class MyType:
    ...     def __init__(self, value):
    ...         self.value = value
    ...
    >>> class MyTransformer(BaseTransformer):
    ...     def transform(self, data, func, registry):
    ...         return MyType(func(data.value))
    ...
    >>> register_transformers({MyType: MyTransformer()})

    ```
    """
    get_default_registry().register_many(mapping, exist_ok=exist_ok)
