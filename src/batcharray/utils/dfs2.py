r"""Contain code to iterate over the data to find the arrays with a
Depth-First Search (DFS) strategy."""

from __future__ import annotations

__all__ = [
    "BaseArrayIterator",
    "DefaultArrayIterator",
    "IterableArrayIterator",
    "IteratorRegistry",
    "MappingArrayIterator",
    "dfs_array",
]

import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Generator, Iterable, Mapping
from typing import Any, TypeVar

import numpy as np
from coola.utils import str_indent, str_mapping

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


def dfs_array(data: Any, registry: IteratorRegistry | None = None) -> Generator[np.ndarray]:
    r"""Perform depth-first search to find all numpy arrays in nested
    data structures.

    This function recursively traverses nested data structures (lists, dicts, tuples,
    etc.) using a depth-first search strategy and yields all numpy arrays found.
    The traversal assumes the data has a tree-like structure without circular references.

    Args:
        data: The data structure to search. Can contain nested combinations of
            lists, tuples, dicts, sets, and other registered types.
        registry: Custom iterator registry to use. If ``None``, uses the default
            global registry with pre-registered iterators for common Python types.
            Providing a custom registry allows you to handle custom data structures
            without modifying global state.

    Yields:
        All numpy arrays found within the data structure, in depth-first order.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from batcharray.utils.dfs2 import dfs_array
    >>> data = ["text", np.ones((2, 3)), 42, np.array([0, 1, 2, 3, 4])]
    >>> list(dfs_array(data))
    [array([[1., 1., 1.], [1., 1., 1.]]), array([0, 1, 2, 3, 4])]
    >>>
    >>> # With nested structures
    >>> nested = {"a": [np.array([1, 2])], "b": {"c": np.array([3, 4])}}
    >>> list(dfs_array(nested))
    [array([1, 2]), array([3, 4])]

    ```
    """
    if registry is None:
        registry = get_default_registry()
    yield from registry.iterate(data)


class BaseArrayIterator(ABC):
    r"""Base class for iterators that traverse data structures to find
    numpy arrays.

    This abstract class defines the interface for custom iterators that can be
    registered to handle specific data types during depth-first search traversal.
    Subclasses must implement the ``iterate`` method to define how their specific
    data type should be traversed.
    """

    @abstractmethod
    def iterate(self, data: Any, registry: IteratorRegistry) -> Generator[np.ndarray]:
        r"""Iterate over the data and yield numpy arrays found within.

        This method should traverse the data structure and recursively yield
        all numpy arrays contained within it. For nested structures, use
        ``registry.iterate()`` to handle nested items.

        Args:
            data: The data structure to iterate over.
            registry: The registry instance used to resolve iterators for
                nested data structures.

        Yields:
            numpy arrays found in the data structure.
        """


class DefaultArrayIterator(BaseArrayIterator):
    r"""Default iterator that handles leaf nodes and numpy arrays.

    This iterator checks if the data is a numpy array and yields it if
    so. For all other data types, it yields nothing, treating them as
    leaf nodes that don't contain arrays.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def iterate(
        self,
        data: Any,
        registry: IteratorRegistry,  # noqa: ARG002
    ) -> Generator[np.ndarray]:
        if isinstance(data, np.ndarray):
            yield data


class IterableArrayIterator(BaseArrayIterator):
    r"""Iterator for iterable data structures (lists, tuples, sets,
    etc.).

    This iterator traverses iterable objects by iterating over each
    element and recursively searching for numpy arrays within them. It's
    suitable for any object that implements the iterable protocol.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def iterate(self, data: Iterable[Any], registry: IteratorRegistry) -> Generator[np.ndarray]:
        for item in data:
            yield from registry.iterate(item)


class MappingArrayIterator(BaseArrayIterator):
    r"""Iterator for mapping data structures (dicts, OrderedDict, etc.).

    This iterator traverses mapping objects by iterating over their
    values (not keys) and recursively searching for numpy arrays. Keys
    are ignored as they typically don't contain array data.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def iterate(self, data: Mapping[Any, Any], registry: IteratorRegistry) -> Generator[np.ndarray]:
        for value in data.values():
            yield from registry.iterate(value)


class IteratorRegistry:
    r"""Registry that manages and dispatches to appropriate iterators
    based on data type.

    This class maintains a mapping from data types to their corresponding iterator
    instances. When asked to iterate over data, it finds the appropriate iterator
    using the type's Method Resolution Order (MRO) and delegates to it.

    The registry separates concerns between iterator registration/lookup and the
    actual iteration logic, making the system more maintainable and testable.

    Example usage:

    ```pycon
    >>> from batcharray.utils.dfs2 import IteratorRegistry, IterableArrayIterator
    >>> import numpy as np
    >>> registry = IteratorRegistry()
    >>> registry.register(list, IterableArrayIterator())
    >>> data = [np.array([1, 2, 3]), "text", np.array([4, 5])]
    >>> list(registry.iterate(data))
    [array([1, 2, 3]), array([4, 5])]

    ```
    """

    def __init__(self) -> None:
        self._registry: dict[type, BaseArrayIterator] = {}
        self._default_iterator: BaseArrayIterator = DefaultArrayIterator()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self._registry))}\n)"

    def register(
        self, data_type: type, iterator: BaseArrayIterator, exist_ok: bool = False
    ) -> None:
        r"""Register an iterator for a given data type.

        Args:
            data_type: The data type for this iterator.
            iterator: The iterator instance.
            exist_ok: If ``False``, raises ``RuntimeError`` if the
                data type already exists. Set to ``True`` to overwrite.

        Raises:
            RuntimeError: if an iterator is already registered for the
                data type and ``exist_ok=False``.

        Example usage:

        ```pycon
        >>> from batcharray.utils.dfs2 import IteratorRegistry, IterableArrayIterator
        >>> registry = IteratorRegistry()
        >>> registry.register(list, IterableArrayIterator(), exist_ok=True)

        ```
        """
        if data_type in self._registry and not exist_ok:
            msg = (
                f"An iterator ({self._registry[data_type]}) is already registered for "
                f"{data_type}. Use `exist_ok=True` to overwrite."
            )
            raise RuntimeError(msg)
        self._registry[data_type] = iterator

    def register_many(
        self, mapping: Mapping[type, BaseArrayIterator], exist_ok: bool = False
    ) -> None:
        r"""Register multiple iterators at once.

        Args:
            mapping: Dictionary mapping types to their iterators.
            exist_ok: If ``False``, raises ``RuntimeError`` if any
                data type already exists. Set to ``True`` to overwrite.

        Raises:
            RuntimeError: if any iterator is already registered and ``exist_ok=False``.

        Example usage:

        ```pycon
        >>> from batcharray.utils.dfs2 import IteratorRegistry, IterableArrayIterator
        >>> registry = IteratorRegistry()
        >>> registry.register_many({list: IterableArrayIterator(), tuple: IterableArrayIterator()})

        ```
        """
        for typ, iterator in mapping.items():
            self.register(typ, iterator, exist_ok=exist_ok)

    def has_iterator(self, data_type: type) -> bool:
        r"""Check if an iterator is registered for the given data type.

        Args:
            data_type: The data type to check.

        Returns:
            ``True`` if an iterator is registered, otherwise ``False``.

        Example usage:

        ```pycon
        >>> from batcharray.utils.dfs2 import IteratorRegistry
        >>> registry = IteratorRegistry()
        >>> registry.has_iterator(list)
        False

        ```
        """
        return data_type in self._registry

    def find_iterator(self, data_type: type) -> BaseArrayIterator:
        r"""Find the appropriate iterator for a given data type.

        This method uses Python's Method Resolution Order (MRO) to find the
        most specific registered iterator for the given type. If an exact match
        isn't found, it walks up the inheritance hierarchy.

        Args:
            data_type: The data type to find an iterator for.

        Returns:
            The iterator associated with the data type, one of its parent classes,
            or the default iterator if no match is found.

        Example usage:

        ```pycon
        >>> from batcharray.utils.dfs2 import get_default_registry
        >>> registry = get_default_registry()
        >>> registry.find_iterator(list)
        IterableArrayIterator()

        ```
        """
        # Direct lookup first (most common case)
        if data_type in self._registry:
            return self._registry[data_type]

        # MRO lookup for inheritance
        for base_type in data_type.__mro__:  # Skip the type itself
            if base_type in self._registry:
                return self._registry[base_type]

        # Fall back to default
        return self._default_iterator

    def iterate(self, data: Any) -> Generator[np.ndarray]:
        r"""Perform depth-first iteration over data to find all numpy
        arrays.

        This method finds the appropriate iterator for the data's type and
        delegates to it. The iterator will recursively traverse the data
        structure, yielding all numpy arrays found.

        Args:
            data: The data structure to iterate over. Can be any type with
                a registered iterator.

        Yields:
            All numpy arrays found within the data structure.

        Example usage:

        ```pycon
        >>> from batcharray.utils.dfs2 import IteratorRegistry
        >>> import numpy as np
        >>> registry = IteratorRegistry()
        >>> data = {"a": np.array([1, 2]), "b": [np.array([3, 4]), "text"]}
        >>> list(registry.iterate(data))
        [array([1, 2]), array([3, 4])]

        ```
        """
        iterator = self.find_iterator(type(data))
        yield from iterator.iterate(data, self)


def get_default_registry() -> IteratorRegistry:
    r"""Get or lazily create the default global registry instance.

    This function returns a singleton registry instance that is shared across
    all calls. The registry is initialized on first access with default iterators
    for common Python data structures. Subsequent calls return the same instance.

    Returns:
        The default global registry instance with pre-registered iterators
        for common types (list, dict, tuple, set, etc.).

    Example usage:

    ```pycon
    >>> from batcharray.utils.dfs2 import get_default_registry
    >>> registry = get_default_registry()
    >>> registry.has_iterator(list)
    True

    ```
    """
    if not hasattr(get_default_registry, "_registry"):
        registry = IteratorRegistry()
        _register_default_iterators(registry)
        get_default_registry._registry = registry
    return get_default_registry._registry


def _register_default_iterators(registry: IteratorRegistry) -> None:
    r"""Register default iterators for common Python data structures.

    This internal function populates a registry with iterators for standard
    Python types. It registers:
    - DefaultArrayIterator for leaf nodes (object, str)
    - IterableArrayIterator for sequences (list, tuple, set, deque)
    - MappingArrayIterator for mappings (dict, Mapping)

    Args:
        registry: The registry to populate with default iterators.
    """
    default = DefaultArrayIterator()
    iterable = IterableArrayIterator()
    mapping = MappingArrayIterator()

    registry.register_many(
        {
            # Object is the catch-all base
            object: default,
            # Strings should not be iterated character by character
            str: default,
            # Collections
            Iterable: iterable,
            list: iterable,
            tuple: iterable,
            set: iterable,
            deque: iterable,
            # Mappings
            Mapping: mapping,
            dict: mapping,
        }
    )


# For backwards compatibility - users can still do module-level registration
def register_iterators(mapping: Mapping[type, BaseArrayIterator], exist_ok: bool = False) -> None:
    r"""Register multiple iterators to the default global registry.

    This is a convenience function for registering custom iterators
    that will be used by the default ``dfs_array`` function.

    Args:
        mapping: Dictionary mapping data types to their iterator instances.
        exist_ok: If ``False``, raises ``RuntimeError`` if any data type
            already has a registered iterator. Set to ``True`` to overwrite
            existing registrations.

    Raises:
        RuntimeError: if any iterator is already registered and ``exist_ok=False``.

    Example usage:

    ```pycon
    >>> from batcharray.utils.dfs2 import register_iterators, IterableArrayIterator
    >>> class MyCustomList(list):
    ...     pass
    ...
    >>> register_iterators({MyCustomList: IterableArrayIterator()}, exist_ok=True)

    ```
    """
    get_default_registry().register_many(mapping, exist_ok=exist_ok)
