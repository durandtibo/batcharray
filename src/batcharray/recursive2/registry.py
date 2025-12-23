r"""Define the transformer registry for recursive data transformation.

This module provides a registry system that manages and dispatches
transformers based on data types, enabling recursive transformation of
nested data structures while preserving their original structure.
"""

from __future__ import annotations

__all__ = ["TransformerRegistry"]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from batcharray.recursive2.transformer import (
    DefaultTransformer,
    MappingTransformer,
    SequenceTransformer,
    SetTransformer,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from batcharray.recursive2.transformer import BaseTransformer


class TransformerRegistry:
    """Registry that manages and dispatches transformers based on data
    type.

    This registry maintains a mapping from Python types to transformer instances
    and uses the Method Resolution Order (MRO) for type lookup. When transforming
    data, it automatically selects the most specific registered transformer for
    the data's type, falling back to parent types or a default transformer if needed.

    The registry includes an LRU cache for type lookups to optimize performance
    in applications that repeatedly transform similar data structures.

    Args:
        registry: Optional initial mapping of types to transformers. If provided,
            the registry is copied to prevent external mutations.

    Attributes:
        _registry: Internal mapping of registered types to transformers
        _default_transformer: Fallback transformer for unregistered types
        _find_transformer_cached: Cached version of transformer lookup

    Example:
        Basic usage with a sequence transformer:

        ```pycon
        >>> from batcharray.recursive2 import TransformerRegistry
        >>> from batcharray.recursive2.transformer import SequenceTransformer
        >>> registry = TransformerRegistry({list: SequenceTransformer()})
        >>> registry
        TransformerRegistry(
          (<class 'list'>): SequenceTransformer()
        )
        >>> registry.transform([1, 2, 3], str)
        ['1', '2', '3']

        ```

        Registering custom transformers:

        ```pycon
        >>> from batcharray.recursive2 import TransformerRegistry
        >>> from batcharray.recursive2.transformer import SequenceTransformer
        >>> registry = TransformerRegistry()
        >>> registry.register(list, SequenceTransformer())
        >>> registry.transform([1, 2, 3], lambda x: x * 2)
        [2, 4, 6]

        ```

        Working with nested structures:

        ```pycon
        >>> from batcharray.recursive2 import get_default_registry
        >>> registry = get_default_registry()
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> registry.transform(data, str)
        {'a': ['1', '2'], 'b': ['3', '4']}

        ```
    """

    def __init__(self, registry: dict[type, BaseTransformer[Any]] | None = None) -> None:
        self._registry: dict[type, BaseTransformer[Any]] = registry.copy() if registry else {}
        self._default_transformer: BaseTransformer[Any] = DefaultTransformer()

        # cache for type lookups - improves performance for repeated transforms
        self._transformer_cache: dict[type, BaseTransformer[Any]] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {repr_indent(repr_mapping(self._registry))}\n)"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self._registry))}\n)"

    def register(
        self,
        data_type: type,
        transformer: BaseTransformer[Any],
        exist_ok: bool = False,
    ) -> None:
        """Register a transformer for a given data type.

        This method associates a transformer instance with a specific Python type.
        When data of this type is transformed, the registered transformer will be used.
        The cache is automatically cleared after registration to ensure consistency.

        Args:
            data_type: The Python type to register (e.g., list, dict, custom classes)
            transformer: The transformer instance that handles this type
            exist_ok: If False (default), raises an error if the type is already
                registered. If True, overwrites the existing registration silently.

        Raises:
            RuntimeError: If the type is already registered and exist_ok is False

        Example:
            ```pycon
            >>> from batcharray.recursive2 import TransformerRegistry
            >>> from batcharray.recursive2.transformer import SequenceTransformer
            >>> registry = TransformerRegistry()
            >>> registry.register(list, SequenceTransformer())
            >>> registry.has_transformer(list)
            True

            ```
        """
        if data_type in self._registry and not exist_ok:
            msg = (
                f"Transformer {self._registry[data_type]} already registered "
                f"for {data_type}. Use exist_ok=True to overwrite."
            )
            raise RuntimeError(msg)
        self._registry[data_type] = transformer
        # Clear cache when registry changes to ensure new registrations are used
        self._transformer_cache.clear()

    def register_many(
        self,
        mapping: Mapping[type, BaseTransformer[Any]],
        exist_ok: bool = False,
    ) -> None:
        """Register multiple transformers at once.

        This is a convenience method for bulk registration that internally calls
        register() for each type-transformer pair.

        Args:
            mapping: Dictionary mapping Python types to transformer instances
            exist_ok: If False (default), raises an error if any type is already
                registered. If True, overwrites existing registrations silently.

        Raises:
            RuntimeError: If any type is already registered and exist_ok is False

        Example:
            ```pycon
            >>> from batcharray.recursive2 import TransformerRegistry
            >>> from batcharray.recursive2.transformer import SequenceTransformer
            >>> registry = TransformerRegistry()
            >>> registry.register_many(
            ...     {
            ...         list: SequenceTransformer(),
            ...         dict: MappingTransformer(),
            ...     }
            ... )
            >>> registry
            TransformerRegistry(
              (<class 'list'>): SequenceTransformer()
              (<class 'dict'>): MappingTransformer()
            )

            ```
        """
        for typ, transformer in mapping.items():
            self.register(typ, transformer, exist_ok=exist_ok)

    def has_transformer(self, data_type: type) -> bool:
        """Check if a transformer is explicitly registered for the given
        type.

        Note that this only checks for direct registration. Even if this returns
        False, find_transformer() may still return a transformer via MRO lookup
        or the default transformer.

        Args:
            data_type: The type to check

        Returns:
            True if a transformer is explicitly registered for this type,
            False otherwise

        Example:
            ```pycon
            >>> from batcharray.recursive2 import TransformerRegistry
            >>> from batcharray.recursive2.transformer import SequenceTransformer
            >>> registry = TransformerRegistry()
            >>> registry.register(list, SequenceTransformer())
            >>> registry.has_transformer(list)
            True
            >>> registry.has_transformer(tuple)
            False

            ```
        """
        return data_type in self._registry

    def _find_transformer_uncached(self, data_type: type) -> BaseTransformer[Any]:
        """Find transformer using MRO (uncached version).

        This is the internal implementation that performs the actual lookup.
        It first checks for a direct match, then walks the MRO to find the
        most specific registered transformer, and finally falls back to the
        default transformer.

        Args:
            data_type: The type to find a transformer for

        Returns:
            The appropriate transformer instance
        """
        # Direct lookup first (most common case, O(1))
        if data_type in self._registry:
            return self._registry[data_type]

        # MRO lookup for inheritance - finds the most specific parent type
        for base_type in data_type.__mro__:
            if base_type in self._registry:
                return self._registry[base_type]

        # Fall back to default transformer for unregistered types
        return self._default_transformer

    def find_transformer(self, data_type: type) -> BaseTransformer[Any]:
        """Find the appropriate transformer for a given type.

        Uses the Method Resolution Order (MRO) to find the most specific
        registered transformer. For example, if you register a transformer
        for Sequence but not for list, lists will use the Sequence transformer.

        Results are cached using an LRU cache (256 entries) for performance,
        as transformer lookup is a hot path in recursive transformations.

        Args:
            data_type: The Python type to find a transformer for

        Returns:
            The most specific registered transformer for this type, a parent
            type's transformer via MRO, or the default transformer

        Example:

            ```pycon
            >>> from batcharray.recursive2 import TransformerRegistry
            >>> from batcharray.recursive2.transformer import SequenceTransformer
            >>> registry = TransformerRegistry()
            >>> registry.register(Sequence, SequenceTransformer())
            >>> # list does not inherit from Sequence, so it uses DefaultTransformer
            >>> transformer = registry.find_transformer(list)
            >>> transformer
            DefaultTransformer()

            ```
        """
        if data_type not in self._transformer_cache:
            self._transformer_cache[data_type] = self._find_transformer_uncached(data_type)
        return self._transformer_cache[data_type]

    def transform(self, data: Any, func: Callable[[Any], Any]) -> Any:
        """Transform data by applying a function recursively through the
        structure.

        This is the main entry point for transformation. It automatically:
        1. Determines the data's type
        2. Finds the appropriate transformer
        3. Delegates to that transformer's transform method
        4. The transformer recursively processes nested structures

        The original structure of the data is preserved - only the leaf values
        are transformed by the provided function.

        Args:
            data: The data to transform (can be nested: lists, dicts, tuples, etc.)
            func: Function to apply to leaf values. Should accept one argument
                and return a transformed value.

        Returns:
            Transformed data with the same structure as the input but with
            leaf values transformed by func

        Example:
            Converting all numbers to strings in a nested structure:

            ```pycon
            >>> registry = get_default_registry()
            >>> data = {"scores": [95, 87, 92], "name": "test"}
            >>> registry.transform(data, str)
            {'scores': ['95', '87', '92'], 'name': 'test'}

            ```

            Doubling all numeric values:

            ```pycon
            >>> data = [1, [2, 3], {"a": 4}]
            >>> registry.transform(data, lambda x: x * 2 if isinstance(x, (int, float)) else x)
            [2, [4, 6], {'a': 8}]

            ```
        """
        transformer = self.find_transformer(type(data))
        return transformer.transform(data, func, self)


def get_default_registry() -> TransformerRegistry:
    """Get or create the default global registry with common Python
    types.

    Returns a singleton registry instance that is pre-configured with transformers
    for Python's built-in types including sequences (list, tuple), mappings (dict),
    sets, and scalar types (int, float, str, bool).

    This function uses a singleton pattern to ensure the same registry instance
    is returned on subsequent calls, which is efficient and maintains consistency
    across an application.

    Returns:
        A TransformerRegistry instance with transformers registered for:
            - Scalar types (int, float, complex, bool, str)
            - Sequences (list, tuple, Sequence ABC)
            - Sets (set, frozenset)
            - Mappings (dict, Mapping ABC)

    Example:
        ```pycon
        >>> from batcharray.recursive2 import get_default_registry
        >>> registry = get_default_registry()
        >>> # Registry is ready to use with common Python types
        >>> registry.transform([1, 2, 3], str)
        ['1', '2', '3']
        >>> registry.transform({"a": 1, "b": 2}, lambda x: x * 10)
        {'a': 10, 'b': 20}

        ```

    Note:
        The singleton pattern means modifications to the returned registry
        affect all future calls to this function. If you need an isolated
        registry, create a new TransformerRegistry instance directly.
    """
    if not hasattr(get_default_registry, "_registry"):
        registry = TransformerRegistry()
        _register_default_transformers(registry)
        get_default_registry._registry = registry
    return get_default_registry._registry


def _register_default_transformers(registry: TransformerRegistry) -> None:
    """Register default transformers for common Python types.

    This internal function sets up the standard type-to-transformer mappings
    used by the default registry. It registers transformers in a specific order
    to ensure proper inheritance handling via MRO.

    The registration strategy:
        - Scalar types use DefaultTransformer (no recursion)
        - Strings use DefaultTransformer to prevent character-by-character iteration
        - Sequences use SequenceTransformer for recursive list/tuple processing
        - Sets use SetTransformer for recursive set processing
        - Mappings use MappingTransformer for recursive dict processing

    Args:
        registry: The registry to populate with default transformers

    Note:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    default = DefaultTransformer()
    sequence = SequenceTransformer()
    set_transformer = SetTransformer()
    mapping = MappingTransformer()

    registry.register_many(
        {
            # Object is the catch-all base for unregistered types
            object: default,
            # Strings should not be iterated character by character
            str: default,
            # Numeric types - no recursion needed
            int: default,
            float: default,
            complex: default,
            bool: default,
            # Sequences - recursive transformation preserving order
            list: sequence,
            tuple: sequence,
            Sequence: sequence,
            # Sets - recursive transformation without order
            set: set_transformer,
            frozenset: set_transformer,
            # Mappings - recursive transformation of keys and values
            dict: mapping,
            Mapping: mapping,
        }
    )
