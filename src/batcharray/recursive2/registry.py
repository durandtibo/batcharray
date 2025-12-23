r"""Define the transformer registry."""

from __future__ import annotations

__all__ = ["TransformerRegistry"]

from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from batcharray.recursive2.transformer import DefaultTransformer

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from batcharray.recursive2.transformer import BaseTransformer


class TransformerRegistry:
    """Registry that manages and dispatches transformers based on data
    type.

    Similar to IteratorRegistry from the DFS pattern, this maintains a mapping
    from types to transformers and uses MRO for lookup.

    Example usage:

    ```pycon

    >>> from batcharray.recursive2 import TransformerRegistry
    >>> from batcharray.recursive2.transformer import SequenceTransformer
    >>> registry = TransformerRegistry({list: SequenceTransformer()})
    >>> registry.register()
    >>> print(registry)
    abc
    >>> registry.transform([1, 2, 3], str)
    ['1', '2', '3']

    ```
    """

    def __init__(self, registry: dict[type, BaseTransformer[Any]] | None = None) -> None:
        self._registry: dict[type, BaseTransformer[Any]] = registry.copy() if registry else {}
        self._default_transformer: BaseTransformer[Any] = DefaultTransformer()

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

        Args:
            data_type: The data type
            transformer: The transformer instance
            exist_ok: If False, raises error if type already registered

        Raises:
            RuntimeError: if type already registered and exist_ok=False
        """
        if data_type in self._registry and not exist_ok:
            msg = (
                f"Transformer {self._registry[data_type]} already registered "
                f"for {data_type}. Use exist_ok=True to overwrite."
            )
            raise RuntimeError(msg)
        self._registry[data_type] = transformer
        # Clear cache when registry changes
        self._find_transformer_cached.cache_clear()

    def register_many(
        self,
        mapping: Mapping[type, BaseTransformer[Any]],
        exist_ok: bool = False,
    ) -> None:
        """Register multiple transformers at once.

        Args:
            mapping: Dictionary mapping types to transformers
            exist_ok: If False, raises error if any type already registered
        """
        for typ, transformer in mapping.items():
            self.register(typ, transformer, exist_ok=exist_ok)

    def has_transformer(self, data_type: type) -> bool:
        """Check if a transformer is registered for the given type."""
        return data_type in self._registry

    def find_transformer(self, data_type: type) -> BaseTransformer[Any]:
        """Find the appropriate transformer for a given type.

        Uses MRO to find the most specific registered transformer.
        Results are cached for performance.

        Args:
            data_type: The data type

        Returns:
            The transformer for this type or a parent type
        """
        # Direct lookup first (most common case)
        if data_type in self._registry:
            return self._registry[data_type]

        # MRO lookup for inheritance
        for base_type in data_type.__mro__:
            if base_type in self._registry:
                return self._registry[base_type]

        # Fall back to default
        return self._default_transformer

    def transform(self, data: Any, func: Callable[[Any], Any]) -> Any:
        """Transform data by applying func recursively.

        This is the main entry point that finds the appropriate transformer
        and delegates to it.

        Args:
            data: The data to transform
            func: Function to apply to leaf values

        Returns:
            Transformed data with same structure
        """
        transformer = self.find_transformer(type(data))
        return transformer.transform(data, func, self)
