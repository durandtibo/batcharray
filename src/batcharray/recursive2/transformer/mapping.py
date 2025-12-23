r"""Define the default transformer for mapping data (dict)."""

from __future__ import annotations

__all__ = ["MappingTransformer"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from batcharray.recursive2.transformer.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from batcharray.recursive2.registry import TransformerRegistry


class MappingTransformer(BaseTransformer[Mapping[Any, Any]]):
    """Transformer for mappings (dict).

    Recursively transforms values (not keys) and rebuilds the mapping.

    Example usage:

    ```pycon

    >>> from batcharray.recursive2.transformer import MappingTransformer
    >>> from batcharray.recursive2 import TransformerRegistry
    >>> registry = TransformerRegistry()
    >>> transformer = MappingTransformer()
    >>> transformer
    MappingTransformer()
    >>> transformer.transform({"a": 1, "b": 2}, func=str, registry=registry)
    {'a': '1', 'b': '2'}

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        data: Mapping[Any, Any],
        func: Callable[[Any], Any],
        registry: TransformerRegistry,
    ) -> Mapping[Any, Any]:
        # Transform all values
        transformed = {key: registry.transform(value, func) for key, value in data.items()}

        # Rebuild with original type
        return type(data)(transformed)
