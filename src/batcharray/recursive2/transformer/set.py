r"""Define the default transformer for set data (set, frozenset)."""

from __future__ import annotations

__all__ = ["SetTransformer"]

from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Any

from batcharray.recursive2.transformer.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from batcharray.recursive2.registry import TransformerRegistry


class SetTransformer(BaseTransformer[AbstractSet[Any]]):
    """Transformer for sets (set, frozenset).

    Note: Transformed values must remain hashable.

    Example usage:

    ```pycon

    >>> from batcharray.recursive2.transformer import SetTransformer
    >>> from batcharray.recursive2 import TransformerRegistry
    >>> registry = TransformerRegistry()
    >>> transformer = SetTransformer()
    >>> transformer
    SetTransformer()
    >>> transformer.transform({1}, func=str, registry=registry)
    {'1'}

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        data: AbstractSet[Any],
        func: Callable[[Any], Any],
        registry: TransformerRegistry,
    ) -> AbstractSet[Any]:
        # Transform all elements
        transformed = {registry.transform(item, func) for item in data}

        # Rebuild with original type
        return type(data)(transformed)
