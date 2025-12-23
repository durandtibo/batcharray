r"""Define the default transformer for sequences data (list, tuple)."""

from __future__ import annotations

__all__ = ["SequenceTransformer"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from batcharray.recursive2.transformer.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from batcharray.recursive2.registry import TransformerRegistry


class SequenceTransformer(BaseTransformer[Sequence[Any]]):
    """Transformer for sequences (list, tuple).

    Recursively transforms elements and rebuilds the sequence.

    Example usage:

    ```pycon

    >>> from batcharray.recursive2.transformer import SequenceTransformer
    >>> from batcharray.recursive2 import TransformerRegistry
    >>> registry = TransformerRegistry()
    >>> transformer = SequenceTransformer()
    >>> transformer
    SequenceTransformer()
    >>> transformer.transform([1, 2, 3], func=str, registry=registry)
    ['1', '2', '3']

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        data: Sequence[Any],
        func: Callable[[Any], Any],
        registry: TransformerRegistry,
    ) -> Sequence[Any]:
        # Transform all elements
        transformed = [registry.transform(item, func) for item in data]

        # Rebuild with original type
        if isinstance(data, tuple):
            # Handle named tuples
            if hasattr(data, "_fields"):
                return type(data)(*transformed)
            return tuple(transformed)

        return type(data)(transformed)
