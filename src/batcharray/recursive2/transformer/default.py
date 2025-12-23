r"""Define the default transformer."""

from __future__ import annotations

__all__ = ["DefaultTransformer"]

from typing import TYPE_CHECKING, Any

from batcharray.recursive2.transformer.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from batcharray.recursive2.registry import TransformerRegistry


class DefaultTransformer(BaseTransformer[Any]):
    """Transformer for leaf nodes - just applies the function.

    Example usage:

    ```pycon

    >>> from batcharray.recursive2.transformer import DefaultTransformer
    >>> from batcharray.recursive2 import TransformerRegistry
    >>> registry = TransformerRegistry()
    >>> transformer = DefaultTransformer()
    >>> transformer
    DefaultTransformer()
    >>> transformer.transform([1, 2, 3], func=str, registry=registry)
    '[1, 2, 3]'

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def transform(
        self,
        data: Any,
        func: Callable[[Any], Any],
        registry: TransformerRegistry,  # noqa: ARG002
    ) -> Any:
        return func(data)
