r"""Define a conditional transformer to add filtering without changing
the core design."""

from __future__ import annotations

__all__ = ["ConditionalTransformer"]

from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from batcharray.recursive2.transformer.base import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from batcharray.recursive2.registry import TransformerRegistry


class ConditionalTransformer(BaseTransformer):
    """Wrapper transformer that only applies function if condition
    matches.

    This transformer allows to add filtering without changing the core
    design.

    Example usage:

    ```pycon

    >>> from batcharray.recursive2.transformer import DefaultTransformer, ConditionalTransformer
    >>> from batcharray.recursive2 import TransformerRegistry
    >>> registry = TransformerRegistry()
    >>> transformer = ConditionalTransformer(
    ...     transformer=DefaultTransformer(), condition=lambda x: isinstance(x, str)
    ... )
    >>> transformer
    ConditionalTransformer(
      (transformer): DefaultTransformer()
      (condition): <function <lambda> at 0x...>
    )
    >>> transformer.transform("abc", func=str.upper, registry=registry)
    'ABC'

    ```
    """

    def __init__(
        self,
        transformer: BaseTransformer,
        condition: Callable[[Any], bool],
    ) -> None:
        self._transformer = transformer
        self._condition = condition

    def __repr__(self) -> str:
        params = {"transformer": self._transformer, "condition": self._condition}
        return f"{self.__class__.__qualname__}(\n  {repr_indent(repr_mapping(params))}\n)"

    def __str__(self) -> str:
        params = {"transformer": self._transformer, "condition": self._condition}
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(params))}\n)"

    def transform(
        self,
        data: Any,
        func: Callable[[Any], Any],
        registry: TransformerRegistry,
    ) -> Any:
        if self._condition(data):
            return self._transformer.transform(data, func, registry)
        return data
