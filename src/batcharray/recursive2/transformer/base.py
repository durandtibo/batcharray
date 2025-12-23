r"""Define the transformer base class."""

from __future__ import annotations

__all__ = ["BaseTransformer"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from batcharray.recursive2.registry import TransformerRegistry

T = TypeVar("T")


class BaseTransformer(ABC, Generic[T]):
    """Base class for type-specific transformers.

    Each transformer knows how to rebuild its specific type after
    transformation of nested elements.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    @abstractmethod
    def transform(
        self,
        data: T,
        func: Callable[[T], T],
        registry: TransformerRegistry,
    ) -> Any:
        """Transform data by applying func recursively.

        Args:
            data: The data to transform
            func: Function to apply to leaf values
            registry: Registry to resolve transformers for nested data

        Returns:
            Transformed data
        """
