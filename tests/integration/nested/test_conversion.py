from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from coola.equality import objects_are_equal
from coola.recursive import BaseTransformer, TransformerRegistry, get_default_registry

from batcharray.nested import to_list

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


class LinkedListNode:
    """A simple linked list node for testing custom iterators.

    Args:
        value: The value stored in this node. Can be any data type.
        next_node: Optional reference to the next node in the list.
    """

    def __init__(self, value: Any, next_node: LinkedListNode | None = None) -> None:
        self.value = value
        self.next = next_node


class LinkedListTransformer(BaseTransformer[LinkedListNode]):
    """Custom transformer for linked list structures."""

    def transform(
        self,
        data: LinkedListNode,
        func: Callable[[Any], Any],
        registry: TransformerRegistry,
    ) -> Any:
        return type(data)(value=func(data.value), next_node=registry.transform(data.next, func))


#############################
#     Tests for to_list     #
#############################


def test_to_list_with_linked_list() -> None:
    get_default_registry().register(LinkedListNode, LinkedListTransformer())
    node = to_list(LinkedListNode(np.array([1, 2, 3])))
    assert objects_are_equal(node.value, [1, 2, 3])
