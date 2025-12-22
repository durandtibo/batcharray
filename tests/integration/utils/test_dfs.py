from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from coola import objects_are_equal

from batcharray.utils.dfs2 import (
    BaseArrayIterator,
    DefaultArrayIterator,
    IterableArrayIterator,
    IteratorRegistry,
    dfs_array,
    get_default_registry,
)

if TYPE_CHECKING:
    from collections.abc import Generator


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


class LinkedListIterator(BaseArrayIterator):
    """Custom iterator for traversing linked list structures.

    This iterator demonstrates how to extend the system to handle custom
    data structures. It traverses the linked list by following the
    'next' pointers and recursively searches each node's value for numpy
    arrays.
    """

    def iterate(self, data: LinkedListNode, registry: IteratorRegistry) -> Generator[np.ndarray]:
        current = data
        while current is not None:
            yield from registry.iterate(current.value)
            current = current.next


##############################################
#     Integration tests for extensibility    #
##############################################


def test_custom_data_structure_with_custom_iterator() -> None:
    """Test that users can easily extend the system with custom data
    structures."""
    registry = IteratorRegistry(
        {LinkedListNode: LinkedListIterator(), list: IterableArrayIterator()}
    )
    data = LinkedListNode(
        np.array([1]), LinkedListNode([np.array([2])], LinkedListNode(np.array([3, 4])))
    )
    assert objects_are_equal(
        list(dfs_array(data, registry=registry)), [np.array([1]), np.array([2]), np.array([3, 4])]
    )


def test_multiple_custom_registries_isolated() -> None:
    """Test that multiple custom registries don't interfere with each
    other."""
    registry1 = IteratorRegistry({list: DefaultArrayIterator()})
    registry2 = IteratorRegistry({list: IterableArrayIterator()})
    data = [np.ones(2), np.zeros(3)]

    # Registry 1 should yield nothing (treats list as leaf)
    assert objects_are_equal(list(dfs_array(data, registry=registry1)), [])
    # Registry 2 should yield arrays
    assert objects_are_equal(list(dfs_array(data, registry=registry2)), [np.ones(2), np.zeros(3)])
