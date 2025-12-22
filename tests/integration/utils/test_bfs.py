from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from coola import objects_are_equal

from batcharray.utils.bfs import (
    BaseArrayIterator,
    IterableArrayIterator,
    IteratorRegistry,
    bfs_array,
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
    r"""A simple linked list node for testing custom iterators.

    This class represents a node in a singly-linked list structure,
    used to test that custom iterators can be registered and work
    correctly with the BFS traversal system.

    Args:
        value: The value stored in this node. Can be any type,
            including numpy arrays or nested structures.
        next_node: The next node in the linked list, or ``None``
            if this is the last node.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> node3 = LinkedListNode(np.array([3]))
    >>> node2 = LinkedListNode(np.array([2]), node3)
    >>> node1 = LinkedListNode(np.array([1]), node2)
    >>> node1.value
    array([1])
    >>> node1.next.value
    array([2])

    ```
    """

    def __init__(self, value: Any, next_node: LinkedListNode | None = None) -> None:
        self.value = value
        self.next = next_node


class LinkedListIterator(BaseArrayIterator):
    r"""Iterator for traversing linked list structures in BFS.

    This iterator handles LinkedListNode objects by exposing both
    the node's value and the next node as children. This allows
    BFS to process the linked list level by level, exploring
    values before moving to the next nodes.

    The iterator returns the current node's value first, followed
    by the next node (if it exists). This ensures that arrays
    stored in values are found before moving deeper into the list.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from batcharray.utils.bfs import IteratorRegistry, bfs_array
    >>> registry = IteratorRegistry()
    >>> registry.register(LinkedListNode, LinkedListIterator())
    >>> node2 = LinkedListNode(np.array([2]))
    >>> node1 = LinkedListNode(np.array([1]), node2)
    >>> list(bfs_array(node1, registry=registry))
    [array([1]), array([2])]

    ```
    """

    def get_children(self, data: LinkedListNode) -> list[Any]:
        result = [data.value]
        if data.next is not None:
            result.append(data.next)
        return result


def test_bfs_array_with_linked_list() -> None:
    # Test custom iterator for linked list
    registry = IteratorRegistry()
    registry.register(LinkedListNode, LinkedListIterator())
    registry.register(list, IterableArrayIterator())

    # Create linked list: [1] -> [2, 3] -> [4]
    node3 = LinkedListNode(np.array([4]))
    node2 = LinkedListNode([np.array([2]), np.array([3])], node3)
    node1 = LinkedListNode(np.array([1]), node2)

    assert objects_are_equal(
        list(bfs_array(node1, registry=registry)),
        [np.array([1]), np.array([2]), np.array([3]), np.array([4])],
    )


def test_bfs_array_performance_wide_shallow() -> None:
    # Performance test: wide structure (many items at same level)
    data = [np.array([i]) for i in range(1000)]
    result = list(bfs_array(data))
    assert len(result) == 1000


def test_bfs_array_performance_narrow_deep() -> None:
    # Performance test: deep nesting
    data = np.array([1])
    for _ in range(100):
        data = [data]
    result = list(bfs_array(data))
    assert len(result) == 1
    assert objects_are_equal(result[0], np.array([1]))
