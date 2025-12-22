from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Generator, Iterable, Mapping
from typing import Any

import numpy as np
import pytest
from coola import objects_are_equal

from batcharray.utils.dfs2 import (
    BaseArrayIterator,
    DefaultArrayIterator,
    IterableArrayIterator,
    IteratorRegistry,
    MappingArrayIterator,
    dfs_array,
    get_default_registry,
    register_iterators,
)

###############################
#     Tests for dfs_array     #
###############################


def test_dfs_array_array() -> None:
    assert objects_are_equal(list(dfs_array(np.ones((2, 3)))), [np.ones((2, 3))])


@pytest.mark.parametrize(
    "data",
    [
        pytest.param("abc", id="string"),
        pytest.param(42, id="int"),
        pytest.param(4.2, id="float"),
        pytest.param([1, 2, 3], id="list"),
        pytest.param([], id="empty list"),
        pytest.param(("a", "b", "c"), id="tuple"),
        pytest.param((), id="empty tuple"),
        pytest.param({1, 2, 3}, id="set"),
        pytest.param(set(), id="empty set"),
        pytest.param({"key1": 1, "key2": 2, "key3": 3}, id="dict"),
        pytest.param({}, id="empty dict"),
    ],
)
def test_dfs_array_no_array(data: Any) -> None:
    assert objects_are_equal(list(dfs_array(data)), [])


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([np.ones((2, 3)), np.array([0, 1, 2, 3, 4])], id="list with only arrays"),
        pytest.param(
            ["abc", np.ones((2, 3)), 42, np.array([0, 1, 2, 3, 4])],
            id="list with non array objects",
        ),
        pytest.param((np.ones((2, 3)), np.array([0, 1, 2, 3, 4])), id="tuple with only arrays"),
        pytest.param(
            ("abc", np.ones((2, 3)), 42, np.array([0, 1, 2, 3, 4])),
            id="tuple with non array objects",
        ),
        pytest.param(
            {"key1": np.ones((2, 3)), "key2": np.array([0, 1, 2, 3, 4])}, id="dict with only arrays"
        ),
        pytest.param(
            {"key1": "abc", "key2": np.ones((2, 3)), "key3": 42, "key4": np.array([0, 1, 2, 3, 4])},
            id="dict with non array objects",
        ),
    ],
)
def test_dfs_array_iterable_array(data: Any) -> None:
    assert objects_are_equal(list(dfs_array(data)), [np.ones((2, 3)), np.array([0, 1, 2, 3, 4])])


def test_dfs_array_nested_data() -> None:
    data = [
        {"key1": np.zeros((1, 1, 1)), "key2": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
        np.ones((2, 3)),
        [np.ones(4), np.array([0, -1, -2]), [np.ones(5)]],
        (1, np.array([42.0]), np.zeros(2)),
        np.array([0, 1, 2, 3, 4]),
    ]
    assert objects_are_equal(
        list(dfs_array(data)),
        [
            np.zeros((1, 1, 1)),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            np.ones((2, 3)),
            np.ones(4),
            np.array([0, -1, -2]),
            np.ones(5),
            np.array([42.0]),
            np.zeros(2),
            np.array([0, 1, 2, 3, 4]),
        ],
    )


def test_dfs_array_with_custom_registry() -> None:
    registry = IteratorRegistry()
    registry.register(list, IterableArrayIterator())
    registry.register(dict, MappingArrayIterator())
    assert objects_are_equal(
        list(
            dfs_array(
                ["abc", np.ones((2, 3)), {"key": np.array([0, 1, 2, 3, 4])}], registry=registry
            )
        ),
        [np.ones((2, 3)), np.array([0, 1, 2, 3, 4])],
    )


def test_dfs_array_uses_default_registry_when_none() -> None:
    assert objects_are_equal(list(dfs_array([np.ones((2, 3))])), [np.ones((2, 3))])


##########################################
#     Tests for DefaultArrayIterator     #
##########################################


def test_default_array_iterator_repr() -> None:
    assert repr(DefaultArrayIterator()) == "DefaultArrayIterator()"


def test_default_array_iterator_str() -> None:
    assert str(DefaultArrayIterator()) == "DefaultArrayIterator()"


def test_default_array_iterator_iterate_array() -> None:
    registry = IteratorRegistry()
    iterator = DefaultArrayIterator()
    assert objects_are_equal(list(iterator.iterate(np.ones((2, 3)), registry)), [np.ones((2, 3))])


def test_default_array_iterator_iterate_non_array() -> None:
    registry = IteratorRegistry()
    iterator = DefaultArrayIterator()
    assert list(iterator.iterate("abc", registry)) == []
    assert list(iterator.iterate(42, registry)) == []
    assert list(iterator.iterate([1, 2, 3], registry)) == []


###########################################
#     Tests for IterableArrayIterator     #
###########################################


def test_iterable_array_iterator_repr() -> None:
    assert repr(IterableArrayIterator()) == "IterableArrayIterator()"


def test_iterable_array_iterator_str() -> None:
    assert str(IterableArrayIterator()) == "IterableArrayIterator()"


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([], id="empty list"),
        pytest.param((), id="empty tuple"),
        pytest.param(set(), id="empty set"),
        pytest.param(deque(), id="empty deque"),
    ],
)
def test_iterable_array_iterator_iterate_empty(data: Iterable) -> None:
    registry = get_default_registry()
    assert list(IterableArrayIterator().iterate(data, registry)) == []


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(["abc", np.ones((2, 3)), 42, np.array([0, 1, 2, 3, 4])], id="list"),
        pytest.param(deque(["abc", np.ones((2, 3)), 42, np.array([0, 1, 2, 3, 4])]), id="deque"),
        pytest.param(("abc", np.ones((2, 3)), 42, np.array([0, 1, 2, 3, 4])), id="tuple"),
    ],
)
def test_iterable_array_iterator_iterate(data: Iterable) -> None:
    registry = get_default_registry()
    assert objects_are_equal(
        list(IterableArrayIterator().iterate(data, registry)),
        [np.ones((2, 3)), np.array([0, 1, 2, 3, 4])],
    )


def test_iterable_array_iterator_iterate_nested() -> None:
    registry = get_default_registry()
    assert objects_are_equal(
        list(IterableArrayIterator().iterate([[np.ones(2)], [np.zeros(3)]], registry)),
        [np.ones(2), np.zeros(3)],
    )


##########################################
#     Tests for MappingArrayIterator     #
##########################################


def test_mapping_array_iterator_repr() -> None:
    assert repr(MappingArrayIterator()) == "MappingArrayIterator()"


def test_mapping_array_iterator_str() -> None:
    assert str(MappingArrayIterator()) == "MappingArrayIterator()"


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({}, id="empty dict"),
        pytest.param(OrderedDict(), id="empty OrderedDict"),
    ],
)
def test_mapping_array_iterator_iterate_empty(data: Mapping) -> None:
    registry = get_default_registry()
    assert list(MappingArrayIterator().iterate(data, registry)) == []


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            {"key1": "abc", "key2": np.ones((2, 3)), "key3": 42, "key4": np.array([0, 1, 2, 3, 4])},
            id="dict",
        ),
        pytest.param(
            OrderedDict(
                {
                    "key1": "abc",
                    "key2": np.ones((2, 3)),
                    "key3": 42,
                    "key4": np.array([0, 1, 2, 3, 4]),
                }
            ),
            id="OrderedDict",
        ),
    ],
)
def test_mapping_array_iterator_iterate(data: Mapping) -> None:
    registry = get_default_registry()
    assert objects_are_equal(
        list(MappingArrayIterator().iterate(data, registry)),
        [np.ones((2, 3)), np.array([0, 1, 2, 3, 4])],
    )


def test_mapping_array_iterator_iterate_nested() -> None:
    registry = get_default_registry()
    assert objects_are_equal(
        list(MappingArrayIterator().iterate({"a": {"b": np.ones(2)}, "c": np.zeros(3)}, registry)),
        [np.ones(2), np.zeros(3)],
    )


######################################
#     Tests for IteratorRegistry     #
######################################


def test_iterator_registry_repr() -> None:
    assert repr(IteratorRegistry()).startswith("IteratorRegistry(")


def test_iterator_registry_str() -> None:
    assert str(IteratorRegistry()).startswith("IteratorRegistry(")


def test_iterator_registry_register() -> None:
    registry = IteratorRegistry()
    iterator = IterableArrayIterator()
    registry.register(list, iterator)
    assert registry._registry[list] is iterator


def test_iterator_registry_register_duplicate_exist_ok_true() -> None:
    registry = IteratorRegistry()
    iterator1 = DefaultArrayIterator()
    iterator2 = IterableArrayIterator()
    registry.register(list, iterator1)
    registry.register(list, iterator2, exist_ok=True)
    assert registry._registry[list] is iterator2


def test_iterator_registry_register_duplicate_exist_ok_false() -> None:
    registry = IteratorRegistry()
    registry.register(list, DefaultArrayIterator())
    with pytest.raises(RuntimeError, match=r"An iterator (.*) is already registered"):
        registry.register(list, IterableArrayIterator())


def test_iterator_registry_register_many() -> None:
    registry = IteratorRegistry()
    registry.register_many(
        {
            list: IterableArrayIterator(),
            dict: MappingArrayIterator(),
        }
    )
    assert isinstance(registry._registry[list], IterableArrayIterator)
    assert isinstance(registry._registry[dict], MappingArrayIterator)


def test_iterator_registry_register_many_exist_ok_false() -> None:
    registry = IteratorRegistry()
    registry.register(list, DefaultArrayIterator())
    iterators = {
        list: IterableArrayIterator(),
        dict: MappingArrayIterator(),
    }
    with pytest.raises(RuntimeError, match=r"An iterator (.*) is already registered"):
        registry.register_many(iterators, exist_ok=False)


def test_iterator_registry_register_many_exist_ok_true() -> None:
    registry = IteratorRegistry()
    registry.register(list, DefaultArrayIterator())
    registry.register_many(
        {
            list: IterableArrayIterator(),
            dict: MappingArrayIterator(),
        },
        exist_ok=True,
    )
    assert isinstance(registry._registry[list], IterableArrayIterator)
    assert isinstance(registry._registry[dict], MappingArrayIterator)


def test_iterator_registry_has_iterator_true() -> None:
    registry = IteratorRegistry()
    registry.register(list, IterableArrayIterator())
    assert registry.has_iterator(list)


def test_iterator_registry_has_iterator_false() -> None:
    registry = IteratorRegistry()
    assert not registry.has_iterator(list)


def test_iterator_registry_find_iterator_direct() -> None:
    registry = IteratorRegistry()
    iterator = IterableArrayIterator()
    registry.register(list, iterator)
    assert registry.find_iterator(list) is iterator


def test_iterator_registry_find_iterator_mro_lookup() -> None:
    registry = IteratorRegistry()
    iterator = IterableArrayIterator()
    registry.register(Iterable, iterator)
    assert registry.find_iterator(list) is iterator


def test_iterator_registry_find_iterator_default() -> None:
    registry = IteratorRegistry()
    iterator = registry.find_iterator(int)
    assert isinstance(iterator, DefaultArrayIterator)


def test_iterator_registry_iterate() -> None:
    registry = IteratorRegistry()
    registry.register(list, IterableArrayIterator())
    result = list(registry.iterate([np.ones((2, 3)), "abc", np.array([0, 1, 2, 3, 4])]))
    assert objects_are_equal(result, [np.ones((2, 3)), np.array([0, 1, 2, 3, 4])])


def test_iterator_registry_iterate_uses_find_iterator() -> None:
    registry = IteratorRegistry()
    registry.register(list, IterableArrayIterator())
    assert objects_are_equal(list(registry.iterate([np.ones(3)])), [np.ones(3)])


##########################################
#     Tests for get_default_registry     #
##########################################


def test_get_default_registry_returns_same_instance() -> None:
    registry1 = get_default_registry()
    registry2 = get_default_registry()
    assert registry1 is registry2


def test_get_default_registry_has_default_iterators() -> None:
    registry = get_default_registry()
    assert registry.has_iterator(list)
    assert registry.has_iterator(dict)
    assert registry.has_iterator(tuple)
    assert registry.has_iterator(set)
    assert registry.has_iterator(deque)
    assert registry.has_iterator(str)


def test_get_default_registry_iterators_are_correct_type() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_iterator(list), IterableArrayIterator)
    assert isinstance(registry.find_iterator(dict), MappingArrayIterator)
    assert isinstance(registry.find_iterator(tuple), IterableArrayIterator)
    assert isinstance(registry.find_iterator(set), IterableArrayIterator)
    assert isinstance(registry.find_iterator(deque), IterableArrayIterator)
    assert isinstance(registry.find_iterator(str), DefaultArrayIterator)
    assert isinstance(registry.find_iterator(object), DefaultArrayIterator)


########################################
#     Tests for register_iterators     #
########################################


def test_register_iterators() -> None:
    # Note: This modifies global state, but get_default_registry() is idempotent
    class CustomType:
        pass

    custom_iterator = DefaultArrayIterator()
    register_iterators({CustomType: custom_iterator}, exist_ok=True)

    registry = get_default_registry()
    assert registry.has_iterator(CustomType)
    assert registry.find_iterator(CustomType) is custom_iterator


def test_register_iterators_multiple() -> None:
    class CustomType1:
        pass

    class CustomType2:
        pass

    iterator1 = DefaultArrayIterator()
    iterator2 = IterableArrayIterator()

    register_iterators(
        {
            CustomType1: iterator1,
            CustomType2: iterator2,
        },
        exist_ok=True,
    )

    registry = get_default_registry()
    assert registry.has_iterator(CustomType1)
    assert registry.has_iterator(CustomType2)


def test_register_iterators_exist_ok_false() -> None:
    # Trying to register list again should raise error
    with pytest.raises(RuntimeError, match=r"An iterator (.*) is already registered"):
        register_iterators({list: DefaultArrayIterator()}, exist_ok=False)


def test_register_iterators_exist_ok_true() -> None:
    # Should not raise error with exist_ok=True
    new_iterator = DefaultArrayIterator()
    register_iterators({list: new_iterator}, exist_ok=True)
    # Note: This modifies the global registry


#############################################
#     Integration tests for extensibility #
#############################################


class LinkedListNode:
    def __init__(self, value: Any, next_node: LinkedListNode | None = None) -> None:
        self.value = value
        self.next = next_node


class LinkedListIterator(BaseArrayIterator):
    def iterate(self, data: LinkedListNode, registry: IteratorRegistry) -> Generator[np.ndarray]:
        current = data
        while current is not None:
            yield from registry.iterate(current.value)
            current = current.next


def test_custom_data_structure_with_custom_iterator() -> None:
    """Test that users can easily extend the system with custom data
    structures."""
    # Create custom registry
    registry = IteratorRegistry()
    registry.register(LinkedListNode, LinkedListIterator())
    registry.register(list, IterableArrayIterator())

    # Create linked list with arrays
    node3 = LinkedListNode(np.array([3, 4]))
    node2 = LinkedListNode([np.array([2])], node3)
    node1 = LinkedListNode(np.array([1]), node2)

    result = list(dfs_array(node1, registry=registry))
    assert len(result) == 3
    assert objects_are_equal(result[0], np.array([1]))
    assert objects_are_equal(result[1], np.array([2]))
    assert objects_are_equal(result[2], np.array([3, 4]))


def test_multiple_custom_registries_isolated() -> None:
    """Test that multiple custom registries don't interfere with each
    other."""
    registry1 = IteratorRegistry()
    registry1.register(list, DefaultArrayIterator())  # Treat lists as leaf nodes

    registry2 = IteratorRegistry()
    registry2.register(list, IterableArrayIterator())  # Iterate into lists

    data = [np.ones(2), np.zeros(3)]

    # Registry 1 should yield nothing (treats list as leaf)
    result1 = list(dfs_array(data, registry=registry1))
    assert len(result1) == 0

    # Registry 2 should yield arrays
    result2 = list(dfs_array(data, registry=registry2))
    assert len(result2) == 2
