from __future__ import annotations

from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from coola import objects_are_equal

from batcharray.utils.bfs import (
    BaseArrayIterator,
    DefaultArrayIterator,
    IterableArrayIterator,
    IteratorRegistry,
    MappingArrayIterator,
    bfs_array,
    get_default_registry,
    register_iterators,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Mapping


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


class CustomType:
    r"""Create a custom class."""


class CustomType1:
    r"""Create a custom class."""


class CustomType2:
    r"""Create a custom class."""


class CustomList(list):
    r"""Create a custom class that inherits from list."""


###############################
#     Tests for bfs_array     #
###############################


def test_bfs_array_array() -> None:
    assert objects_are_equal(list(bfs_array(np.ones((2, 3)))), [np.ones((2, 3))])


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
def test_bfs_array_no_array(data: Any) -> None:
    assert objects_are_equal(list(bfs_array(data)), [])


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
def test_bfs_array_iterable_array(data: Any) -> None:
    assert objects_are_equal(list(bfs_array(data)), [np.ones((2, 3)), np.array([0, 1, 2, 3, 4])])


def test_bfs_array_nested_data() -> None:
    # BFS processes level by level, so the order differs from DFS
    # Level 0: root list
    # Level 1: dict, np.ones((2,3)), list, tuple, np.array([0,1,2,3,4])
    # Level 2: items from dict, items from nested list, items from tuple
    # etc.
    data = [
        {"key1": np.zeros((1, 1, 1)), "key2": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])},
        np.ones((2, 3)),
        [np.ones(4), np.array([0, -1, -2]), [np.ones(5)]],
        (1, np.array([42.0]), np.zeros(2)),
        np.array([0, 1, 2, 3, 4]),
    ]
    # BFS order: arrays at same depth level come before deeper ones
    assert objects_are_equal(
        list(bfs_array(data)),
        [
            np.ones((2, 3)),  # Level 1
            np.array([0, 1, 2, 3, 4]),  # Level 1
            np.zeros((1, 1, 1)),  # Level 2 (from dict)
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),  # Level 2 (from dict)
            np.ones(4),  # Level 2 (from nested list)
            np.array([0, -1, -2]),  # Level 2 (from nested list)
            np.array([42.0]),  # Level 2 (from tuple)
            np.zeros(2),  # Level 2 (from tuple)
            np.ones(5),  # Level 3 (from doubly nested list)
        ],
    )


def test_bfs_array_with_custom_registry() -> None:
    registry = IteratorRegistry({list: IterableArrayIterator(), dict: MappingArrayIterator()})
    assert objects_are_equal(
        list(
            bfs_array(
                ["abc", np.ones((2, 3)), {"key": np.array([0, 1, 2, 3, 4])}], registry=registry
            )
        ),
        [np.ones((2, 3)), np.array([0, 1, 2, 3, 4])],
    )


def test_bfs_array_uses_default_registry_when_none() -> None:
    assert objects_are_equal(list(bfs_array([np.ones((2, 3))])), [np.ones((2, 3))])


def test_bfs_array_level_order_simple() -> None:
    data = [
        np.array([1]),  # Level 1
        [np.array([2]), np.array([3])],  # Level 2
    ]
    result = list(bfs_array(data))
    assert objects_are_equal(result, [np.array([1]), np.array([2]), np.array([3])])


def test_bfs_array_level_order_complex() -> None:
    data = {
        "level1_a": np.array([1]),
        "level1_b": [np.array([2]), {"level2": np.array([3])}],
    }
    result = list(bfs_array(data))
    # Level 1: np.array([1])
    # Level 2: np.array([2])
    # Level 3: np.array([3])
    assert objects_are_equal(result, [np.array([1]), np.array([2]), np.array([3])])


def test_bfs_array_deeply_nested() -> None:
    data = [[[[[np.array([1])]]]]]
    result = list(bfs_array(data))
    assert objects_are_equal(result, [np.array([1])])


def test_bfs_array_wide_structure() -> None:
    assert objects_are_equal(
        list(bfs_array([np.array([i]) for i in range(10)])), [np.array([i]) for i in range(10)]
    )


def test_bfs_array_mixed_depth_structure() -> None:
    data = [
        np.array([1]),  # Shallow
        [[[np.array([2])]]],  # Deep
        [np.array([3])],  # Medium
    ]
    result = list(bfs_array(data))
    # Level 1: array([1])
    # Level 2: array([3])
    # Level 4: array([2])
    assert objects_are_equal(result, [np.array([1]), np.array([3]), np.array([2])])


def test_bfs_array_multiple_arrays_same_container() -> None:
    assert objects_are_equal(
        list(bfs_array([np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])])),
        [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])],
    )


def test_bfs_array_deque_structure() -> None:
    assert objects_are_equal(
        list(bfs_array(deque([np.array([1]), deque([np.array([2]), np.array([3])])]))),
        [np.array([1]), np.array([2]), np.array([3])],
    )


def test_bfs_array_ordered_dict() -> None:
    data = OrderedDict(
        [
            ("first", np.array([1])),
            ("second", [np.array([2])]),
            ("third", np.array([3])),
        ]
    )
    # Level 1: arrays directly in dict values
    # Level 2: array nested in list
    assert objects_are_equal(list(bfs_array(data)), [np.array([1]), np.array([3]), np.array([2])])


def test_bfs_array_dict_with_nested_dict() -> None:
    data = {
        "outer1": {"inner1": np.array([1]), "inner2": np.array([2])},
        "outer2": np.array([3]),
    }
    assert objects_are_equal(list(bfs_array(data)), [np.array([3]), np.array([1]), np.array([2])])


def test_bfs_array_empty_nested_containers() -> None:
    assert objects_are_equal(list(bfs_array([[], {}, [[]], np.array([1])])), [np.array([1])])


def test_bfs_array_tuple_of_tuples() -> None:
    # Level 1: first two tuples
    # Level 2: arrays from first tuple
    # Level 3: array and nested tuple from second tuple
    # Level 4: array from deeply nested tuple
    assert objects_are_equal(
        list(bfs_array(((np.array([1]), np.array([2])), (np.array([3]), (np.array([4]),))))),
        [np.array([1]), np.array([2]), np.array([3]), np.array([4])],
    )


def test_bfs_array_multidimensional_arrays() -> None:
    assert objects_are_equal(
        list(bfs_array([np.array([1, 2, 3]), np.ones((2, 3)), np.zeros((2, 3, 4))])),
        [np.array([1, 2, 3]), np.ones((2, 3)), np.zeros((2, 3, 4))],
    )


def test_bfs_array_different_dtypes() -> None:
    assert objects_are_equal(
        list(
            bfs_array(
                [
                    np.array([1, 2, 3], dtype=np.int32),
                    np.array([1.0, 2.0, 3.0], dtype=np.float64),
                    np.array([True, False], dtype=bool),
                    np.array(["a", "b"], dtype=object),
                ]
            )
        ),
        [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([True, False], dtype=bool),
            np.array(["a", "b"], dtype=object),
        ],
    )


def test_bfs_array_mixed_container_types() -> None:
    data = {"list": [np.array([1])], "tuple": (np.array([2]),), "dict": {"nested": np.array([3])}}
    assert objects_are_equal(list(bfs_array(data)), [np.array([1]), np.array([2]), np.array([3])])


def test_iterator_registry_iterate_empty_queue() -> None:
    registry = IteratorRegistry({list: IterableArrayIterator()})
    assert list(registry.iterate([])) == []


def test_bfs_array_none_values() -> None:
    assert objects_are_equal(
        list(bfs_array([None, np.array([1]), None, [None, np.array([2])]])),
        [np.array([1]), np.array([2])],
    )


def test_bfs_array_string_not_iterated() -> None:
    assert objects_are_equal(
        list(bfs_array(["hello", np.array([1]), ["world", np.array([2])]])),
        [np.array([1]), np.array([2])],
    )


def test_bfs_array_scalar_array() -> None:
    assert objects_are_equal(
        list(bfs_array([np.array(42), np.array([1, 2, 3])])), [np.array(42), np.array([1, 2, 3])]
    )


def test_bfs_array_empty_arrays() -> None:
    assert objects_are_equal(
        list(bfs_array([np.array([]), np.array([1, 2, 3]), np.array([])])),
        [np.array([]), np.array([1, 2, 3]), np.array([])],
    )


##########################################
#     Tests for DefaultArrayIterator     #
##########################################


def test_default_array_iterator_repr() -> None:
    assert repr(DefaultArrayIterator()) == "DefaultArrayIterator()"


def test_default_array_iterator_str() -> None:
    assert str(DefaultArrayIterator()) == "DefaultArrayIterator()"


def test_default_array_iterator_get_children_array() -> None:
    assert DefaultArrayIterator().get_children(np.ones((2, 3))) == []


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([1, 2, 3], id="list"),
        pytest.param([], id="empty list"),
        pytest.param((), id="empty tuple"),
        pytest.param(set(), id="empty set"),
        pytest.param(deque(), id="empty deque"),
        pytest.param("abc", id="string"),
        pytest.param(42, id="int"),
    ],
)
def test_default_array_iterator_get_children_non_array(data: Any) -> None:
    assert DefaultArrayIterator().get_children(data) == []


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
def test_iterable_array_iterator_get_children_empty(data: Iterable) -> None:
    assert IterableArrayIterator().get_children(data) == []


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            ["abc", np.ones((2, 3)), 42, np.array([0, 1, 2, 3, 4])],
            ["abc", np.ones((2, 3)), 42, np.array([0, 1, 2, 3, 4])],
            id="list",
        ),
        pytest.param(
            deque(["abc", np.ones((2, 3)), 42, np.array([0, 1, 2, 3, 4])]),
            ["abc", np.ones((2, 3)), 42, np.array([0, 1, 2, 3, 4])],
            id="deque",
        ),
        pytest.param(
            ("abc", np.ones((2, 3)), 42, np.array([0, 1, 2, 3, 4])),
            ["abc", np.ones((2, 3)), 42, np.array([0, 1, 2, 3, 4])],
            id="tuple",
        ),
    ],
)
def test_iterable_array_iterator_get_children(data: Iterable, expected: list) -> None:
    assert objects_are_equal(IterableArrayIterator().get_children(data), expected)


def test_iterable_array_iterator_get_children_nested() -> None:
    assert objects_are_equal(
        IterableArrayIterator().get_children([[np.ones(2)], [np.zeros(3)]]),
        [[np.ones(2)], [np.zeros(3)]],
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
def test_mapping_array_iterator_get_children_empty(data: Mapping) -> None:
    assert MappingArrayIterator().get_children(data) == []


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            {
                "key1": "abc",
                "key2": np.ones((2, 3)),
                "key3": 42,
                "key4": np.array([0, 1, 2, 3, 4]),
            },
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
def test_mapping_array_iterator_get_children(data: Mapping) -> None:
    assert objects_are_equal(
        MappingArrayIterator().get_children(data),
        [
            "abc",
            np.ones((2, 3)),
            42,
            np.array([0, 1, 2, 3, 4]),
        ],
    )


def test_mapping_array_iterator_get_children_nested() -> None:
    assert objects_are_equal(
        MappingArrayIterator().get_children({"a": {"b": np.ones(2)}, "c": np.zeros(3)}),
        [{"b": np.ones(2)}, np.zeros(3)],
    )


######################################
#     Tests for IteratorRegistry     #
######################################


def test_iterator_registry_repr() -> None:
    assert repr(IteratorRegistry()).startswith("IteratorRegistry(")


def test_iterator_registry_str() -> None:
    assert str(IteratorRegistry()).startswith("IteratorRegistry(")


def test_iterator_registry_init_with_dict() -> None:
    registry = IteratorRegistry(
        {
            list: IterableArrayIterator(),
            dict: MappingArrayIterator(),
        }
    )
    assert len(registry._registry) == 2
    assert list in registry._registry
    assert dict in registry._registry


def test_iterator_registry_init_copies_dict() -> None:
    initial: dict[type, BaseArrayIterator] = {list: IterableArrayIterator()}
    registry = IteratorRegistry(initial)
    # Modify original dict
    initial[dict] = MappingArrayIterator()
    # Registry should not be affected
    assert dict not in registry._registry
    assert len(registry._registry) == 1


def test_iterator_registry_init_with_none() -> None:
    registry = IteratorRegistry(None)
    assert len(registry._registry) == 0


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
    registry = IteratorRegistry({list: DefaultArrayIterator()})
    iterators = {
        list: IterableArrayIterator(),
        dict: MappingArrayIterator(),
    }
    with pytest.raises(RuntimeError, match=r"An iterator (.*) is already registered"):
        registry.register_many(iterators, exist_ok=False)


def test_iterator_registry_register_many_exist_ok_true() -> None:
    registry = IteratorRegistry({list: DefaultArrayIterator()})
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
    registry = IteratorRegistry({list: IterableArrayIterator()})
    assert registry.has_iterator(list)


def test_iterator_registry_has_iterator_false() -> None:
    registry = IteratorRegistry()
    assert not registry.has_iterator(list)


def test_iterator_registry_find_iterator_direct() -> None:
    iterator = IterableArrayIterator()
    registry = IteratorRegistry({list: iterator})
    assert registry.find_iterator(list) is iterator


def test_iterator_registry_find_iterator_mro_lookup() -> None:
    iterator = IterableArrayIterator()
    registry = IteratorRegistry({list: iterator})
    assert registry.find_iterator(CustomList) is iterator
    # CustomList should NOT be in the registry after lookup
    assert CustomList not in registry._registry


def test_iterator_registry_find_iterator_default() -> None:
    registry = IteratorRegistry()
    assert isinstance(registry.find_iterator(int), DefaultArrayIterator)


def test_iterator_registry_iterate() -> None:
    registry = IteratorRegistry({list: IterableArrayIterator()})
    assert objects_are_equal(
        list(registry.iterate([np.ones((2, 3)), "abc", np.array([0, 1, 2, 3, 4])])),
        [np.ones((2, 3)), np.array([0, 1, 2, 3, 4])],
    )


def test_iterator_registry_iterate_uses_find_iterator() -> None:
    registry = IteratorRegistry({list: IterableArrayIterator()})
    assert objects_are_equal(list(registry.iterate([np.ones(3)])), [np.ones(3)])


def test_iterator_registry_iterate_bfs_order() -> None:
    registry = IteratorRegistry({list: IterableArrayIterator()})
    assert objects_are_equal(
        list(registry.iterate([np.ones(2), [np.zeros(3), [np.full(4, 5)]]])),
        [np.ones(2), np.zeros(3), np.full(4, 5)],
    )


def test_iterator_registry_iterate_with_unregistered_type() -> None:
    registry = IteratorRegistry()
    # Default iterator treats unknown types as leaf nodes, but list iterator should still work
    # However, without registering list, it won't iterate the list either
    # This tests that unknown types don't break the system
    assert list(registry.iterate([CustomType(), np.array([1]), CustomType()])) == []


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
    iterator = DefaultArrayIterator()
    register_iterators({CustomType: iterator})

    registry = get_default_registry()
    assert registry.has_iterator(CustomType)
    assert registry.find_iterator(CustomType) is iterator


def test_register_iterators_multiple() -> None:
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
    with pytest.raises(RuntimeError, match=r"An iterator (.*) is already registered"):
        register_iterators({list: DefaultArrayIterator()}, exist_ok=False)


def test_register_iterators_exist_ok_true() -> None:
    new_iterator = DefaultArrayIterator()
    register_iterators({list: new_iterator}, exist_ok=True)
