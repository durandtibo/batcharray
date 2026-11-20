from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray.nested.structure import concatenate_structures, split_structure

#################################
#     split_structure           #
#################################


def test_split_structure_array() -> None:
    out = split_structure(np.arange(4), lambda a: np.array_split(a, 2), axis=0)
    assert objects_are_equal(out, [np.array([0, 1]), np.array([2, 3])])


def test_split_structure_nested() -> None:
    data = {"a": [np.arange(4)], "b": (np.arange(4) + 10,)}
    out = split_structure(data, lambda a: np.array_split(a, 2), axis=0)
    assert objects_are_equal(
        out,
        [
            {"a": [np.array([0, 1])], "b": (np.array([10, 11]),)},
            {"a": [np.array([2, 3])], "b": (np.array([12, 13]),)},
        ],
    )


def test_split_structure_empty() -> None:
    assert split_structure({}, lambda a: [a], axis=0) == []


def test_split_structure_non_array_raises() -> None:
    with pytest.raises(TypeError, match="Expected an array"):
        split_structure({"a": 1}, lambda a: [a], axis=0)


def test_split_structure_different_sizes_batch_axis_raises() -> None:
    with pytest.raises(ValueError, match="same batch size"):
        split_structure(
            {"a": np.arange(5), "b": np.arange(3)}, lambda a: np.array_split(a, 2), axis=0
        )


def test_split_structure_different_sizes_seq_axis_raises() -> None:
    with pytest.raises(ValueError, match="same sequence size"):
        split_structure(
            {"a": np.ones((2, 5)), "b": np.ones((2, 3))},
            lambda a: np.array_split(a, 2, axis=1),
            axis=1,
        )


def test_split_structure_seq_axis() -> None:
    out = split_structure(np.ones((2, 4)), lambda a: np.array_split(a, 2, axis=1), axis=1)
    assert objects_are_equal(out, [np.ones((2, 2)), np.ones((2, 2))])


#######################################
#     concatenate_structures          #
#######################################


def _cat(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(arrays)


def test_concatenate_structures_array() -> None:
    assert objects_are_equal(
        concatenate_structures([np.array([1]), np.array([2])], _cat), np.array([1, 2])
    )


@pytest.mark.parametrize("container", [list, tuple])
def test_concatenate_structures_sequence_keeps_type(container: type) -> None:
    out = concatenate_structures([container([np.array([1])]), container([np.array([2])])], _cat)
    assert objects_are_equal(out, container([np.array([1, 2])]))


def test_concatenate_structures_nested() -> None:
    items = [
        {"a": [np.array([1])], "b": {"c": np.array([3])}},
        {"a": [np.array([2])], "b": {"c": np.array([4])}},
    ]
    assert objects_are_equal(
        concatenate_structures(items, _cat),
        {"a": [np.array([1, 2])], "b": {"c": np.array([3, 4])}},
    )


@pytest.mark.parametrize(
    ("items", "match"),
    [
        ([np.array([1]), 2], "all the items to be arrays"),
        ([{"a": np.array([1])}, {"a": 2}], "Expected all the items to be arrays"),
        ([{"a": np.array([1])}, np.array([1])], "all the items to be dictionaries"),
        ([[np.array([1])], np.array([1])], "all the items to be lists or tuples"),
        ([1, 2], "Expected an array or a nested structure"),
    ],
)
def test_concatenate_structures_type_mismatch_raises(items: list, match: str) -> None:
    with pytest.raises(TypeError, match=match):
        concatenate_structures(items, _cat)


def test_concatenate_structures_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty sequence"):
        concatenate_structures([], _cat)


@pytest.mark.parametrize(
    "items",
    [
        [{"a": np.array([1])}, [np.array([1])]],
        [{"a": np.array([1])}, (np.array([1]),)],
        [[np.array([1])], {"a": np.array([1])}],
        [(np.array([1]),), {"a": np.array([1])}],
    ],
)
def test_concatenate_structures_mapping_and_sequence_raises(items: list) -> None:
    with pytest.raises(ValueError, match="dictionary with a sequence"):
        concatenate_structures(items, _cat)


def test_concatenate_structures_list_and_tuple_gives_type_of_first_item() -> None:
    out = concatenate_structures([[np.array([1])], (np.array([2]),)], _cat)
    assert objects_are_equal(out, [np.array([1, 2])])


def test_concatenate_structures_tuple_and_list_gives_type_of_first_item() -> None:
    out = concatenate_structures([(np.array([1]),), [np.array([2])]], _cat)
    assert objects_are_equal(out, (np.array([1, 2]),))


def test_concatenate_structures_dict_and_ordered_dict_gives_type_of_first_item() -> None:
    out = concatenate_structures([{"a": np.array([1])}, OrderedDict(a=np.array([2]))], _cat)
    assert objects_are_equal(out, {"a": np.array([1, 2])})


def test_concatenate_structures_ordered_dict_and_dict_gives_type_of_first_item() -> None:
    out = concatenate_structures([OrderedDict(a=np.array([1])), {"a": np.array([2])}], _cat)
    assert objects_are_equal(out, OrderedDict(a=np.array([1, 2])))


def test_concatenate_structures_different_keys_raises() -> None:
    with pytest.raises(ValueError, match="same keys"):
        concatenate_structures([{"a": np.array([1])}, {"b": np.array([1])}], _cat)


def test_concatenate_structures_different_lengths_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        concatenate_structures([[np.array([1])], [np.array([1]), np.array([2])]], _cat)
