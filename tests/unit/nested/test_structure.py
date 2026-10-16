from __future__ import annotations

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray.nested.structure import concatenate_structures, split_structure

#################################
#     split_structure           #
#################################


def test_split_structure_array() -> None:
    out = split_structure(np.arange(4), lambda a: np.array_split(a, 2))
    assert objects_are_equal(out, [np.array([0, 1]), np.array([2, 3])])


def test_split_structure_nested() -> None:
    data = {"a": [np.arange(4)], "b": (np.arange(4) + 10,)}
    out = split_structure(data, lambda a: np.array_split(a, 2))
    assert objects_are_equal(
        out,
        [
            {"a": [np.array([0, 1])], "b": (np.array([10, 11]),)},
            {"a": [np.array([2, 3])], "b": (np.array([12, 13]),)},
        ],
    )


def test_split_structure_empty() -> None:
    assert split_structure({}, lambda a: [a]) == []


def test_split_structure_non_array_raises() -> None:
    with pytest.raises(TypeError, match="Expected an array"):
        split_structure({"a": 1}, lambda a: [a])


def test_split_structure_different_number_of_pieces_raises() -> None:
    with pytest.raises(ValueError, match="same number of pieces"):
        split_structure({"a": np.arange(2), "b": np.arange(4)}, lambda a: np.array_split(a, len(a)))


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
        ([{"a": np.array([1])}, [np.array([1])]], "all the items to be dictionaries"),
        ([[np.array([1])], (np.array([1]),)], "of type list"),
        ([(np.array([1]),), [np.array([1])]], "of type tuple"),
        ([1, 2], "Expected an array or a nested structure"),
    ],
)
def test_concatenate_structures_type_mismatch_raises(items: list, match: str) -> None:
    with pytest.raises(TypeError, match=match):
        concatenate_structures(items, _cat)


def test_concatenate_structures_different_keys_raises() -> None:
    with pytest.raises(ValueError, match="same keys"):
        concatenate_structures([{"a": np.array([1])}, {"b": np.array([1])}], _cat)


def test_concatenate_structures_different_lengths_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        concatenate_structures([[np.array([1])], [np.array([1]), np.array([2])]], _cat)
