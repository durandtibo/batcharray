from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import NamedTuple

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray.constants import BATCH_AXIS, SEQ_AXIS
from batcharray.utils import apply_to_arrays, combine_arrays, get_size, split_arrays
from batcharray.utils.structure import _validate_array


class Point(NamedTuple):
    x: np.ndarray
    y: np.ndarray


#####################################
#     Tests for apply_to_arrays     #
#####################################


def test_apply_to_arrays_array() -> None:
    assert objects_are_equal(apply_to_arrays(np.array([1, 2]), np.negative), np.array([-1, -2]))


def test_apply_to_arrays_nested() -> None:
    data = {"a": [np.array([1])], "b": (np.array([2]),), "c": {"d": np.array([3])}}
    assert objects_are_equal(
        apply_to_arrays(data, np.negative),
        {"a": [np.array([-1])], "b": (np.array([-2]),), "c": {"d": np.array([-3])}},
    )


@pytest.mark.parametrize(
    "data",
    [
        {"a": np.array([1])},
        OrderedDict(a=np.array([1])),
        defaultdict(list, a=np.array([1])),
        [np.array([1])],
        (np.array([1]),),
        Point(np.array([1]), np.array([1])),
    ],
)
def test_apply_to_arrays_keeps_container_type(data: object) -> None:
    assert type(apply_to_arrays(data, np.negative)) is type(data)


def test_apply_to_arrays_empty() -> None:
    assert apply_to_arrays({}, np.negative) == {}


def test_apply_to_arrays_fn_can_return_anything() -> None:
    assert apply_to_arrays({"a": np.array([1, 2])}, np.ndarray.tolist) == {"a": [1, 2]}


def test_apply_to_arrays_does_not_modify_input() -> None:
    data = {"a": np.array([1, 2])}
    apply_to_arrays(data, np.negative)
    assert objects_are_equal(data, {"a": np.array([1, 2])})


@pytest.mark.parametrize(("leaf", "name"), [(1, "int"), ("x", "str"), (None, "NoneType")])
def test_apply_to_arrays_non_array_leaf_raises(leaf: object, name: str) -> None:
    with pytest.raises(TypeError, match=f"object of type {name}"):
        apply_to_arrays({"a": np.ones(2), "b": leaf}, np.negative)


def test_apply_to_arrays_numpy_scalar_leaf_raises() -> None:
    with pytest.raises(TypeError, match="object of type float64"):
        apply_to_arrays({"a": np.float64(1.0)}, np.negative)


def test_apply_to_arrays_list_of_int_raises() -> None:
    with pytest.raises(TypeError, match="object of type int"):
        apply_to_arrays([1, 2], np.negative)


def test_apply_to_arrays_cycle_raises() -> None:
    data: list = [np.ones(2)]
    data.append(data)
    with pytest.raises(ValueError, match="Cycle detected"):
        apply_to_arrays(data, np.negative)


##################################
#     Tests for split_arrays     #
##################################


def _halves(array: np.ndarray) -> list[np.ndarray]:
    return np.array_split(array, 2)


def test_split_arrays_array() -> None:
    assert objects_are_equal(
        split_arrays(np.arange(4), _halves, axis=BATCH_AXIS), [np.array([0, 1]), np.array([2, 3])]
    )


def test_split_arrays_nested() -> None:
    data = {"a": [np.arange(4)], "b": (np.arange(4) + 10,)}
    assert objects_are_equal(
        split_arrays(data, _halves, axis=BATCH_AXIS),
        [
            {"a": [np.array([0, 1])], "b": (np.array([10, 11]),)},
            {"a": [np.array([2, 3])], "b": (np.array([12, 13]),)},
        ],
    )


def test_split_arrays_seq_axis() -> None:
    out = split_arrays(
        np.ones((2, 4)), lambda array: np.array_split(array, 2, axis=SEQ_AXIS), axis=SEQ_AXIS
    )
    assert objects_are_equal(out, [np.ones((2, 2)), np.ones((2, 2))])


def test_split_arrays_keeps_container_type() -> None:
    out = split_arrays(Point(np.arange(4), np.arange(4)), _halves, axis=BATCH_AXIS)
    assert type(out[0]) is Point


def test_split_arrays_empty_structure_raises() -> None:
    with pytest.raises(ValueError, match="at least one array"):
        split_arrays({}, _halves, axis=BATCH_AXIS)


def test_split_arrays_non_array_raises() -> None:
    with pytest.raises(TypeError, match="object of type int"):
        split_arrays({"a": 1}, _halves, axis=BATCH_AXIS)


@pytest.mark.parametrize("size", [3, 4, 6])
def test_split_arrays_different_batch_sizes_raises(size: int) -> None:
    with pytest.raises(ValueError, match="same batch size"):
        split_arrays({"a": np.arange(5), "b": np.arange(size)}, _halves, axis=BATCH_AXIS)


def test_split_arrays_different_seq_sizes_raises() -> None:
    with pytest.raises(ValueError, match="same sequence size"):
        split_arrays({"a": np.ones((2, 5)), "b": np.ones((2, 3))}, _halves, axis=SEQ_AXIS)


####################################
#     Tests for combine_arrays     #
####################################


def test_combine_arrays_array() -> None:
    assert objects_are_equal(
        combine_arrays([np.array([1]), np.array([2])], np.concatenate), np.array([1, 2])
    )


@pytest.mark.parametrize("container", [list, tuple])
def test_combine_arrays_sequence_keeps_type(container: type) -> None:
    out = combine_arrays([container([np.array([1])]), container([np.array([2])])], np.concatenate)
    assert objects_are_equal(out, container([np.array([1, 2])]))


def test_combine_arrays_nested() -> None:
    items = [
        {"a": [np.array([1])], "b": {"c": np.array([3])}},
        {"a": [np.array([2])], "b": {"c": np.array([4])}},
    ]
    assert objects_are_equal(
        combine_arrays(items, np.concatenate),
        {"a": [np.array([1, 2])], "b": {"c": np.array([3, 4])}},
    )


def test_combine_arrays_list_and_tuple_gives_type_of_first_item() -> None:
    out = combine_arrays([[np.array([1])], (np.array([2]),)], np.concatenate)
    assert objects_are_equal(out, [np.array([1, 2])])


def test_combine_arrays_tuple_and_list_gives_type_of_first_item() -> None:
    out = combine_arrays([(np.array([1]),), [np.array([2])]], np.concatenate)
    assert objects_are_equal(out, (np.array([1, 2]),))


def test_combine_arrays_dict_and_ordered_dict_gives_type_of_first_item() -> None:
    out = combine_arrays([{"a": np.array([1])}, OrderedDict(a=np.array([2]))], np.concatenate)
    assert objects_are_equal(out, {"a": np.array([1, 2])})


def test_combine_arrays_ordered_dict_and_dict_gives_type_of_first_item() -> None:
    out = combine_arrays([OrderedDict(a=np.array([1])), {"a": np.array([2])}], np.concatenate)
    assert objects_are_equal(out, OrderedDict(a=np.array([1, 2])))


def test_combine_arrays_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one structure"):
        combine_arrays([], np.concatenate)


@pytest.mark.parametrize(
    "items",
    [
        [{"a": np.array([1])}, [np.array([1])]],
        [{"a": np.array([1])}, (np.array([1]),)],
        [[np.array([1])], {"a": np.array([1])}],
        [(np.array([1]),), {"a": np.array([1])}],
    ],
)
def test_combine_arrays_mapping_and_sequence_raises(items: list) -> None:
    with pytest.raises(ValueError, match="same kind"):
        combine_arrays(items, np.concatenate)


def test_combine_arrays_different_keys_raises() -> None:
    with pytest.raises(ValueError, match="same keys"):
        combine_arrays([{"a": np.array([1])}, {"b": np.array([1])}], np.concatenate)


def test_combine_arrays_different_lengths_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        combine_arrays([[np.array([1])], [np.array([1]), np.array([2])]], np.concatenate)


def test_combine_arrays_array_and_container_raises() -> None:
    with pytest.raises(ValueError, match="is a leaf"):
        combine_arrays([np.array([1]), {"a": np.array([1])}], np.concatenate)


@pytest.mark.parametrize("leaf", [1, "x", None])
def test_combine_arrays_non_array_leaf_raises(leaf: object) -> None:
    with pytest.raises(TypeError, match="Expected an array"):
        combine_arrays([{"a": np.array([1])}, {"a": leaf}], np.concatenate)


def test_combine_arrays_non_array_first_leaf_raises() -> None:
    with pytest.raises(TypeError, match="object of type int"):
        combine_arrays([{"a": 1}, {"a": np.array([1])}], np.concatenate)


##############################
#     Tests for get_size     #
##############################


def test_get_size_array() -> None:
    assert get_size(np.ones((2, 3)), axis=BATCH_AXIS) == 2


def test_get_size_seq_axis() -> None:
    assert get_size(np.ones((2, 3)), axis=SEQ_AXIS) == 3


def test_get_size_nested() -> None:
    data = {"a": np.ones((2, 3)), "b": np.zeros(2), "c": [np.ones((2, 5, 4))], "d": (np.ones(2),)}
    assert get_size(data, axis=BATCH_AXIS) == 2


def test_get_size_zero() -> None:
    assert get_size({"a": np.ones((0, 3))}, axis=BATCH_AXIS) == 0


def test_get_size_different_batch_sizes_raises() -> None:
    with pytest.raises(ValueError, match=r"same batch size, but got \[2, 4\]"):
        get_size({"a": np.ones((2, 3)), "b": np.zeros((4, 3))}, axis=BATCH_AXIS)


def test_get_size_different_seq_sizes_raises() -> None:
    with pytest.raises(ValueError, match=r"same sequence size, but got \[3, 4\]"):
        get_size({"a": np.ones((2, 3)), "b": np.zeros((2, 4))}, axis=SEQ_AXIS)


def test_get_size_other_axis_is_named_in_error() -> None:
    with pytest.raises(ValueError, match=r"same axis 2 size"):
        get_size({"a": np.ones((2, 3, 4)), "b": np.zeros((2, 3, 5))}, axis=2)


def test_get_size_0d_raises() -> None:
    with pytest.raises(ValueError, match=r"batch size of item 0: it has 0 dimension"):
        get_size(np.array(1.0), axis=BATCH_AXIS)


def test_get_size_1d_seq_raises() -> None:
    with pytest.raises(ValueError, match=r"sequence size of item 1: it has 1 dimension"):
        get_size({"a": np.ones((2, 3)), "b": np.zeros(2)}, axis=SEQ_AXIS)


@pytest.mark.parametrize("data", [{}, [], ()])
def test_get_size_no_array_raises(data: object) -> None:
    with pytest.raises(ValueError, match="at least one array"):
        get_size(data, axis=BATCH_AXIS)


@pytest.mark.parametrize("leaf", [1, "x", None])
def test_get_size_non_array_leaf_raises(leaf: object) -> None:
    with pytest.raises(TypeError, match="Expected an array"):
        get_size({"a": np.ones(2), "b": leaf}, axis=BATCH_AXIS)


#####################################
#     Tests for _validate_array     #
#####################################


def test_validate_array_array() -> None:
    _validate_array(np.ones(2))


def test_validate_array_masked_array() -> None:
    _validate_array(np.ma.ones(2))


@pytest.mark.parametrize(("leaf", "name"), [(1, "int"), ([1], "list"), (np.float32(1), "float32")])
def test_validate_array_non_array_raises(leaf: object, name: str) -> None:
    with pytest.raises(TypeError, match=f"Expected an array, but got an object of type {name}"):
        _validate_array(leaf)
