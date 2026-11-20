r"""Tests of P17: the result on an empty or degenerate input is
documented. One test per row of the edge-case table of
``docs/docs/design_principles.md``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray import array as ba, nested as bn

if TYPE_CHECKING:
    from collections.abc import Callable

#####################################################
#     Tests for a zero batch size (reductions)      #
#####################################################


def test_sum_along_batch_zero_batch_size_is_the_identity() -> None:
    assert objects_are_equal(ba.sum_along_batch(np.ones((0, 3))), np.zeros(3))


def test_prod_along_batch_zero_batch_size_is_the_identity() -> None:
    assert objects_are_equal(ba.prod_along_batch(np.ones((0, 3))), np.ones(3))


def test_mean_along_batch_zero_batch_size_is_nan() -> None:
    with pytest.warns(RuntimeWarning, match=r"Mean of empty slice|invalid value encountered"):
        out = ba.mean_along_batch(np.ones((0, 3)))
    assert objects_are_equal(out, np.full(3, np.nan), equal_nan=True)


@pytest.mark.parametrize("name", ["max_along_batch", "min_along_batch"])
def test_max_min_along_batch_zero_batch_size_raises(name: str) -> None:
    with pytest.raises(ValueError, match="zero-size array to reduction operation"):
        getattr(ba, name)(np.ones((0, 3)))


@pytest.mark.parametrize("name", ["argmax_along_batch", "argmin_along_batch"])
def test_argmax_argmin_along_batch_zero_batch_size_raises(name: str) -> None:
    with pytest.raises(ValueError, match=r"attempt to get arg.* of an empty sequence"):
        getattr(ba, name)(np.ones((0, 3)))


def test_nested_sum_along_batch_zero_batch_size_is_the_identity() -> None:
    assert objects_are_equal(bn.sum_along_batch({"a": np.ones((0, 3))}), {"a": np.zeros(3)})


##########################################
#     Tests for chunk and split          #
##########################################


@pytest.mark.parametrize("func", [ba.chunk_along_batch, bn.chunk_along_batch])
def test_chunk_more_chunks_than_items_gives_empty_chunks(func: Callable) -> None:
    out = func(np.arange(2), 3)
    assert [chunk.shape for chunk in out] == [(1,), (1,), (0,)]


def test_chunk_along_seq_more_chunks_than_items_gives_empty_chunks() -> None:
    out = ba.chunk_along_seq(np.ones((2, 2)), 3)
    assert [chunk.shape for chunk in out] == [(2, 1), (2, 1), (2, 0)]


@pytest.mark.parametrize(
    "func",
    [
        lambda data: bn.chunk_along_batch(data, 2),
        lambda data: bn.chunk_along_seq(data, 2),
        lambda data: bn.split_along_batch(data, 2),
        lambda data: bn.split_along_seq(data, 2),
        bn.shuffle_along_batch,
        bn.shuffle_along_seq,
        bn.batch_size,
        bn.seq_len,
    ],
)
@pytest.mark.parametrize("data", [{}, [], {"a": [], "b": {}}])
def test_structure_without_array_raises(func: Callable, data: object) -> None:
    with pytest.raises(ValueError, match="at least one array"):
        func(data)


################################################
#     Tests for an empty structure             #
################################################


@pytest.mark.parametrize("data", [{}, [], ()])
def test_map_function_on_empty_structure_gives_the_empty_structure(data: object) -> None:
    assert bn.sum_along_batch(data) == data


def test_map_function_on_empty_structure_keeps_the_type() -> None:
    assert type(bn.sort_along_batch({})) is dict


#############################################
#     Tests for permute and tile            #
#############################################


def test_permute_along_batch_repeated_index_is_not_checked() -> None:
    assert objects_are_equal(
        ba.permute_along_batch(np.array([[0], [1], [2]]), np.array([0, 0, 1])),
        np.array([[0], [0], [1]]),
    )


def test_permute_along_seq_repeated_index_is_not_checked() -> None:
    assert objects_are_equal(
        ba.permute_along_seq(np.array([[0, 1, 2]]), np.array([2, 2, 2])), np.array([[2, 2, 2]])
    )


def test_tile_along_seq_zero_reps_gives_a_zero_length_sequence() -> None:
    assert ba.tile_along_seq(np.ones((2, 3)), 0).shape == (2, 0)


def test_nested_tile_along_seq_zero_reps_gives_a_zero_length_sequence() -> None:
    assert bn.tile_along_seq({"a": np.ones((2, 3))}, 0)["a"].shape == (2, 0)


######################################
#     Tests for zero sizes           #
######################################


def test_slice_along_seq_zero_sequence_length() -> None:
    assert ba.slice_along_seq(np.ones((2, 0)), 0, 1).shape == (2, 0)


def test_sum_along_seq_zero_sequence_length_is_the_identity() -> None:
    assert objects_are_equal(ba.sum_along_seq(np.ones((2, 0))), np.zeros(2))


def test_sort_along_batch_zero_batch_size() -> None:
    assert ba.sort_along_batch(np.ones((0, 3))).shape == (0, 3)


def test_concatenate_along_batch_of_empty_arrays() -> None:
    assert ba.concatenate_along_batch([np.ones((0, 3)), np.ones((0, 3))]).shape == (0, 3)


def test_stack_along_batch_of_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="need at least one array to stack"):
        ba.stack_along_batch([])


def test_stack_along_seq_of_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="need at least one array to stack"):
        ba.stack_along_seq([])


def test_nested_stack_of_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="at least one structure"):
        bn.stack_along_batch([])


#######################################################
#     Tests for a fully masked row or column          #
#######################################################


@pytest.mark.parametrize("name", ["sum_along_batch", "mean_along_batch", "max_along_batch"])
def test_masked_reduction_of_a_fully_masked_column_is_masked(name: str) -> None:
    x = np.ma.array(
        np.arange(6.0).reshape(3, 2), mask=[[True, False], [True, False], [True, False]]
    )
    assert np.ma.getmaskarray(getattr(ba, name)(x)).tolist() == [True, False]


@pytest.mark.parametrize("name", ["sum_along_seq", "mean_along_seq", "min_along_seq"])
def test_masked_reduction_of_a_fully_masked_row_is_masked(name: str) -> None:
    x = np.ma.array(np.arange(6.0).reshape(2, 3), mask=[[True, True, True], [False, False, False]])
    assert np.ma.getmaskarray(getattr(ba, name)(x)).tolist() == [True, False]


def test_masked_reduction_of_a_fully_masked_1d_array_is_the_masked_constant() -> None:
    assert ba.sum_along_batch(np.ma.array(np.ones(3), mask=True)) is np.ma.masked
