r"""Tests of the error contract: one row per line of the table of
``docs/docs/design_principles.md``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from batcharray import array as ba, nested as bn

if TYPE_CHECKING:
    from collections.abc import Callable

VECTOR = np.arange(4.0)


###############################################################
#     Tests for the `_along_seq` functions on a 1-D array     #
###############################################################

# Our own validation: the output of NumPy would be wrong or silent.
VALIDATED: dict[str, Callable[[np.ndarray], object]] = {
    "tile_along_seq": lambda x: ba.tile_along_seq(x, 2),
    "shuffle_along_seq": ba.shuffle_along_seq,
    "permute_along_seq": lambda x: ba.permute_along_seq(x, np.arange(4)),
}

# NumPy's AxisError (a ValueError and an IndexError).
AXIS_ERROR: dict[str, Callable[[np.ndarray], object]] = {
    "sum_along_seq": ba.sum_along_seq,
    "cumsum_along_seq": ba.cumsum_along_seq,
    "sort_along_seq": ba.sort_along_seq,
    "argmax_along_seq": ba.argmax_along_seq,
    "flip_along_seq": ba.flip_along_seq,
    "repeat_along_seq": lambda x: ba.repeat_along_seq(x, 2),
    "concatenate_along_seq": lambda x: ba.concatenate_along_seq([x, x]),
    "index_select_along_seq": lambda x: ba.index_select_along_seq(x, np.array([0])),
}

# NumPy's IndexError.
INDEX_ERROR: dict[str, Callable[[np.ndarray], object]] = {
    "slice_along_seq": lambda x: ba.slice_along_seq(x, 0, 2),
    "select_along_seq": lambda x: ba.select_along_seq(x, 0),
    "split_along_seq": lambda x: ba.split_along_seq(x, 2),
    "chunk_along_seq": lambda x: ba.chunk_along_seq(x, 2),
    "masked_select_along_seq": lambda x: ba.masked_select_along_seq(x, x > 1),
}


@pytest.mark.parametrize("name", VALIDATED)
def test_seq_function_on_1d_array_validated_value_error(name: str) -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        VALIDATED[name](VECTOR)


@pytest.mark.parametrize("name", AXIS_ERROR)
def test_seq_function_on_1d_array_axis_error_is_value_error(name: str) -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        AXIS_ERROR[name](VECTOR)


@pytest.mark.parametrize("name", AXIS_ERROR)
def test_seq_function_on_1d_array_axis_error_is_index_error(name: str) -> None:
    with pytest.raises(IndexError, match="out of bounds"):
        AXIS_ERROR[name](VECTOR)


@pytest.mark.parametrize("name", INDEX_ERROR)
def test_seq_function_on_1d_array_index_error(name: str) -> None:
    with pytest.raises(IndexError, match=r"too many indices|out of range"):
        INDEX_ERROR[name](VECTOR)


######################################################
#     Tests for the other rows of the error table    #
######################################################

X = np.arange(24.0).reshape(4, 3, 2)

# (id, call, exception, match): the table of docs/docs/design_principles.md.
ROWS: dict[str, tuple[Callable[[], object], type[Exception], str]] = {
    "leaf is not an ndarray (array level)": (
        lambda: bn.sum_along_batch({"a": 1}),
        TypeError,
        r"object of type int",
    ),
    "concatenate without item": (
        lambda: bn.concatenate_along_batch([]),
        ValueError,
        r"at least one structure",
    ),
    "concatenate with different keys": (
        lambda: bn.concatenate_along_batch([{"a": X}, {"b": X}]),
        ValueError,
        r"same keys",
    ),
    "concatenate with different lengths": (
        lambda: bn.concatenate_along_batch([[X], [X, X]]),
        ValueError,
        r"same length",
    ),
    "concatenate a mapping with a sequence": (
        lambda: bn.concatenate_along_batch([{"a": X}, [X]]),
        ValueError,
        r"same kind",
    ),
    "chunk with leaves of different sizes": (
        lambda: bn.chunk_along_batch({"a": X, "b": X[:3]}, 2),
        ValueError,
        r"same batch size",
    ),
    "split with leaves of different sizes": (
        lambda: bn.split_along_seq({"a": X, "b": X[:, :2]}, 2),
        ValueError,
        r"same sequence size",
    ),
    "shuffle with leaves of different sizes": (
        lambda: bn.shuffle_along_batch({"a": X, "b": X[:3]}),
        ValueError,
        r"same batch size",
    ),
    "batch_size with leaves of different sizes": (
        lambda: bn.batch_size({"a": X, "b": X[:3]}),
        ValueError,
        r"same batch size",
    ),
    "seq_len with leaves of different sizes": (
        lambda: bn.seq_len({"a": X, "b": X[:, :2]}),
        ValueError,
        r"same sequence size",
    ),
    "split size is not positive": (
        lambda: ba.split_along_batch(X, 0),
        ValueError,
        r"split size must be a positive integer",
    ),
    "split sections do not sum to the size": (
        lambda: ba.split_along_batch(X, [1, 1]),
        ValueError,
        r"sum to the axis size",
    ),
    "split sections are negative": (
        lambda: ba.split_along_batch(X, [-1, 5]),
        ValueError,
        r"non-negative",
    ),
    "index_select with an index that is not 1-D": (
        lambda: ba.index_select_along_batch(X, np.zeros((2, 2), dtype=int)),
        ValueError,
        r"indices must be a 1-D array",
    ),
    "index_select with a boolean index": (
        lambda: ba.index_select_along_batch(X, np.array([True, False, True, False])),
        ValueError,
        r"indices must be an integer array",
    ),
    "index_select with a list index": (
        lambda: ba.index_select_along_batch(X, [0, 1]),  # ty: ignore[invalid-argument-type]
        TypeError,
        r"indices must be a numpy.ndarray, but got an object of type list",
    ),
    "masked_select with a mask that is not boolean": (
        lambda: ba.masked_select_along_batch(X, np.array([0, 1, 2, 3])),
        ValueError,
        r"mask must be a boolean array",
    ),
    "masked_select with a 2-D mask": (
        lambda: ba.masked_select_along_batch(X, np.ones((4, 3), dtype=bool)),
        ValueError,
        r"mask must be a 1-D array",
    ),
    "masked_select with a list mask": (
        lambda: ba.masked_select_along_batch(X, [True, False]),  # ty: ignore[invalid-argument-type]
        TypeError,
        r"mask must be a numpy.ndarray, but got an object of type list",
    ),
    "gather with an index that is not 2-D": (
        lambda: ba.gather_along_seq(X, np.array([0, 1, 2, 0])),
        ValueError,
        r"indices must be a 2-D array",
    ),
    "gather with a different batch size": (
        lambda: ba.gather_along_seq(X, np.zeros((3, 1), dtype=int)),
        ValueError,
        r"indices must have 4 elements along the first axis",
    ),
    "gather with a boolean index": (
        lambda: ba.gather_along_seq(X, np.ones((4, 1), dtype=bool)),
        ValueError,
        r"indices must be an integer array",
    ),
    "permute with a permutation of the wrong length": (
        lambda: ba.permute_along_batch(X, np.array([0, 1])),
        ValueError,
        r"indices must have 4 elements along the first axis",
    ),
    "permute with a permutation of the wrong rank": (
        lambda: ba.permute_along_batch(X, np.zeros((4, 1), dtype=int)),
        ValueError,
        r"indices must be a 1-D array",
    ),
    "permute with a boolean permutation": (
        lambda: ba.permute_along_batch(X, np.array([True, False, True, False])),
        ValueError,
        r"indices must be an integer array",
    ),
    "out-of-range index": (
        lambda: ba.index_select_along_batch(X, np.array([4])),
        IndexError,
        r"out of bounds",
    ),
    "float index": (
        lambda: ba.index_select_along_batch(X, np.array([0.0])),
        TypeError,
        r"Cannot cast array data",
    ),
    "negative count": (
        lambda: ba.repeat_along_batch(X, -1),
        ValueError,
        r"negative",
    ),
    "bad kind": (
        lambda: ba.sort_along_batch(X, kind="bad"),
        ValueError,
        r"sort kind must be one of",
    ),
    "bad ddof": (
        lambda: ba.std_along_batch(X, ddof="a"),  # ty: ignore[invalid-argument-type]
        TypeError,
        r"ufunc",
    ),
}


@pytest.mark.parametrize("row", ROWS)
def test_error_contract(row: str) -> None:
    call, exception, match = ROWS[row]
    with pytest.raises(exception, match=match):
        call()
