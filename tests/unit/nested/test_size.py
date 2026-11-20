from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from batcharray.nested import batch_size, seq_len

if TYPE_CHECKING:
    from collections.abc import Callable

################################
#     Tests for batch_size     #
################################


def test_batch_size_array() -> None:
    assert batch_size(np.ones((2, 3))) == 2


def test_batch_size_dict() -> None:
    assert batch_size({"a": np.ones((2, 3)), "b": np.zeros(2)}) == 2


def test_batch_size_nested() -> None:
    data = {"a": [np.ones((2, 3))], "b": (np.ones((2, 5, 4)),), "c": {"d": np.ones(2)}}
    assert batch_size(data) == 2


def test_batch_size_masked_array() -> None:
    assert batch_size({"a": np.ma.ones((4, 3))}) == 4


def test_batch_size_zero() -> None:
    assert batch_size({"a": np.ones((0, 3))}) == 0


def test_batch_size_returns_int() -> None:
    assert type(batch_size(np.ones((2, 3)))) is int


def test_batch_size_different_sizes_raises() -> None:
    with pytest.raises(ValueError, match=r"same batch size, but got \[2, 4\]"):
        batch_size({"a": np.ones((2, 3)), "b": np.ones((4, 3))})


def test_batch_size_0d_array_raises() -> None:
    with pytest.raises(ValueError, match="batch size of item 0: it has 0 dimension"):
        batch_size(np.array(1.0))


def test_batch_size_no_array_raises() -> None:
    with pytest.raises(ValueError, match="at least one array"):
        batch_size({})


@pytest.mark.parametrize("leaf", [1, "x", None])
def test_batch_size_non_array_leaf_raises(leaf: object) -> None:
    with pytest.raises(TypeError, match="Expected an array"):
        batch_size({"a": np.ones(2), "b": leaf})


#############################
#     Tests for seq_len     #
#############################


def test_seq_len_array() -> None:
    assert seq_len(np.ones((2, 3))) == 3


def test_seq_len_dict() -> None:
    assert seq_len({"a": np.ones((2, 3)), "b": np.zeros((5, 3, 4))}) == 3


def test_seq_len_nested() -> None:
    data = {"a": [np.ones((2, 3))], "b": (np.ones((4, 3, 5)),)}
    assert seq_len(data) == 3


def test_seq_len_masked_array() -> None:
    assert seq_len({"a": np.ma.ones((4, 3))}) == 3


def test_seq_len_zero() -> None:
    assert seq_len({"a": np.ones((2, 0))}) == 0


def test_seq_len_different_sizes_raises() -> None:
    with pytest.raises(ValueError, match=r"same sequence size, but got \[3, 4\]"):
        seq_len({"a": np.ones((2, 3)), "b": np.ones((2, 4))})


def test_seq_len_1d_array_raises() -> None:
    with pytest.raises(ValueError, match="sequence size of item 1: it has 1 dimension"):
        seq_len({"a": np.ones((2, 3)), "b": np.ones(2)})


def test_seq_len_no_array_raises() -> None:
    with pytest.raises(ValueError, match="at least one array"):
        seq_len([])


@pytest.mark.parametrize("func", [batch_size, seq_len])
def test_size_does_not_modify_data(func: Callable) -> None:
    data = {"a": np.ones((2, 3))}
    func(data)
    assert data["a"].shape == (2, 3)
