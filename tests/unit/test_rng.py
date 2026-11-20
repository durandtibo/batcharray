r"""Tests of P9: the randomness comes only from ``rng``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray import array as ba, nested as bn

if TYPE_CHECKING:
    from collections.abc import Callable

SHUFFLES: dict[str, Callable[..., Any]] = {
    "array_batch": ba.shuffle_along_batch,
    "array_seq": ba.shuffle_along_seq,
    "nested_batch": lambda x, **kwargs: bn.shuffle_along_batch({"a": x}, **kwargs)["a"],
    "nested_seq": lambda x, **kwargs: bn.shuffle_along_seq({"a": x}, **kwargs)["a"],
}


def make_array() -> np.ndarray:
    return np.arange(60).reshape(6, 10)


@pytest.mark.parametrize("name", SHUFFLES)
def test_rng_generator_is_accepted(name: str) -> None:
    assert SHUFFLES[name](make_array(), rng=np.random.default_rng(0)).shape == (6, 10)


@pytest.mark.parametrize("name", SHUFFLES)
def test_rng_seed_is_accepted(name: str) -> None:
    assert SHUFFLES[name](make_array(), rng=0).shape == (6, 10)


@pytest.mark.parametrize("name", SHUFFLES)
def test_rng_none_is_accepted(name: str) -> None:
    assert SHUFFLES[name](make_array(), rng=None).shape == (6, 10)


@pytest.mark.parametrize("name", SHUFFLES)
def test_rng_default_is_accepted(name: str) -> None:
    assert SHUFFLES[name](make_array()).shape == (6, 10)


@pytest.mark.parametrize("name", SHUFFLES)
def test_rng_seed_is_reproducible(name: str) -> None:
    x = make_array()
    assert objects_are_equal(SHUFFLES[name](x, rng=42), SHUFFLES[name](x, rng=42))


@pytest.mark.parametrize("name", SHUFFLES)
def test_rng_different_seeds_give_different_results(name: str) -> None:
    x = make_array()
    assert not objects_are_equal(SHUFFLES[name](x, rng=1), SHUFFLES[name](x, rng=2))


@pytest.mark.parametrize("name", SHUFFLES)
def test_rng_seed_equal_generator_with_the_same_seed(name: str) -> None:
    x = make_array()
    assert objects_are_equal(
        SHUFFLES[name](x, rng=7), SHUFFLES[name](x, rng=np.random.default_rng(7))
    )


@pytest.mark.parametrize("name", SHUFFLES)
def test_rng_generator_state_advances(name: str) -> None:
    x = make_array()
    rng = np.random.default_rng(0)
    assert not objects_are_equal(SHUFFLES[name](x, rng=rng), SHUFFLES[name](x, rng=rng))


@pytest.mark.parametrize("name", SHUFFLES)
def test_rng_does_not_use_the_global_random_state(name: str) -> None:
    x = make_array()
    np.random.seed(0)  # noqa: NPY002
    first = SHUFFLES[name](x, rng=3)
    np.random.seed(1)  # noqa: NPY002
    assert objects_are_equal(first, SHUFFLES[name](x, rng=3))


@pytest.mark.parametrize("name", SHUFFLES)
def test_rng_invalid_raises_numpy_error(name: str) -> None:
    with pytest.raises(TypeError, match="SeedSequence expects int or sequence of ints"):
        SHUFFLES[name](make_array(), rng="seed")


def test_rng_one_permutation_is_shared_by_all_the_leaves_batch() -> None:
    x = make_array()
    out = bn.shuffle_along_batch({"a": x, "b": [x.copy()], "c": (x[:, :1],)}, rng=0)
    assert objects_are_equal(out["a"], out["b"][0])


def test_rng_one_permutation_is_shared_by_all_the_leaves_batch_rows() -> None:
    x = make_array()
    out = bn.shuffle_along_batch({"a": x, "c": x[:, :1]}, rng=0)
    assert objects_are_equal(out["a"][:, :1], out["c"])


def test_rng_one_permutation_is_shared_by_all_the_leaves_seq() -> None:
    x = make_array()
    out = bn.shuffle_along_seq({"a": x, "b": [x.copy()], "c": (x[:1],)}, rng=0)
    assert objects_are_equal(out["a"][:1], out["c"][0])
