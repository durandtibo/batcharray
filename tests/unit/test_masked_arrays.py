from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from coola.equality import objects_are_equal

from batcharray import array as ba
from batcharray.constants import BATCH_AXIS, SEQ_AXIS

if TYPE_CHECKING:
    from collections.abc import Callable


def make_masked_array() -> np.ma.MaskedArray:
    r"""Create a masked array of shape ``(2, 3, 4)`` with fully masked
    cells along the batch axis and along the sequence axis."""
    data = np.random.default_rng(0).normal(size=(2, 3, 4))
    mask = np.zeros((2, 3, 4), dtype=bool)
    mask[:, 1, 1] = True  # fully masked along the batch axis
    mask[0, :, 0] = True  # fully masked along the sequence axis
    mask[1, 2, 3] = True
    return np.ma.array(data, mask=mask)


INDICES = np.array([1, 0])
BOOL_MASK = np.array([True, False])
PERMUTATIONS = {"batch": np.array([1, 0]), "seq": np.array([2, 0, 1])}

# Each case is (function of batcharray.array, reference built with numpy.ma).
FUNCTIONS: dict[str, tuple[Callable[[np.ndarray], Any], Callable[[np.ma.MaskedArray], Any]]] = {}
for _name, _axis in [("batch", BATCH_AXIS), ("seq", SEQ_AXIS)]:
    for _f in ["max", "min", "mean", "median", "sum", "prod", "std", "var"]:
        FUNCTIONS[f"{_f}_along_{_name}"] = (
            getattr(ba, f"{_f}_along_{_name}"),
            lambda x, f=_f, a=_axis: getattr(np.ma, f)(x, axis=a),
        )
    FUNCTIONS[f"any_along_{_name}"] = (
        getattr(ba, f"any_along_{_name}"),
        lambda x, a=_axis: np.ma.any(x, axis=a),
    )
    FUNCTIONS[f"all_along_{_name}"] = (
        getattr(ba, f"all_along_{_name}"),
        lambda x, a=_axis: np.ma.all(x, axis=a),
    )
    for _f in ["cumsum", "cumprod", "sort"]:
        FUNCTIONS[f"{_f}_along_{_name}"] = (
            getattr(ba, f"{_f}_along_{_name}"),
            lambda x, f=_f, a=_axis: getattr(np.ma, f)(x, axis=a),
        )
    FUNCTIONS[f"flip_along_{_name}"] = (
        getattr(ba, f"flip_along_{_name}"),
        lambda x, a=_axis: np.ma.array(np.flip(x, axis=a)),
    )
    FUNCTIONS[f"index_select_along_{_name}"] = (
        lambda x, n=_name: getattr(ba, f"index_select_along_{n}")(x, INDICES),
        lambda x, a=_axis: np.take(x, INDICES, axis=a),
    )
    FUNCTIONS[f"permute_along_{_name}"] = (
        lambda x, n=_name: getattr(ba, f"permute_along_{n}")(x, PERMUTATIONS[n]),
        lambda x, a=_axis, n=_name: np.take(x, PERMUTATIONS[n], axis=a),
    )
    FUNCTIONS[f"repeat_along_{_name}"] = (
        lambda x, n=_name: getattr(ba, f"repeat_along_{n}")(x, 2),
        lambda x, a=_axis: np.repeat(x, 2, axis=a),
    )
    FUNCTIONS[f"concatenate_along_{_name}"] = (
        lambda x, n=_name: getattr(ba, f"concatenate_along_{n}")([x, x]),
        lambda x, a=_axis: np.ma.concatenate([x, x], axis=a),
    )
FUNCTIONS["tile_along_seq"] = (
    lambda x: ba.tile_along_seq(x, 2),
    lambda x: np.tile(x, (1, 2, 1)),
)
FUNCTIONS["masked_select_along_batch"] = (
    lambda x: ba.masked_select_along_batch(x, BOOL_MASK),
    lambda x: x[BOOL_MASK],
)

INDEX_FUNCTIONS = {
    "argsort_along_batch": (ba.argsort_along_batch, lambda x: np.ma.argsort(x, axis=BATCH_AXIS)),
    "argsort_along_seq": (ba.argsort_along_seq, lambda x: np.ma.argsort(x, axis=SEQ_AXIS)),
    "argmax_along_batch": (ba.argmax_along_batch, lambda x: np.ma.argmax(x, axis=BATCH_AXIS)),
    "argmax_along_seq": (ba.argmax_along_seq, lambda x: np.ma.argmax(x, axis=SEQ_AXIS)),
    "argmin_along_batch": (ba.argmin_along_batch, lambda x: np.ma.argmin(x, axis=BATCH_AXIS)),
    "argmin_along_seq": (ba.argmin_along_seq, lambda x: np.ma.argmin(x, axis=SEQ_AXIS)),
}


##########################################
#     Tests for masked arrays (P11)      #
##########################################


@pytest.mark.parametrize("name", FUNCTIONS)
def test_masked_array_gives_masked_array(name: str) -> None:
    func, _ = FUNCTIONS[name]
    assert isinstance(func(make_masked_array()), np.ma.MaskedArray)


@pytest.mark.parametrize("name", FUNCTIONS)
def test_masked_array_equals_numpy_ma(name: str) -> None:
    func, reference = FUNCTIONS[name]
    x = make_masked_array()
    assert objects_are_equal(func(x), reference(x))


@pytest.mark.parametrize("name", INDEX_FUNCTIONS)
def test_masked_array_index_function_gives_array(name: str) -> None:
    func, _ = INDEX_FUNCTIONS[name]
    assert type(func(make_masked_array())) is np.ndarray


@pytest.mark.parametrize("name", INDEX_FUNCTIONS)
def test_masked_array_index_function_equals_numpy_ma(name: str) -> None:
    func, reference = INDEX_FUNCTIONS[name]
    x = make_masked_array()
    assert objects_are_equal(np.asarray(func(x)), np.asarray(reference(x)))
