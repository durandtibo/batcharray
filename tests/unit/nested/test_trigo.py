from collections.abc import Callable

import numpy as np
import pytest
from coola import objects_are_allclose

from batcharray import nested

DTYPES = [np.float32, np.float64, np.int64]
POINTWISE_FUNCTIONS = [
    (np.arccos, nested.arccos),
    (np.arccosh, nested.arccosh),
    (np.arcsin, nested.arcsin),
    (np.arcsinh, nested.arcsinh),
    (np.arctan, nested.arctan),
    (np.arctanh, nested.arctanh),
    (np.cos, nested.cos),
    (np.cosh, nested.cosh),
    (np.sin, nested.sin),
    (np.sinh, nested.sinh),
    (np.tan, nested.tan),
    (np.tanh, nested.tanh),
]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("functions", POINTWISE_FUNCTIONS)
def test_trigo_pointwise_function_array(
    dtype: np.dtype, functions: tuple[Callable, Callable]
) -> None:
    np_fn, nested_fn = functions
    array = np.random.randn(5, 2).astype(dtype=dtype)
    assert objects_are_allclose(nested_fn(array), np_fn(array), equal_nan=True)
