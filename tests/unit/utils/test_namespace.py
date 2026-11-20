from __future__ import annotations

import numpy as np
import pytest

from batcharray.utils import get_namespace
from tests.unit.helpers import MyArray, MyMaskedArray

###################################
#     Tests for get_namespace     #
###################################


@pytest.mark.parametrize(
    "array",
    [np.ones((2, 3)), np.ones((2, 3), dtype=int), np.ones(3), np.ones((2, 3)).view(MyArray)],
)
def test_get_namespace_array(array: np.ndarray) -> None:
    assert get_namespace(array) is np


@pytest.mark.parametrize(
    "array", [np.ma.ones((2, 3)), np.ma.ones(3), np.ma.ones((2, 3)).view(MyMaskedArray)]
)
def test_get_namespace_masked_array(array: np.ndarray) -> None:
    assert get_namespace(array) is np.ma
