from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from batcharray.recursive import ApplyState, AutoApplier, DefaultApplier


@pytest.fixture
def state() -> ApplyState:
    return ApplyState(applier=AutoApplier())


####################################
#     Tests for DefaultApplier     #
####################################


def test_default_applier_str() -> None:
    assert str(DefaultApplier()) == "DefaultApplier()"


def test_default_applier_apply_str(state: ApplyState) -> None:
    assert objects_are_equal(DefaultApplier().apply([1, "abc"], str, state), "[1, 'abc']")


def test_default_applier_apply_array(state: ApplyState) -> None:
    assert objects_are_equal(
        DefaultApplier().apply(np.ones((2, 3)), lambda array: array.shape, state), (2, 3)
    )
