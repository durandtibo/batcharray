from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
from coola import objects_are_equal

from batcharray.computation import (
    ArrayComputationModel,
    AutoComputationModel,
    BaseComputationModel,
    MaskedArrayComputationModel,
)

##########################################
#     Tests for AutoComputationModel     #
##########################################


def test_auto_computation_model_repr() -> None:
    assert repr(AutoComputationModel()).startswith("AutoComputationModel(")


def test_auto_computation_model_str() -> None:
    assert str(AutoComputationModel()).startswith("AutoComputationModel(")


@patch.dict(AutoComputationModel.registry, {}, clear=True)
def test_auto_computation_model_add_computation_model() -> None:
    assert len(AutoComputationModel.registry) == 0
    AutoComputationModel.add_computation_model(np.ndarray, ArrayComputationModel())
    assert AutoComputationModel.registry[np.ndarray] == ArrayComputationModel()


@patch.dict(AutoComputationModel.registry, {}, clear=True)
def test_auto_computation_model_add_computation_model_exist_ok_false() -> None:
    assert len(AutoComputationModel.registry) == 0
    AutoComputationModel.add_computation_model(np.ndarray, ArrayComputationModel())
    with pytest.raises(
        RuntimeError, match="A computation model .* is already registered for the array type"
    ):
        AutoComputationModel.add_computation_model(np.ndarray, ArrayComputationModel())


@patch.dict(AutoComputationModel.registry, {}, clear=True)
def test_auto_computation_model_add_computation_model_exist_ok_true() -> None:
    assert len(AutoComputationModel.registry) == 0
    AutoComputationModel.add_computation_model(np.ndarray, Mock(spec=BaseComputationModel))
    AutoComputationModel.add_computation_model(np.ndarray, ArrayComputationModel(), exist_ok=True)
    assert AutoComputationModel.registry[np.ndarray] == ArrayComputationModel()


def test_auto_computation_model_has_computation_model_true() -> None:
    assert AutoComputationModel.has_computation_model(np.ndarray)


def test_auto_computation_model_has_computation_model_false() -> None:
    assert not AutoComputationModel.has_computation_model(str)


def test_auto_computation_model_find_computation_model_ndarray() -> None:
    assert AutoComputationModel.find_computation_model(np.ndarray) == ArrayComputationModel()


def test_auto_computation_model_find_computation_model_masked_array() -> None:
    assert (
        AutoComputationModel.find_computation_model(np.ma.MaskedArray)
        == MaskedArrayComputationModel()
    )


def test_auto_computation_model_find_computation_model_missing() -> None:
    with pytest.raises(TypeError, match="Incorrect array type:"):
        AutoComputationModel.find_computation_model(str)


def test_auto_computation_model_registered_computation_models() -> None:
    assert len(AutoComputationModel.registry) >= 2
    assert AutoComputationModel.registry[np.ndarray] == ArrayComputationModel()
    assert AutoComputationModel.registry[np.ma.MaskedArray] == MaskedArrayComputationModel()


def test_auto_computation_model_concatenate() -> None:
    out = AutoComputationModel().concatenate(
        [
            np.array([[0, 1, 2], [4, 5, 6]]),
            np.array([[10, 11, 12], [13, 14, 15]]),
        ],
        axis=0,
    )
    assert objects_are_equal(out, np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))
