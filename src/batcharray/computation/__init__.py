r"""Contain the computation models."""

from __future__ import annotations

__all__ = [
    "ArrayComputationModel",
    "BaseComputationModel",
    "MaskedArrayComputationModel",
    "AutoComputationModel",
    "argmax",
    "concatenate",
    "mean",
    "median",
    "register_computation_models",
]

from batcharray.computation.array import ArrayComputationModel
from batcharray.computation.auto import (
    AutoComputationModel,
    register_computation_models,
)
from batcharray.computation.base import BaseComputationModel
from batcharray.computation.interface import argmax, concatenate, mean, median
from batcharray.computation.masked_array import MaskedArrayComputationModel

register_computation_models()
