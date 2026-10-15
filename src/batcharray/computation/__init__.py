r"""Contain the computation models.

Note:
    ``max``, ``min`` and ``sum`` have the same names as Python builtins. Use
    them through the namespace (``cmpt.max(arr)``) and avoid
    ``from batcharray.computation import *``, which would shadow the
    builtins.
"""

from __future__ import annotations

__all__ = [
    "ArrayComputationModel",
    "AutoComputationModel",
    "BaseComputationModel",
    "MaskedArrayComputationModel",
    "argmax",
    "argmin",
    "argsort",
    "concatenate",
    "max",
    "mean",
    "median",
    "min",
    "prod",
    "register_computation_models",
    "sort",
    "std",
    "sum",
    "var",
]

from batcharray.computation.array import ArrayComputationModel
from batcharray.computation.auto import (
    AutoComputationModel,
    register_computation_models,
)
from batcharray.computation.base import BaseComputationModel
from batcharray.computation.interface import (
    argmax,
    argmin,
    argsort,
    concatenate,
    max,  # noqa: A004
    mean,
    median,
    min,  # noqa: A004
    prod,
    sort,
    std,
    sum,  # noqa: A004
    var,
)
from batcharray.computation.masked_array import MaskedArrayComputationModel

register_computation_models()
