r"""Contain reusable input validation functions."""

from __future__ import annotations

__all__ = [
    "validate_first_array",
    "validate_min_ndim",
    "validate_same_batch_size",
    "validate_same_keys",
    "validate_same_seq_size",
    "validate_split_sections",
]

from batcharray.validation.array import validate_min_ndim, validate_split_sections
from batcharray.validation.nested import (
    validate_first_array,
    validate_same_batch_size,
    validate_same_keys,
    validate_same_seq_size,
)
