r"""Contain functions to manipulate arrays."""

from __future__ import annotations

__all__ = ["concatenate_along_batch", "concatenate_along_seq", "take_along_batch", "take_along_seq"]

from batcharray.array.indexing import take_along_batch, take_along_seq
from batcharray.array.joining import concatenate_along_batch, concatenate_along_seq
