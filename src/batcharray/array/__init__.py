r"""Contain functions to manipulate arrays."""

from __future__ import annotations

__all__ = [
    "concatenate_along_batch",
    "concatenate_along_seq",
    "cumprod_along_batch",
    "cumprod_along_seq",
    "cumsum_along_batch",
    "cumsum_along_seq",
    "permute_along_batch",
    "permute_along_seq",
    "shuffle_along_batch",
    "shuffle_along_seq",
    "take_along_batch",
    "take_along_seq",
    "tile_along_seq",
]

from batcharray.array.indexing import take_along_batch, take_along_seq
from batcharray.array.joining import (
    concatenate_along_batch,
    concatenate_along_seq,
    tile_along_seq,
)
from batcharray.array.math import (
    cumprod_along_batch,
    cumprod_along_seq,
    cumsum_along_batch,
    cumsum_along_seq,
)
from batcharray.array.permutation import (
    permute_along_batch,
    permute_along_seq,
    shuffle_along_batch,
    shuffle_along_seq,
)
