r"""batcharray: NumPy array manipulation for batches and sequences.

`batcharray` is a lightweight library built on top of NumPy to manipulate
nested data structures with NumPy arrays. It provides functions for arrays
where the first axis is the batch axis, and optionally the second axis is
the sequence axis.

Main modules:
    - array: Functions to manipulate single arrays with batch/sequence dimensions
    - nested: Functions to manipulate nested structures (dicts, lists) of arrays
    - computation: Computation models for abstracting array operations

Example:
```pycon
>>> import numpy as np
>>> from batcharray import array, nested
>>> # Working with single arrays
>>> batch = np.array([[1, 2], [3, 4], [5, 6]])
>>> sliced = array.slice_along_batch(batch, stop=2)
>>> # Working with nested structures
>>> data = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
>>> sliced = nested.slice_along_batch(data, stop=2)

```

For more information, see https://durandtibo.github.io/batcharray/
"""

from __future__ import annotations

__all__ = ["__version__", "array", "computation", "nested"]

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from batcharray import array, computation, nested

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed, fallback if needed
    __version__ = "0.0.0"

_SUBPACKAGES = frozenset({"array", "computation", "nested"})


def __getattr__(name: str) -> Any:
    # Lazily import the subpackages so `import batcharray` stays cheap but
    # `batcharray.array` / `batcharray.nested` work without an explicit import.
    if name in _SUBPACKAGES:
        return import_module(f"{__name__}.{name}")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
