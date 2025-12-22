# Frequently Asked Questions (FAQ)

## General Questions

### What is batcharray?

`batcharray` is a lightweight Python library built on top of NumPy that provides functions to manipulate nested data structures containing NumPy arrays. It's particularly useful when working with batches of data where the first axis represents the batch dimension, and optionally the second axis represents a sequence dimension.

### Why should I use batcharray?

You should use `batcharray` if you:

- Work with complex nested data structures (dictionaries, lists) containing NumPy arrays
- Need to manipulate batches of data consistently across multiple arrays
- Want to avoid writing repetitive batch manipulation code
- Work with sequences where batch and time dimensions need special handling
- Need to handle masked arrays alongside regular arrays seamlessly

### How is batcharray different from NumPy?

While NumPy provides low-level array operations, `batcharray`:

- Provides higher-level abstractions for batch and sequence operations
- Handles nested data structures automatically (dictionaries, lists of arrays)
- Offers a consistent API specifically designed for batch operations
- Seamlessly works with both regular and masked arrays
- Includes utilities for recursive operations on complex structures

### What are the main use cases?

Common use cases include:

- Machine learning data pipelines (batching, splitting, shuffling)
- Processing time series data with variable length sequences
- Managing complex data structures in scientific computing
- Data preprocessing and augmentation
- Batch processing in production systems

## Installation and Setup

### How do I install batcharray?

The simplest way is using pip:

```bash
pip install batcharray
```

For all optional dependencies:

```bash
pip install batcharray[all]
```

### What Python versions are supported?

`batcharray` supports Python 3.10 and above (3.10, 3.11, 3.12, 3.13, 3.14).

### What are the dependencies?

The minimal dependencies are:

- `numpy >= 1.22, < 3.0`
- `coola >= 0.9.1, < 1.0`

### Can I use batcharray with PyTorch or TensorFlow?

Yes! While `batcharray` is designed for NumPy arrays, you can easily convert between NumPy and other frameworks:

```python
import numpy as np
import torch
from batcharray import array

# Convert PyTorch tensor to NumPy
torch_tensor = torch.randn(10, 5)
numpy_array = torch_tensor.numpy()

# Use batcharray
sliced = array.slice_along_batch(numpy_array, stop=5)

# Convert back to PyTorch
result_tensor = torch.from_numpy(sliced)
```

## Usage Questions

### How do I work with dictionaries of arrays?

Use the `nested` module:

```python
import numpy as np
from batcharray import nested

batch = {"features": np.array([[1, 2], [3, 4], [5, 6]]), "labels": np.array([0, 1, 0])}

# Slice all arrays together
sliced = nested.slice_along_batch(batch, stop=2)
# Result: {"features": [[1, 2], [3, 4]], "labels": [0, 1]}
```

### What's the difference between `array` and `nested` modules?

- **`array` module**: Works with single NumPy arrays
- **`nested` module**: Works with nested structures (dicts, lists) containing arrays

Example:

```python
from batcharray import array, nested
import numpy as np

# Single array - use array module
arr = np.array([[1, 2], [3, 4]])
sliced_arr = array.slice_along_batch(arr, stop=1)

# Nested structure - use nested module
data = {"a": np.array([[1, 2], [3, 4]]), "b": np.array([5, 6])}
sliced_data = nested.slice_along_batch(data, stop=1)
```

### How do I handle missing data?

Use NumPy masked arrays:

```python
import numpy as np
import numpy.ma as ma
from batcharray import array

# Create masked array (some values are missing)
data = ma.array(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    mask=[
        [False, True, False],  # 2nd value is missing
        [False, False, True],  # 3rd value is missing
        [True, False, False],
    ],  # 1st value is missing
)

# Operations automatically handle masked values
mean = array.mean_along_batch(data)
# Computes mean only over non-masked values
```

### Can I shuffle a batch while keeping arrays synchronized?

Yes! The nested module maintains relationships between arrays:

```python
import numpy as np
from batcharray import nested

batch = {
    "images": np.random.randn(100, 28, 28),
    "labels": np.random.randint(0, 10, 100),
}

# Shuffles both arrays with the same permutation
shuffled = nested.shuffle_along_batch(batch)
# Images and labels remain synchronized
```

### How do I split a batch into training and validation sets?

Use slicing or splitting functions:

```python
import numpy as np
from batcharray import nested

data = {"X": np.random.randn(100, 10), "y": np.random.randint(0, 2, 100)}

# Split 80/20
train = nested.slice_along_batch(data, stop=80)
val = nested.slice_along_batch(data, start=80)

# Or use split_along_batch
splits = nested.split_along_batch(data, split_size_or_sections=[80, 20])
train, val = splits[0], splits[1]
```

## Performance Questions

### Is batcharray fast?

Yes! `batcharray` is built on NumPy, which uses highly optimized C and Fortran libraries. The overhead from `batcharray` is minimal as it mostly provides convenient wrappers around NumPy operations.

### Does batcharray support parallel processing?

`batcharray` leverages NumPy's built-in parallelization for operations that support it. For custom parallel processing, you can use:

```python
import numpy as np
from batcharray import array
from concurrent.futures import ProcessPoolExecutor


def process_batch(batch):
    return array.mean_along_seq(batch)


# Split data and process in parallel
chunks = array.chunk_along_batch(large_array, chunks=4)
with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_batch, chunks))
```

### How much memory does batcharray use?

`batcharray` operations typically use views rather than copies when possible, minimizing memory overhead. However, some operations (like sorting) may require copies. Check the documentation for specific functions.

## Troubleshooting

### Why am I getting "axis out of bounds" errors?

This usually means your array doesn't have the expected number of dimensions. `batcharray` assumes:

- Batch operations: arrays have shape `(batch_size, ...)`
- Sequence operations: arrays have shape `(batch_size, seq_len, ...)`

Check your array shapes:

```python
import numpy as np

arr = np.array([1, 2, 3])  # 1D array
print(arr.shape)  # (3,)

# Need at least 2D for sequence operations
arr_2d = arr.reshape(1, -1)  # Shape: (1, 3)
```

### Why doesn't `nested` work with my data structure?

The `nested` module expects dictionaries where all values are arrays with compatible shapes along the batch/sequence dimension. Ensure:

1. All arrays have the same batch size (first dimension)
2. For sequence operations, all arrays have compatible sequence length (second dimension)

```python
import numpy as np
from batcharray import nested

# This works - both arrays have batch_size=3
good_data = {
    "a": np.array([[1, 2], [3, 4], [5, 6]]),  # (3, 2)
    "b": np.array([7, 8, 9]),  # (3,)
}

# This fails - incompatible batch sizes
bad_data = {
    "a": np.array([[1, 2], [3, 4]]),  # batch_size=2
    "b": np.array([7, 8, 9]),  # batch_size=3
}
```

### How do I debug issues with nested structures?

Use the `utils` module to inspect your data:

```python
import numpy as np
from batcharray.utils import bfs_array

data = {"a": np.array([1, 2]), "b": {"c": np.array([3, 4])}}

# List all arrays
arrays = list(bfs_array(data))
print(f"Found {len(arrays)} arrays")

# Check shapes
for i, arr in enumerate(arrays):
    print(f"Array {i}: shape={arr.shape}, dtype={arr.dtype}")
```

## Advanced Topics

### Can I create custom operations?

Yes! You can extend `batcharray` functionality:

```python
import numpy as np
from batcharray.recursive import recursive_apply


def custom_normalize(data):
    """Custom normalization for nested data."""

    def normalize_array(x):
        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating):
            return (x - x.mean()) / (x.std() + 1e-8)
        return x

    return recursive_apply(data, normalize_array)
```

### How do I contribute to batcharray?

We welcome contributions! Please see the [CONTRIBUTING.md](.github/CONTRIBUTING.md) guide for details on:

- Setting up the development environment
- Running tests
- Code style guidelines
- Submitting pull requests

### Is the API stable?

⚠️ `batcharray` is currently in development (version 0.x). The API may change between releases. We're working toward a stable 1.0.0 release. See the [CHANGELOG.md](../CHANGELOG.md) for version-specific changes.

### Where can I get help?

- **Documentation**: [https://durandtibo.github.io/batcharray/](https://durandtibo.github.io/batcharray/)
- **Issues**: [GitHub Issues](https://github.com/durandtibo/batcharray/issues)
- **Discussions**: [GitHub Discussions](https://github.com/durandtibo/batcharray/discussions)

### How do I report a bug?

Please [open an issue](https://github.com/durandtibo/batcharray/issues) with:

1. A clear description of the problem
2. A minimal code example that reproduces the issue
3. Your environment details (Python version, OS, batcharray version)
4. Expected vs actual behavior

## Migration Questions

### How do I migrate from version 0.1.0 to 0.2.0?

Key changes in 0.2.0:

1. **Build system**: Changed from poetry to uv/hatchling
2. **Python support**: Minimum version is now 3.10 (was 3.9)
3. **Dependencies**: Updated to coola >= 0.9.1

Code changes should be minimal. Check the [CHANGELOG.md](../CHANGELOG.md) for specific details.

### Can I use batcharray with older NumPy versions?

`batcharray` requires NumPy >= 1.22. For older NumPy versions, you may need to use an older version of `batcharray`. Check the compatibility table in the [README](../README.md).
