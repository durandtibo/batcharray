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

### What Python versions are supported?

`batcharray` supports Python 3.10 and above.

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

### How do I report a bug?

Please [open an issue](https://github.com/durandtibo/batcharray/issues) with:

1. A clear description of the problem
2. A minimal code example that reproduces the issue
3. Your environment details (Python version, OS, batcharray version)
4. Expected vs actual behavior
