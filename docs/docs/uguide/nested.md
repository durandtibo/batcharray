# Nested Data Structures

The `batcharray.nested` module provides functions to manipulate nested data structures containing
NumPy arrays. This is particularly useful when working with complex data like dictionaries or lists
of arrays that represent batches or sequences.

## Overview

When working with machine learning or data processing pipelines, you often have data organized in
nested structures:

- Dictionaries with multiple arrays (e.g., features, labels, metadata)
- Lists or tuples of arrays
- Combinations of both

The `nested` module allows you to apply batch or sequence operations to all arrays in these
structures simultaneously.

## Working with Dictionaries

### Basic Operations

```python
import numpy as np
from batcharray import nested

# Create a batch as a dictionary
batch = {
    "features": np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
    "labels": np.array([0, 1, 0, 1, 0]),
    "weights": np.array([1.0, 2.0, 1.5, 2.5, 1.0]),
}

# Slice all arrays in the batch
sliced = nested.slice_along_batch(batch, stop=3)
# Result: {
#     "features": [[1, 2], [3, 4], [5, 6]],
#     "labels": [0, 1, 0],
#     "weights": [1.0, 2.0, 1.5]
# }

# Split into multiple batches
batches = nested.split_along_batch(batch, split_size_or_sections=2)
# Returns list of 3 dictionaries, each with 2, 2, and 1 items
```

### Indexing and Selection

```python
import numpy as np
from batcharray import nested

batch = {
    "data": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    "mask": np.array([True, False, True]),
}

# Select specific indices
indices = np.array([0, 2])
selected = nested.index_select_along_batch(batch, indices=indices)
# Result: {
#     "data": [[1, 2, 3], [7, 8, 9]],
#     "mask": [True, True]
# }

# Select using boolean mask
mask = np.array([True, False, True])
masked = nested.masked_select_along_batch(batch, mask=mask)
```

### Combining and Concatenating

```python
import numpy as np
from batcharray import nested

batch1 = {"x": np.array([[1, 2], [3, 4]]), "y": np.array([0, 1])}

batch2 = {"x": np.array([[5, 6], [7, 8]]), "y": np.array([1, 0])}

# Concatenate batches
combined = nested.concatenate_along_batch([batch1, batch2])
# Result: {
#     "x": [[1, 2], [3, 4], [5, 6], [7, 8]],
#     "y": [0, 1, 1, 0]
# }
```

## Sequence Operations

Just like with arrays, you can perform sequence operations on nested structures:

```python
import numpy as np
from batcharray import nested

sequences = {
    "inputs": np.array(
        [
            [[1, 2], [3, 4], [5, 6]],  # Sequence 1
            [[7, 8], [9, 10], [11, 12]],  # Sequence 2
        ]
    ),
    "targets": np.array([[[0], [1], [0]], [[1], [1], [0]]]),
}

# Slice sequences
sliced = nested.slice_along_seq(sequences, start=1)
# Takes timesteps 1 and 2 from both arrays

# Chunk sequences
chunks = nested.chunk_along_seq(sequences, chunks=3)
# Splits each sequence into 3 parts
```

## Reductions and Statistics

Compute statistics across nested structures:

```python
import numpy as np
from batcharray import nested

batch = {
    "scores": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    "values": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
}

# Compute mean along batch
means = nested.mean_along_batch(batch)
# Result: {
#     "scores": [2.5, 3.5, 4.5],
#     "values": [0.25, 0.35, 0.45]
# }

# Find maximum values
maxes = nested.max_along_batch(batch)
```

## Element-wise Operations

`batcharray.nested` does not wrap the element-wise NumPy functions, because they have no batch or
sequence axis. Apply any function to all the arrays of a structure with `recursive_apply` of
`coola`:

```python
import numpy as np
from batcharray.utils import apply_to_arrays

data = {
    "angles": np.array([[0.1, np.pi / 2], [np.pi, 3 * np.pi / 2]]),
    "values": np.array([[1, -2], [3, -4]]),
}

sines = apply_to_arrays(data["angles"], np.sin)
absolute = apply_to_arrays(data, np.abs)
exponential = apply_to_arrays(data, np.exp)
clipped = apply_to_arrays(data, lambda x: np.clip(x, -2, 2))
```

See the [migration guide](../migration.md) for the functions that were removed.

## Shuffling and Permutation

Randomly shuffle or permute data while maintaining relationships:

```python
import numpy as np
from batcharray import nested

batch = {
    "images": np.random.randn(100, 28, 28),
    "labels": np.random.randint(0, 10, 100),
}

# Shuffle all arrays with the same permutation
shuffled = nested.shuffle_along_batch(batch)
# Images and labels are shuffled together

# Or provide your own permutation
perm = np.random.permutation(100)
permuted = nested.permute_along_batch(batch, permutation=perm)
```

## Sorting

Sort nested structures by values in one array:

```python
import numpy as np
from batcharray import nested

batch = {
    "scores": np.array([3.0, 1.0, 4.0, 2.0]),
    "names": np.array(["Alice", "Bob", "Charlie", "David"]),
}

# Get sorting indices for scores
indices = nested.argsort_along_batch(batch["scores"])

# Sort all arrays by the scores
sorted_batch = nested.index_select_along_batch(batch, indices=indices)
# Result: scores [1.0, 2.0, 3.0, 4.0], names ["Bob", "David", "Alice", "Charlie"]
```

## Converting to Lists

Convert nested structures to native Python lists with `apply_to_arrays`:

```python
import numpy as np
from batcharray.utils import apply_to_arrays

batch = {"data": np.array([[1, 2], [3, 4]]), "info": np.array([True, False])}

# Convert to lists recursively
as_lists = apply_to_arrays(batch, np.ndarray.tolist)
# Result: {
#     "data": [[1, 2], [3, 4]],
#     "info": [True, False]
# }
```

## Common Use Cases

### Machine Learning Pipelines

```python
import numpy as np
from batcharray import nested

# Training data batch
batch = {
    "inputs": np.random.randn(32, 784),  # MNIST-like data
    "labels": np.random.randint(0, 10, 32),
    "sample_weights": np.random.rand(32),
}

# Create train/val split
train_batch = nested.slice_along_batch(batch, stop=24)
val_batch = nested.slice_along_batch(batch, start=24)
```

### Data Augmentation

```python
import numpy as np
from batcharray import nested

# Original batch
batch = {"images": np.random.randn(16, 28, 28), "labels": np.random.randint(0, 10, 16)}

# Shuffle for data augmentation
augmented = nested.shuffle_along_batch(batch)

# Split into mini-batches
mini_batches = nested.split_along_batch(augmented, split_size_or_sections=4)
```

## Function Reference

The `nested` module provides all operations from the `array` module, but for nested structures.
Here's a non comprehensive list organized by category:

### Comparison and Sorting

- `argsort_along_batch()` - Get sorting indices (batch)
- `argsort_along_seq()` - Get sorting indices (sequence)
- `sort_along_batch()` - Sort nested arrays (batch)
- `sort_along_seq()` - Sort nested arrays (sequence)

### Indexing and Selection

- `gather_along_seq()` - Gather different indices for each sample (sequence)
- `index_select_along_batch()` - Select using indices (batch)
- `index_select_along_seq()` - Select using indices (sequence)
- `masked_select_along_batch()` - Select using mask (batch)
- `masked_select_along_seq()` - Select using mask (sequence)

### Joining Operations

- `concatenate_along_batch()` - Concatenate nested structures (batch)
- `concatenate_along_seq()` - Concatenate nested structures (sequence)
- `repeat_along_batch()` - Repeat each item (batch)
- `repeat_along_seq()` - Repeat each item (sequence)
- `tile_along_seq()` - Tile nested arrays along sequence

### Mathematical Operations

- `cumprod_along_batch()` - Cumulative product (batch)
- `cumprod_along_seq()` - Cumulative product (sequence)
- `cumsum_along_batch()` - Cumulative sum (batch)
- `cumsum_along_seq()` - Cumulative sum (sequence)

### Permutation and Shuffling

- `flip_along_batch()` - Reverse the order (batch)
- `flip_along_seq()` - Reverse the order (sequence)
- `permute_along_batch()` - Apply permutation (batch)
- `permute_along_seq()` - Apply permutation (sequence)
- `shuffle_along_batch()` - Random shuffle (batch)
- `shuffle_along_seq()` - Random shuffle (sequence)

### Reduction Operations

- `all_along_batch()` - Logical and (batch)
- `all_along_seq()` - Logical and (sequence)
- `any_along_batch()` - Logical or (batch)
- `any_along_seq()` - Logical or (sequence)
- `max_along_batch()` - Maximum (batch)
- `max_along_seq()` - Maximum (sequence)
- `min_along_batch()` - Minimum (batch)
- `min_along_seq()` - Minimum (sequence)
- `argmax_along_batch()` - Indices of maximum (batch)
- `argmax_along_seq()` - Indices of maximum (sequence)
- `argmin_along_batch()` - Indices of minimum (batch)
- `argmin_along_seq()` - Indices of minimum (sequence)
- `mean_along_batch()` - Mean (batch)
- `mean_along_seq()` - Mean (sequence)
- `median_along_batch()` - Median (batch)
- `median_along_seq()` - Median (sequence)
- `prod_along_batch()` - Product (batch)
- `prod_along_seq()` - Product (sequence)
- `std_along_batch()` - Standard deviation (batch)
- `std_along_seq()` - Standard deviation (sequence)
- `sum_along_batch()` - Sum (batch)
- `sum_along_seq()` - Sum (sequence)
- `var_along_batch()` - Variance (batch)
- `var_along_seq()` - Variance (sequence)

### Slicing Operations

- `chunk_along_batch()` - Split into chunks (batch)
- `chunk_along_seq()` - Split into chunks (sequence)
- `select_along_batch()` - Select single index (batch)
- `select_along_seq()` - Select single index (sequence)
- `slice_along_batch()` - Slice range (batch)
- `slice_along_seq()` - Slice range (sequence)
- `split_along_batch()` - Split into sections (batch)
- `split_along_seq()` - Split into sections (sequence)

For detailed API documentation, see the [nested API reference](../refs/nested.md).
