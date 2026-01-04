# Nested Data Structures

The `batcharray.nested` module provides functions to manipulate nested data structures containing NumPy arrays. This is particularly useful when working with complex data like dictionaries or lists of arrays that represent batches or sequences.

## Overview

When working with machine learning or data processing pipelines, you often have data organized in nested structures:

- Dictionaries with multiple arrays (e.g., features, labels, metadata)
- Lists or tuples of arrays
- Combinations of both

The `nested` module allows you to apply batch or sequence operations to all arrays in these structures simultaneously.

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
maxes = nested.amax_along_batch(batch)
```

## Mathematical Operations

Apply mathematical operations element-wise to all arrays:

```python
import numpy as np
from batcharray import nested

data = {
    "angles": np.array([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]]),
    "values": np.array([[1, -2], [3, -4]]),
}

# Apply trigonometric functions
sines = nested.sin(data["angles"])
cosines = nested.cos(data["angles"])

# Apply other math functions
absolute = nested.abs(data)
# Result: {
#     "angles": [[0, π/2], [π, 3π/2]],
#     "values": [[1, 2], [3, 4]]  # Absolute values
# }

exponential = nested.exp(data)
logarithm = nested.log(nested.abs(data))
clipped = nested.clip(data, -2, 2)
```

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
sorted_batch = nested.take_along_batch(batch, indices=indices)
# Result: scores [1.0, 2.0, 3.0, 4.0], names ["Bob", "David", "Alice", "Charlie"]
```

## Converting to Lists

Convert nested structures to native Python lists:

```python
import numpy as np
from batcharray import nested

batch = {"data": np.array([[1, 2], [3, 4]]), "info": np.array([True, False])}

# Convert to lists recursively
as_lists = nested.to_list(batch)
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

## Complete Function Reference

The `nested` module provides all operations from the `array` module, but for nested structures. Here's a complete list organized by category:

### Comparison and Sorting
- `argsort_along_batch()` - Get sorting indices (batch)
- `argsort_along_seq()` - Get sorting indices (sequence)
- `sort_along_batch()` - Sort nested arrays (batch)
- `sort_along_seq()` - Sort nested arrays (sequence)

### Conversion
- `to_list()` - Convert arrays to native Python lists

### Indexing and Selection
- `index_select_along_batch()` - Select using indices (batch)
- `index_select_along_seq()` - Select using indices (sequence)
- `masked_select_along_batch()` - Select using mask (batch)
- `masked_select_along_seq()` - Select using mask (sequence)
- `take_along_batch()` - Take elements using index array (batch)
- `take_along_seq()` - Take elements using index array (sequence)

### Joining Operations
- `concatenate_along_batch()` - Concatenate nested structures (batch)
- `concatenate_along_seq()` - Concatenate nested structures (sequence)
- `tile_along_seq()` - Tile nested arrays along sequence

### Mathematical Operations
- `cumprod_along_batch()` - Cumulative product (batch)
- `cumprod_along_seq()` - Cumulative product (sequence)
- `cumsum_along_batch()` - Cumulative sum (batch)
- `cumsum_along_seq()` - Cumulative sum (sequence)

### Permutation and Shuffling
- `permute_along_batch()` - Apply permutation (batch)
- `permute_along_seq()` - Apply permutation (sequence)
- `shuffle_along_batch()` - Random shuffle (batch)
- `shuffle_along_seq()` - Random shuffle (sequence)

### Pointwise Operations
- `abs()` - Absolute value
- `clip()` - Clip values to range
- `exp()` - Exponential (base e)
- `exp2()` - Exponential (base 2)
- `expm1()` - exp(x) - 1
- `log()` - Natural logarithm
- `log1p()` - log(1 + x)
- `log2()` - Base-2 logarithm
- `log10()` - Base-10 logarithm

### Reduction Operations
- `amax_along_batch()` / `max_along_batch()` - Maximum (batch)
- `amax_along_seq()` / `max_along_seq()` - Maximum (sequence)
- `amin_along_batch()` / `min_along_batch()` - Minimum (batch)
- `amin_along_seq()` / `min_along_seq()` - Minimum (sequence)
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
- `sum_along_batch()` - Sum (batch)
- `sum_along_seq()` - Sum (sequence)

### Slicing Operations
- `chunk_along_batch()` - Split into chunks (batch)
- `chunk_along_seq()` - Split into chunks (sequence)
- `select_along_batch()` - Select single index (batch)
- `select_along_seq()` - Select single index (sequence)
- `slice_along_batch()` - Slice range (batch)
- `slice_along_seq()` - Slice range (sequence)
- `split_along_batch()` - Split into sections (batch)
- `split_along_seq()` - Split into sections (sequence)

### Trigonometric Functions
- `sin()` - Sine
- `cos()` - Cosine
- `tan()` - Tangent
- `arcsin()` - Inverse sine
- `arccos()` - Inverse cosine
- `arctan()` - Inverse tangent
- `sinh()` - Hyperbolic sine
- `cosh()` - Hyperbolic cosine
- `tanh()` - Hyperbolic tangent
- `arcsinh()` - Inverse hyperbolic sine
- `arccosh()` - Inverse hyperbolic cosine
- `arctanh()` - Inverse hyperbolic tangent

For detailed API documentation, see the [nested API reference](../refs/nested.md).
