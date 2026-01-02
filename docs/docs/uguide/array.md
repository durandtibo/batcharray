# Array Operations

The `batcharray.array` module provides functions to manipulate NumPy arrays representing batches of data and sequences.

## Overview

`batcharray` provides two main categories of array operations:

1. **Batch operations**: Work on arrays with shape `(batch_size, ...)` where the first axis is the batch dimension
2. **Sequence operations**: Work on arrays with shape `(batch_size, seq_len, ...)` where the first axis is batch and second is sequence

## Batch Operations

Batch operations work along the first axis (batch dimension) of arrays. All functions that operate on the batch dimension have the suffix `_along_batch`.

### Slicing and Indexing

```python
import numpy as np
from batcharray import array

# Create a batch of data
batch = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# Slice the batch
sliced = array.slice_along_batch(batch, start=1, stop=3)
# Result: [[4, 5, 6], [7, 8, 9]]

# Select specific indices
selected = array.index_select_along_batch(batch, indices=np.array([0, 2]))
# Result: [[1, 2, 3], [7, 8, 9]]

# Split into chunks
chunks = array.chunk_along_batch(batch, chunks=2)
# Result: [array([[1, 2, 3], [4, 5, 6]]), array([[7, 8, 9], [10, 11, 12]])]
```

### Reductions

```python
import numpy as np
from batcharray import array

batch = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Compute statistics along batch
mean_val = array.mean_along_batch(batch)  # [3.0, 4.0]
max_val = array.amax_along_batch(batch)  # [5.0, 6.0]
sum_val = array.sum_along_batch(batch)  # [9.0, 12.0]
```

### Sorting and Permutation

```python
import numpy as np
from batcharray import array

batch = np.array([[5, 2], [1, 4], [3, 6]])

# Sort along batch
sorted_batch = array.sort_along_batch(batch)
# Result: [[1, 2], [3, 4], [5, 6]]

# Get sort indices
indices = array.argsort_along_batch(batch)

# Shuffle batch randomly
shuffled = array.shuffle_along_batch(batch)
```

## Sequence Operations

Sequence operations work along the second axis (sequence dimension). All functions that operate on sequences have the suffix `_along_seq`.

### Sequence Manipulation

```python
import numpy as np
from batcharray import array

# Batch of sequences: (batch_size=2, seq_len=4, features=3)
sequences = np.array(
    [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]],
    ]
)

# Slice sequences
sliced = array.slice_along_seq(sequences, start=1, stop=3)
# Shape: (2, 2, 3)

# Tile sequences
tiled = array.tile_along_seq(sequences, reps=2)
# Repeats each sequence twice along the sequence dimension

# Split sequences
chunks = array.split_along_seq(sequences, split_size_or_sections=2)
```

### Sequence Reductions

```python
import numpy as np
from batcharray import array

sequences = np.array(
    [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]
)

# Compute mean for each batch across sequence
mean_seq = array.mean_along_seq(sequences)
# Shape: (2, 2) - one value per batch per feature

# Cumulative sum along sequences
cumsum = array.cumsum_along_seq(sequences)
```

## Combining Operations

You can chain operations for complex transformations:

```python
import numpy as np
from batcharray import array

# Create batch of sequences
data = np.random.randn(10, 20, 5)  # 10 batches, 20 timesteps, 5 features

# Select specific batch items
data = array.index_select_along_batch(data, indices=np.array([0, 2, 4, 6, 8]))

# Slice sequences to first 10 timesteps
data = array.slice_along_seq(data, stop=10)

# Compute statistics
mean = array.mean_along_seq(data)  # Average over time for each batch
```

## Working with Masked Arrays

All functions in `batcharray.array` support NumPy masked arrays, allowing you to handle missing or invalid data:

```python
import numpy as np
import numpy.ma as ma
from batcharray import array

# Create masked array
data = ma.array(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mask=[[0, 0, 1], [0, 1, 0], [1, 0, 0]]
)

# Operations work with masked values
mean = array.mean_along_batch(data)
sorted_data = array.sort_along_batch(data)
```

## Complete Function Reference

The `array` module provides the following categories of functions:

### Comparison and Sorting
- `argsort_along_batch()` - Get indices that would sort along batch
- `argsort_along_seq()` - Get indices that would sort along sequence
- `sort_along_batch()` - Sort along batch dimension
- `sort_along_seq()` - Sort along sequence dimension

### Indexing and Selection
- `index_select_along_batch()` - Select using integer indices (batch)
- `index_select_along_seq()` - Select using integer indices (sequence)
- `masked_select_along_batch()` - Select using boolean mask (batch)
- `masked_select_along_seq()` - Select using boolean mask (sequence)
- `take_along_batch()` - Take elements using index array (batch)
- `take_along_seq()` - Take elements using index array (sequence)

### Joining and Combining
- `concatenate_along_batch()` - Concatenate arrays along batch
- `concatenate_along_seq()` - Concatenate arrays along sequence
- `tile_along_seq()` - Repeat array along sequence dimension

### Mathematical Operations
- `cumprod_along_batch()` - Cumulative product along batch
- `cumprod_along_seq()` - Cumulative product along sequence
- `cumsum_along_batch()` - Cumulative sum along batch
- `cumsum_along_seq()` - Cumulative sum along sequence

### Permutation and Shuffling
- `permute_along_batch()` - Apply permutation to batch
- `permute_along_seq()` - Apply permutation to sequence
- `shuffle_along_batch()` - Random shuffle along batch
- `shuffle_along_seq()` - Random shuffle along sequence

### Reduction Operations
- `amax_along_batch()` / `max_along_batch()` - Maximum along batch
- `amax_along_seq()` / `max_along_seq()` - Maximum along sequence
- `amin_along_batch()` / `min_along_batch()` - Minimum along batch
- `amin_along_seq()` / `min_along_seq()` - Minimum along sequence
- `argmax_along_batch()` - Indices of maximum along batch
- `argmax_along_seq()` - Indices of maximum along sequence
- `argmin_along_batch()` - Indices of minimum along batch
- `argmin_along_seq()` - Indices of minimum along sequence
- `mean_along_batch()` - Mean along batch
- `mean_along_seq()` - Mean along sequence
- `median_along_batch()` - Median along batch
- `median_along_seq()` - Median along sequence
- `prod_along_batch()` - Product along batch
- `prod_along_seq()` - Product along sequence
- `sum_along_batch()` - Sum along batch
- `sum_along_seq()` - Sum along sequence

### Slicing Operations
- `chunk_along_batch()` - Split into equal chunks (batch)
- `chunk_along_seq()` - Split into equal chunks (sequence)
- `select_along_batch()` - Select single index (batch)
- `select_along_seq()` - Select single index (sequence)
- `slice_along_batch()` - Slice range of indices (batch)
- `slice_along_seq()` - Slice range of indices (sequence)
- `split_along_batch()` - Split into specified sections (batch)
- `split_along_seq()` - Split into specified sections (sequence)

For detailed API documentation of each function, see the [array API reference](../refs/array.md).
