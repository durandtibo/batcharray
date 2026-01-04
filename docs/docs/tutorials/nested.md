# Tutorial: Advanced Nested Operations

This tutorial covers advanced techniques for working with complex nested data structures in
`batcharray`.

## Introduction

Real-world data is often organized in complex nested structures:

- Multiple related arrays in dictionaries
- Lists or tuples of arrays
- Deeply nested combinations of both

The `batcharray.nested` module provides powerful tools to work with these structures efficiently.

## Understanding Nested Structures

### What Are Nested Structures?

Nested structures are hierarchical data organizations containing NumPy arrays:

```python
import numpy as np

# Simple dictionary
simple_nested = {"features": np.array([[1, 2], [3, 4]]), "labels": np.array([0, 1])}

# List of arrays
list_nested = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]

# Complex nesting
complex_nested = {
    "train": {"inputs": np.array([[1, 2], [3, 4]]), "targets": np.array([0, 1])},
    "metadata": {"weights": np.array([1.0, 0.8]), "ids": np.array(["a", "b"])},
}
```

### Why Use Nested Operations?

Nested operations ensure **consistency** across related arrays:

```python
import numpy as np
from batcharray import nested

# Without nested operations - INCONSISTENT!
data = {"features": np.array([[1, 2], [3, 4], [5, 6]]), "labels": np.array([0, 1, 0])}

# Manually slicing - easy to make mistakes
sliced_features = data["features"][:2]
sliced_labels = data["labels"][:3]  # Oops! Different size!

# With nested operations - CONSISTENT!
sliced_data = nested.slice_along_batch(data, stop=2)
# Both features and labels are sliced to 2 items
```

## Mathematical Operations on Nested Structures

### Element-wise Operations

Apply mathematical functions to all arrays in a structure:

```python
import numpy as np
from batcharray import nested

data = {
    "values": np.array([[-2.0, 3.0], [1.0, -4.0]]),
    "scores": np.array([[0.5, -0.5], [0.2, -0.8]]),
}

# Absolute values
abs_data = nested.abs(data)
# {
#     'values': array([[2., 3.],
#                      [1., 4.]]),
#     'scores': array([[0.5, 0.5],
#                      [0.2, 0.8]])
# }

# Exponential
exp_data = nested.exp(data)

# Logarithm (of absolute values to avoid errors)
log_data = nested.log(nested.abs(data))

# Clipping
clipped_data = nested.clip(data, a_min=-1.0, a_max=1.0)
# {
#     'values': array([[-1.,  1.],
#                      [ 1., -1.]]),
#     'scores': array([[ 0.5, -0.5],
#                      [ 0.2, -0.8]])
# }
```

### Exponential and Logarithmic Functions

```python
import numpy as np
from batcharray import nested

# Use positive values to avoid log(0) errors
data = {
    "x": np.array([[0.1, 1.0], [2.0, 3.0]]),
    "y": np.array([[0.5, 1.5], [2.5, 3.5]]),
}

# Standard exponential
exp_result = nested.exp(data)

# Base-2 exponential
exp2_result = nested.exp2(data)

# exp(x) - 1 (more accurate for small values)
expm1_result = nested.expm1(data)

# Natural logarithm
log_result = nested.log(data)

# Base-2 logarithm
log2_result = nested.log2(data)

# Base-10 logarithm
log10_result = nested.log10(data)

# log(1 + x) (more accurate for small values)
log1p_result = nested.log1p(data)
```

### Trigonometric Functions

```python
import numpy as np
from batcharray import nested

# Angles in radians
angles = {
    "theta": np.array([[0.0, np.pi / 4], [np.pi / 2, np.pi]]),
    "phi": np.array([[0.0, np.pi / 6], [np.pi / 3, np.pi / 2]]),
}

# Sine
sin_result = nested.sin(angles)
# [[0.        0.70710678]
#  [1.        0.        ]]

# Cosine
cos_result = nested.cos(angles)

# Tangent
tan_result = nested.tan(angles)

# Hyperbolic functions
sinh_result = nested.sinh(angles)
cosh_result = nested.cosh(angles)
tanh_result = nested.tanh(angles)

# Inverse trigonometric functions (arcsin, arccos require values in [-1, 1])
values_trig = {
    "x": np.array([[0.0, 0.5], [0.7, 1.0]]),
    "y": np.array([[0.0, 0.3], [0.6, 0.9]]),
}

arcsin_result = nested.arcsin(values_trig)
arccos_result = nested.arccos(values_trig)
arctan_result = nested.arctan(values_trig)

# Inverse hyperbolic functions
# arcsinh works for all values
arcsinh_result = nested.arcsinh(values_trig)

# arccosh requires values >= 1
values_cosh = {
    "x": np.array([[1.0, 1.5], [2.0, 3.0]]),
    "y": np.array([[1.0, 1.2], [1.8, 2.5]]),
}
arccosh_result = nested.arccosh(values_cosh)

# arctanh requires -1 < values < 1 (strictly between, not including -1 and 1)
values_tanh = {
    "x": np.array([[0.0, 0.5], [0.7, 0.9]]),
    "y": np.array([[0.0, 0.3], [0.6, 0.8]]),
}
arctanh_result = nested.arctanh(values_tanh)
```

## Advanced Indexing and Selection

### Masked Selection

Select elements based on boolean masks:

```python
import numpy as np
from batcharray import nested

data = {
    "features": np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
    "scores": np.array([0.1, 0.9, 0.3, 0.8]),
    "ids": np.array([10, 20, 30, 40]),
}

# Select high-scoring items (scores > 0.5)
mask = data["scores"] > 0.5
filtered = nested.masked_select_along_batch(data, mask=mask)
# {
#     'features': array([[3, 4],
#                        [7, 8]]),
#     'scores': array([0.9, 0.8]),
#     'ids': array([20, 40])
# }
```

### Advanced Indexing

```python
import numpy as np
from batcharray import nested

data = {"A": np.array([[1, 2], [3, 4], [5, 6]]), "B": np.array([10, 20, 30])}

# Select in custom order
indices = np.array([2, 0, 2, 1])  # Can repeat indices
reordered = nested.index_select_along_batch(data, indices=indices)
# {
#     'A': array([[5, 6],
#                 [1, 2],
#                 [5, 6],
#                 [3, 4]]),
#     'B': array([30, 10, 30, 20])
# }
```

### Take Along Axis

Select elements using indices array (useful after sorting):

```python
import numpy as np
from batcharray import nested

data = {"values": np.array([[5, 2], [1, 4], [3, 6]]), "labels": np.array([0, 1, 0])}

# Get sort indices
sort_indices = nested.argsort_along_batch(data["values"])
# [[1 0]   # For column 0: indices that would sort [5,1,3] -> [1,3,5]
#  [2 1]   # For column 1: indices that would sort [2,4,6] -> [2,4,6]
#  [0 2]]

# Use those indices to sort all arrays
sorted_data = nested.take_along_batch(data, indices=sort_indices[:, 0:1])
# Takes first column of indices for all arrays
```

## Reductions and Aggregations

### Statistical Reductions

```python
import numpy as np
from batcharray import nested

data = {
    "train_scores": np.array([[0.8, 0.9], [0.7, 0.85], [0.75, 0.88]]),
    "val_scores": np.array([[0.7, 0.8], [0.65, 0.75], [0.68, 0.78]]),
}

# Mean across batches
mean_scores = nested.mean_along_batch(data)
# {
#     'train_scores': array([0.75      , 0.87666667]),
#     'val_scores': array([0.67666667, 0.77666667])
# }

# Maximum values
max_scores = nested.amax_along_batch(data)

# Minimum values
min_scores = nested.amin_along_batch(data)

# Median values
median_scores = nested.median_along_batch(data)

# Sum across batches
sum_scores = nested.sum_along_batch(data)

# Product across batches
prod_scores = nested.prod_along_batch(data)
```

### Finding Extremes

```python
import numpy as np
from batcharray import nested

data = {
    "scores": np.array([[0.2, 0.8], [0.5, 0.6], [0.9, 0.3]]),
    "metrics": np.array([[1.0, 2.0], [1.5, 1.8], [0.8, 2.2]]),
}

# Indices of maximum values
argmax_data = nested.argmax_along_batch(data)
# {
#     'scores': array([2, 0]),    # Max scores: col 0 at idx 2, col 1 at idx 0
#     'metrics': array([1, 2])    # Max metrics: col 0 at idx 1, col 1 at idx 2
# }

# Indices of minimum values
argmin_data = nested.argmin_along_batch(data)
```

### Sequence-wise Reductions

```python
import numpy as np
from batcharray import nested

sequences = {
    "inputs": np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]),
    "outputs": np.array(
        [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]]
    ),
}

# Mean over sequence dimension
mean_over_time = nested.mean_along_seq(sequences)
# {
#     'inputs': array([[ 3.,  4.],    # Mean of [[1,2], [3,4], [5,6]]
#                      [ 9., 10.]]),  # Mean of [[7,8], [9,10], [11,12]]
#     'outputs': array([[0.3, 0.4],
#                       [0.9, 1. ]])
# }

# Sum over sequences
sum_over_time = nested.sum_along_seq(sequences)

# Max over sequences
max_over_time = nested.amax_along_seq(sequences)
```

## Cumulative Operations

### Cumulative Sum and Product

```python
import numpy as np
from batcharray import nested

data = {
    "sales": np.array([[10, 20], [15, 25], [12, 18]]),
    "costs": np.array([[5, 10], [8, 12], [6, 9]]),
}

# Cumulative sum along batches (running totals)
cumsum_batch = nested.cumsum_along_batch(data)
# {
#     'sales': array([[10, 20],
#                     [25, 45],   # 10+15, 20+25
#                     [37, 63]]), # 10+15+12, 20+25+18
#     'costs': array([[ 5, 10],
#                     [13, 22],   # 5+8, 10+12
#                     [19, 31]])  # 5+8+6, 10+12+9
# }

# Cumulative product
cumprod_batch = nested.cumprod_along_batch(data)
```

### Sequence Cumulative Operations

```python
import numpy as np
from batcharray import nested

sequences = {"values": np.array([[[1, 2], [3, 4], [5, 6]], [[2, 3], [4, 5], [6, 7]]])}

# Cumulative sum over time
cumsum_seq = nested.cumsum_along_seq(sequences)
# {
#     'values': array([[[ 1,  2],
#                       [ 4,  6],   # 1+3, 2+4
#                       [ 9, 12]],  # 1+3+5, 2+4+6
#
#                      [[ 2,  3],
#                       [ 6,  8],   # 2+4, 3+5
#                       [12, 15]]]) # 2+4+6, 3+5+7
# }

# Cumulative product over time
cumprod_seq = nested.cumprod_along_seq(sequences)
```

## Combining and Joining

### Concatenation

```python
import numpy as np
from batcharray import nested

batch1 = {
    "features": np.array([[1, 2], [3, 4]]),
    "labels": np.array([0, 1]),
    "weights": np.array([1.0, 0.8]),
}

batch2 = {
    "features": np.array([[5, 6]]),
    "labels": np.array([0]),
    "weights": np.array([1.2]),
}

batch3 = {
    "features": np.array([[7, 8], [9, 10]]),
    "labels": np.array([1, 0]),
    "weights": np.array([0.9, 1.1]),
}

# Concatenate multiple batches
combined = nested.concatenate_along_batch([batch1, batch2, batch3])
# {
#     'features': array([[1, 2],
#                        [3, 4],
#                        [5, 6],
#                        [7, 8],
#                        [9, 10]]),
#     'labels': array([0, 1, 0, 1, 0]),
#     'weights': array([1. , 0.8, 1.2, 0.9, 1.1])
# }
```

### Sequence Concatenation

```python
import numpy as np
from batcharray import nested

seq1 = {"tokens": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])}

seq2 = {"tokens": np.array([[[9, 10]], [[11, 12]]])}

# Concatenate along sequence dimension
combined_seq = nested.concatenate_along_seq([seq1, seq2])
# (2, 3, 2)
```

### Tiling

```python
import numpy as np
from batcharray import nested

sequences = {
    "pattern": np.array([[[1, 2]], [[3, 4]]]),
    "mask": np.array([[True], [False]]),
}

# Repeat sequence 4 times
tiled = nested.tile_along_seq(sequences, reps=4)
```

## Shuffling and Permutations

### Random Shuffling

```python
import numpy as np
from batcharray import nested

# Set seed for reproducibility
np.random.seed(42)

data = {
    "features": np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
    "labels": np.array([0, 1, 0, 1]),
    "ids": np.array(["a", "b", "c", "d"]),
}

# Shuffle all arrays with same permutation
shuffled = nested.shuffle_along_batch(data)
# All arrays shuffled identically to maintain alignment
```

### Custom Permutations

```python
import numpy as np
from batcharray import nested

data = {"X": np.array([[1, 2], [3, 4], [5, 6]]), "y": np.array([0, 1, 0])}

# Apply specific permutation
perm = np.array([2, 0, 1])  # Reverse order
permuted = nested.permute_along_batch(data, permutation=perm)
# {
#     'X': array([[5, 6],
#                 [1, 2],
#                 [3, 4]]),
#     'y': array([0, 0, 1])
# }
```

### Sequence Shuffling

```python
import numpy as np
from batcharray import nested

sequences = {"frames": np.array([[[1], [2], [3], [4]], [[5], [6], [7], [8]]])}

# Shuffle time steps (useful for certain augmentations)
shuffled_seq = nested.shuffle_along_seq(sequences)
# Time steps are shuffled within each sequence
```

## Sorting

```python
import numpy as np
from batcharray import nested

data = {
    "scores": np.array([[5, 2], [1, 4], [3, 6]]),
    "names": np.array([["e", "b"], ["a", "d"], ["c", "f"]]),
}

# Sort along batch dimension
sorted_data = nested.sort_along_batch(data)
# {
#     'scores': array([[1, 2],
#                      [3, 4],
#                      [5, 6]]),
#     'names': array([['a', 'b'],
#                     ['c', 'd'],
#                     ['e', 'f']], dtype='<U1')
# }

# Get sort indices
sort_indices = nested.argsort_along_batch(data["scores"])
```

## Converting to Native Python

```python
import numpy as np
from batcharray import nested

data = {
    "features": np.array([[1, 2], [3, 4]]),
    "labels": np.array([0, 1]),
    "nested": {"values": np.array([10.0, 20.0])},
}

# Convert to native Python lists
python_data = nested.to_list(data)
# {
#     'features': [[1, 2], [3, 4]],
#     'labels': [0, 1],
#     'nested': {
#         'values': [10.0, 20.0]
#     }
# }
```

## Complex Examples

### Time-Series Processing

```python
import numpy as np
from batcharray import nested

# Multi-variate time series
sequences = {
    "sensor_1": np.random.randn(10, 100, 5),  # 10 sequences, 100 steps
    "sensor_2": np.random.randn(10, 100, 3),
    "labels": np.random.randint(0, 2, (10, 100)),
}

# Slice to analysis window
window = nested.slice_along_seq(sequences, start=20, stop=80)

# Compute statistics over time
summary = nested.mean_along_seq(window)
# (10, 5)

# Normalize
normalized = {
    "sensor_1": (window["sensor_1"] - summary["sensor_1"][:, None, :])
    / (window["sensor_1"].std(axis=1, keepdims=True) + 1e-8),
    "sensor_2": (window["sensor_2"] - summary["sensor_2"][:, None, :])
    / (window["sensor_2"].std(axis=1, keepdims=True) + 1e-8),
    "labels": window["labels"],
}
```

## Next Steps

- Review the [nested API reference](../refs/nested.md) for all available functions
- Learn about [computation models](../uguide/computation.md) for low-level operations
- Check the [FAQ](../faq.md) for common questions and solutions
- See [batch operations tutorial](batches.md) for simpler cases
