# Tutorial: Working with Batches

This tutorial will guide you through the basics of working with batches of data using `batcharray`.

## Introduction

In machine learning and data processing, we often work with batches of data - collections of samples processed together. `batcharray` provides convenient utilities to manipulate these batches, whether they're single arrays or complex nested structures.

## Basic Batch Operations

### Creating a Batch

Let's start by creating a simple batch of data:

```python
import numpy as np
from batcharray import array

# Create a batch of 5 samples, each with 3 features
batch = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0]
])

print(f"Batch shape: {batch.shape}")  # (5, 3)
print(f"Number of samples: {batch.shape[0]}")
print(f"Features per sample: {batch.shape[1]}")
```

### Slicing Batches

You can extract a subset of samples from a batch:

```python
from batcharray import array

# Get first 3 samples
first_three = array.slice_along_batch(batch, stop=3)
print(first_three)
# [[1. 2. 3.]
#  [4. 5. 6.]
#  [7. 8. 9.]]

# Get samples 2-4 (indices 1, 2, 3)
middle_samples = array.slice_along_batch(batch, start=1, stop=4)
print(middle_samples)
# [[ 4.  5.  6.]
#  [ 7.  8.  9.]
#  [10. 11. 12.]]

# Get last 2 samples
last_two = array.slice_along_batch(batch, start=3)
print(last_two)
# [[10. 11. 12.]
#  [13. 14. 15.]]
```

### Selecting Specific Samples

Use `index_select_along_batch` to select specific samples by index:

```python
from batcharray import array

# Select samples at indices 0, 2, and 4
indices = np.array([0, 2, 4])
selected = array.index_select_along_batch(batch, indices=indices)
print(selected)
# [[ 1.  2.  3.]
#  [ 7.  8.  9.]
#  [13. 14. 15.]]
```

### Splitting Batches

Split a batch into multiple smaller batches:

```python
from batcharray import array

# Split into batches of size 2
chunks = array.chunk_along_batch(batch, chunks=3)
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i} shape: {chunk.shape}")
# Chunk 0 shape: (2, 3)
# Chunk 1 shape: (2, 3)
# Chunk 2 shape: (1, 3)

# Split at specific sizes
splits = array.split_along_batch(batch, split_size_or_sections=[2, 2, 1])
print(f"Number of splits: {len(splits)}")
for i, split in enumerate(splits):
    print(f"Split {i} shape: {split.shape}")
```

## Working with Nested Batches

Real-world data often comes in nested structures - dictionaries with multiple arrays, lists of arrays, etc.

### Dictionary Batches

```python
import numpy as np
from batcharray import nested

# Create a batch as a dictionary
batch = {
    "features": np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]),
    "labels": np.array([0, 1, 0]),
    "weights": np.array([1.0, 0.8, 1.2])
}

# Slice all arrays together
train_batch = nested.slice_along_batch(batch, stop=2)
print(train_batch)
# {
#     'features': array([[1., 2., 3.],
#                        [4., 5., 6.]]),
#     'labels': array([0, 1]),
#     'weights': array([1. , 0.8])
# }

# Split into train/validation
splits = nested.split_along_batch(batch, split_size_or_sections=[2, 1])
train, val = splits[0], splits[1]
print(f"Train samples: {train['features'].shape[0]}")
print(f"Val samples: {val['features'].shape[0]}")
```

### Maintaining Consistency

The key advantage of `nested` operations is that they maintain consistency across all arrays:

```python
import numpy as np
from batcharray import nested

# Shuffle while keeping features and labels aligned
batch = {
    "features": np.array([[1, 2], [3, 4], [5, 6]]),
    "labels": np.array([0, 1, 0])
}

shuffled = nested.shuffle_along_batch(batch)
# Features and labels are shuffled with the same permutation
```

## Computing Statistics

### Batch-level Statistics

Compute statistics across samples in a batch:

```python
import numpy as np
from batcharray import array

data = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])

# Mean across samples (for each feature)
mean_features = array.mean_along_batch(data)
print(mean_features)  # [4. 5. 6.]

# Maximum value for each feature
max_features = array.amax_along_batch(data)
print(max_features)  # [7. 8. 9.]

# Sum across samples
sum_features = array.sum_along_batch(data)
print(sum_features)  # [12. 15. 18.]
```

### Finding Extremes

```python
import numpy as np
from batcharray import array

scores = np.array([
    [0.2, 0.5, 0.3],
    [0.1, 0.8, 0.1],
    [0.6, 0.2, 0.2]
])

# Index of maximum value for each feature
max_indices = array.argmax_along_batch(scores)
print(max_indices)  # [2, 1, 0]

# Actual maximum values
max_values = array.amax_along_batch(scores)
print(max_values)  # [0.6, 0.8, 0.3]
```

## Sorting and Ordering

### Sorting Batches

```python
import numpy as np
from batcharray import array

# Unsorted batch
data = np.array([
    [5, 2],
    [1, 4],
    [3, 6]
])

# Sort along batch dimension
sorted_data = array.sort_along_batch(data)
print(sorted_data)
# [[1 2]
#  [3 4]
#  [5 6]]

# Get sorting indices
sort_indices = array.argsort_along_batch(data)
print(sort_indices)
# [[1 0]
#  [2 1]
#  [0 2]]
```

### Random Shuffling

```python
import numpy as np
from batcharray import array

data = np.array([[1, 2], [3, 4], [5, 6]])

# Random shuffle
shuffled = array.shuffle_along_batch(data)
print(shuffled)
# Order is randomized, e.g.:
# [[5 6]
#  [1 2]
#  [3 4]]
```

## Combining Batches

### Concatenation

```python
import numpy as np
from batcharray import array

batch1 = np.array([[1, 2], [3, 4]])
batch2 = np.array([[5, 6], [7, 8]])

# Combine batches
combined = array.concatenate_along_batch([batch1, batch2])
print(combined)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]
```

### Nested Concatenation

```python
import numpy as np
from batcharray import nested

batch1 = {
    "features": np.array([[1, 2], [3, 4]]),
    "labels": np.array([0, 1])
}

batch2 = {
    "features": np.array([[5, 6]]),
    "labels": np.array([0])
}

combined = nested.concatenate_along_batch([batch1, batch2])
print(combined)
# {
#     'features': array([[1, 2],
#                        [3, 4],
#                        [5, 6]]),
#     'labels': array([0, 1, 0])
# }
```

## Working with Missing Data

NumPy masked arrays allow you to handle missing or invalid data:

```python
import numpy as np
import numpy.ma as ma
from batcharray import array

# Create data with missing values (marked as masked)
data = ma.array(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0],
     [7.0, 8.0, 9.0]],
    mask=[[False, True, False],   # 2nd value missing
          [False, False, True],   # 3rd value missing
          [True, False, False]]   # 1st value missing
)

# Compute mean (ignoring masked values)
mean_vals = array.mean_along_batch(data)
print(mean_vals)  # [2.5, 5.0, 4.5]

# Sort (masked values handled appropriately)
sorted_data = array.sort_along_batch(data)
print(sorted_data)
```

## Next Steps

- Learn about [sequence operations](sequences.md) for time-series data
- Explore [advanced nested operations](advanced_nested.md)
- See [computation models](../uguide/computation.md) for low-level operations
- Check the [FAQ](../faq.md) for common questions

## Common Patterns

### Train/Test Split

```python
import numpy as np
from batcharray import nested

# Full dataset
dataset = {
    "X": np.random.randn(1000, 784),  # MNIST-like
    "y": np.random.randint(0, 10, 1000)
}

# 80/20 split
train_size = int(0.8 * 1000)
train_data = nested.slice_along_batch(dataset, stop=train_size)
test_data = nested.slice_along_batch(dataset, start=train_size)

print(f"Train samples: {train_data['X'].shape[0]}")  # 800
print(f"Test samples: {test_data['X'].shape[0]}")    # 200
```

### Mini-batch Processing

```python
import numpy as np
from batcharray import nested

# Large dataset
dataset = {
    "X": np.random.randn(1000, 10),
    "y": np.random.randint(0, 2, 1000)
}

# Process in mini-batches
batch_size = 32
num_batches = (1000 + batch_size - 1) // batch_size

for i in range(num_batches):
    start = i * batch_size
    stop = min((i + 1) * batch_size, 1000)
    mini_batch = nested.slice_along_batch(dataset, start=start, stop=stop)
    
    # Process mini_batch
    print(f"Processing batch {i+1}/{num_batches} with {mini_batch['X'].shape[0]} samples")
```

### Data Augmentation

```python
import numpy as np
from batcharray import nested

batch = {
    "images": np.random.randn(32, 28, 28),
    "labels": np.random.randint(0, 10, 32)
}

# Shuffle for augmentation
augmented = nested.shuffle_along_batch(batch)

# Select random subset
indices = np.random.choice(32, size=16, replace=False)
subset = nested.index_select_along_batch(augmented, indices=indices)
```
