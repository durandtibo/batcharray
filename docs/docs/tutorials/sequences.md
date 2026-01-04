# Tutorial: Working with Sequences

This tutorial covers how to work with sequences (time-series data) using `batcharray`.

## Introduction

Sequences are ordered collections of data points, commonly found in:
- Time-series data (stock prices, sensor readings)
- Natural language processing (sentences, documents)
- Video processing (frames over time)
- Audio processing (audio samples)

In `batcharray`, sequences have two important dimensions:
- **Batch dimension** (axis 0): Different sequences
- **Sequence dimension** (axis 1): Time steps within each sequence

## Basic Sequence Operations

### Creating Sequence Data

```python
import numpy as np
from batcharray import array

# Create a batch of 3 sequences, each with 4 time steps and 2 features
sequences = np.array(
    [
        # Sequence 1
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],  # t=0  # t=1  # t=2  # t=3
        # Sequence 2
        [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
        # Sequence 3
        [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0]],
    ]
)

print(f"Shape: {sequences.shape}")  # (3, 4, 2)
print(f"Batch size: {sequences.shape[0]}")  # 3
print(f"Sequence length: {sequences.shape[1]}")  # 4
print(f"Features: {sequences.shape[2]}")  # 2
```

### Slicing Sequences

You can slice along the sequence dimension to extract time windows:

```python
from batcharray import array

# Get first 2 time steps for all sequences
first_steps = array.slice_along_seq(sequences, stop=2)
print(first_steps.shape)  # (3, 2, 2)
print(first_steps)
# [[[ 1.  2.]
#   [ 3.  4.]]
#
#  [[ 9. 10.]
#   [11. 12.]]
#
#  [[17. 18.]
#   [19. 20.]]]

# Get time steps 1-3 (middle portion)
middle_steps = array.slice_along_seq(sequences, start=1, stop=3)
print(middle_steps.shape)  # (3, 2, 2)

# Get last time step
last_step = array.slice_along_seq(sequences, start=3)
print(last_step.shape)  # (3, 1, 2)
```

### Selecting Specific Time Steps

```python
from batcharray import array

# Select specific time steps
time_indices = np.array([0, 2])  # Get t=0 and t=2
selected = array.index_select_along_seq(sequences, indices=time_indices)
print(selected.shape)  # (3, 2, 2)
```

### Splitting Sequences

```python
from batcharray import array

# Split sequence into chunks
chunks = array.chunk_along_seq(sequences, chunks=2)
print(f"Number of chunks: {len(chunks)}")  # 2
print(f"First chunk shape: {chunks[0].shape}")  # (3, 2, 2)
print(f"Second chunk shape: {chunks[1].shape}")  # (3, 2, 2)

# Split at specific points
splits = array.split_along_seq(sequences, split_size_or_sections=[1, 2, 1])
print(f"Number of splits: {len(splits)}")
for i, split in enumerate(splits):
    print(f"Split {i} shape: {split.shape}")
```

## Sequence Statistics

### Computing Statistics Over Time

```python
import numpy as np
from batcharray import array

sequences = np.array(
    [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]
)
# Shape: (2, 3, 2) - 2 sequences, 3 time steps, 2 features

# Mean over time for each sequence
mean_over_time = array.mean_along_seq(sequences)
print(mean_over_time)
# [[ 3.  4.]    # Sequence 1: mean of all time steps
#  [ 9. 10.]]   # Sequence 2: mean of all time steps
print(mean_over_time.shape)  # (2, 2)

# Maximum over time
max_over_time = array.amax_along_seq(sequences)
print(max_over_time)
# [[ 5.  6.]    # Sequence 1: max across time
#  [11. 12.]]   # Sequence 2: max across time

# Sum over time
sum_over_time = array.sum_along_seq(sequences)
print(sum_over_time)
# [[ 9. 12.]    # Sequence 1: sum across time
#  [27. 30.]]   # Sequence 2: sum across time
```

### Finding Extremes in Sequences

```python
import numpy as np
from batcharray import array

sequences = np.array(
    [[[0.1, 0.2], [0.5, 0.3], [0.2, 0.8]], [[0.3, 0.4], [0.1, 0.9], [0.7, 0.2]]]
)

# Find time step with maximum value for each feature
max_indices = array.argmax_along_seq(sequences)
print(max_indices)
# [[1 2]    # Sequence 1: max at t=1 for feat 0, t=2 for feat 1
#  [2 1]]   # Sequence 2: max at t=2 for feat 0, t=1 for feat 1

# Find actual maximum values
max_values = array.amax_along_seq(sequences)
print(max_values)
# [[0.5 0.8]
#  [0.7 0.9]]
```

## Cumulative Operations

Cumulative operations are particularly useful for sequences:

```python
import numpy as np
from batcharray import array

sequences = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])

# Cumulative sum over time
cumsum = array.cumsum_along_seq(sequences)
print(cumsum)
# [[[ 1  2]
#   [ 4  6]   # 1+3, 2+4
#   [ 9 12]]  # 1+3+5, 2+4+6
#
#  [[ 7  8]
#   [16 18]   # 7+9, 8+10
#   [27 30]]] # 7+9+11, 8+10+12

# Cumulative product over time
cumprod = array.cumprod_along_seq(sequences)
print(cumprod)
# [[[ 1   2]
#   [ 3   8]   # 1*3, 2*4
#   [15  48]]  # 1*3*5, 2*4*6
#
#  [[ 7   8]
#   [63  80]   # 7*9, 8*10
#   [693 880]]] # 7*9*11, 8*10*12
```

## Working with Nested Sequences

When you have multiple related sequences, use the `nested` module:

```python
import numpy as np
from batcharray import nested

# Batch of sequences with inputs and targets
sequences = {
    "inputs": np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]),
    "targets": np.array([[[0], [1], [0]], [[1], [1], [0]]]),
    "masks": np.array([[True, True, False], [True, True, True]]),
}

# Slice all sequences to first 2 time steps
sliced = nested.slice_along_seq(sequences, stop=2)
print(sliced["inputs"].shape)  # (2, 2, 2)
print(sliced["targets"].shape)  # (2, 2, 1)
print(sliced["masks"].shape)  # (2, 2)
```

## Sorting Sequences

### Sorting by Time

```python
import numpy as np
from batcharray import array

sequences = np.array([[[5, 2], [1, 4], [3, 6]], [[8, 7], [9, 5], [6, 8]]])

# Sort along sequence dimension
sorted_seq = array.sort_along_seq(sequences)
print(sorted_seq)
# [[[1 2]   # Sorted by time
#   [3 4]
#   [5 6]]
#
#  [[6 5]
#   [8 7]
#   [9 8]]]
```

## Tiling and Repeating Sequences

```python
import numpy as np
from batcharray import array

sequences = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# Shape: (2, 2, 2)

# Repeat sequence 3 times
tiled = array.tile_along_seq(sequences, reps=3)
print(tiled.shape)  # (2, 6, 2)
print(tiled[0])
# [[1 2]
#  [3 4]
#  [1 2]
#  [3 4]
#  [1 2]
#  [3 4]]
```

## Combining Sequence Operations

You can combine batch and sequence operations:

```python
import numpy as np
from batcharray import array

# Create sequences: 10 sequences, 20 time steps, 5 features
sequences = np.random.randn(10, 20, 5)

# Select specific sequences (batch operation)
selected_sequences = array.index_select_along_batch(
    sequences, indices=np.array([0, 2, 4, 6, 8])
)
print(selected_sequences.shape)  # (5, 20, 5)

# Then slice time window (sequence operation)
time_window = array.slice_along_seq(selected_sequences, start=5, stop=15)
print(time_window.shape)  # (5, 10, 5)

# Compute statistics over time
time_mean = array.mean_along_seq(time_window)
print(time_mean.shape)  # (5, 5) - mean for each sequence
```

## Variable-Length Sequences

Real-world sequences often have different lengths. Use masked arrays:

```python
import numpy as np
import numpy.ma as ma
from batcharray import array

# Create sequences with different lengths (padded and masked)
# Actual lengths: [4, 3, 2]
sequences = ma.array(
    [
        [[1, 2], [3, 4], [5, 6], [7, 8]],  # Full sequence
        [[9, 10], [11, 12], [13, 14], [0, 0]],  # Length 3 (last padded)
        [[15, 16], [17, 18], [0, 0], [0, 0]],  # Length 2 (last 2 padded)
    ],
    mask=[
        [[0, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [1, 1]],
        [[0, 0], [0, 0], [1, 1], [1, 1]],
    ],
)

# Compute mean (automatically handles variable lengths)
mean_over_time = array.mean_along_seq(sequences)
print(mean_over_time)
# [[ 4.  5.]     # Mean of all 4 time steps
#  [11. 12.]     # Mean of first 3 time steps only
#  [16. 17.]]    # Mean of first 2 time steps only
```

## Common Use Cases

### Sliding Window Analysis

```python
import numpy as np
from batcharray import array

# Time series data
time_series = np.random.randn(1, 100, 3)  # 1 long sequence

# Create sliding windows of size 10
window_size = 10
stride = 5
windows = []

for i in range(0, 100 - window_size + 1, stride):
    window = array.slice_along_seq(time_series, start=i, stop=i + window_size)
    windows.append(window)

# Stack windows into batch
windowed_batch = np.concatenate(windows, axis=0)
print(windowed_batch.shape)  # (num_windows, 10, 3)
```

### Sequence Truncation

```python
import numpy as np
from batcharray import nested

# Truncate all sequences to same length
sequences = {
    "text": np.random.randint(0, 1000, (32, 50)),  # 32 sequences, max 50 tokens
    "labels": np.random.randint(0, 2, (32,)),
}

# Truncate to 30 tokens
max_length = 30
truncated = nested.slice_along_seq(sequences, stop=max_length)
print(truncated["text"].shape)  # (32, 30)
```

### Time-Series Forecasting Preparation

```python
import numpy as np
from batcharray import array

# Historical data: (batch, time, features)
history = np.random.randn(100, 24, 5)  # 24 hours of history

# Use first 18 hours for input, predict last 6 hours
input_window = array.slice_along_seq(history, stop=18)
target_window = array.slice_along_seq(history, start=18)

print(f"Input shape: {input_window.shape}")  # (100, 18, 5)
print(f"Target shape: {target_window.shape}")  # (100, 6, 5)
```

### Sequence Aggregation

```python
import numpy as np
from batcharray import array

sequences = np.array(
    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]
)

# Different aggregations over time
mean_rep = array.mean_along_seq(sequences)  # Average over time
max_rep = array.amax_along_seq(sequences)  # Max pooling over time
sum_rep = array.sum_along_seq(sequences)  # Sum over time

print(f"Mean: {mean_rep.shape}")  # (2, 3) - one per sequence
print(f"Max: {max_rep.shape}")  # (2, 3)
print(f"Sum: {sum_rep.shape}")  # (2, 3)
```

## Next Steps

- Learn about [batch operations](batches.md) for working with collections
- Explore [nested structures](advanced_nested.md) for complex data
- See the [array reference](../refs/array.md) for all available functions
- Check out the [FAQ](../faq.md) for common questions
