# Best Practices

This guide outlines recommended patterns and best practices when using `batcharray`.

## General Principles

### Choose the Right Module

Use the appropriate module for your data structure:

```python
import numpy as np
from batcharray import array, nested

# ✅ GOOD: Use array module for single arrays
single_array = np.array([[1, 2], [3, 4]])
result = array.slice_along_batch(single_array, stop=1)

# ✅ GOOD: Use nested module for dictionaries/lists
data_dict = {
    "features": np.array([[1, 2], [3, 4]]),
    "labels": np.array([0, 1])
}
result = nested.slice_along_batch(data_dict, stop=1)

# ❌ BAD: Using nested for single array (unnecessary overhead)
result = nested.slice_along_batch(single_array, stop=1)
```

### Maintain Batch Dimension Consistency

Always ensure the first axis represents the batch dimension:

```python
import numpy as np
from batcharray import array

# ✅ GOOD: Batch dimension first (shape: batch, features)
batch = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3 samples, 3 features
result = array.mean_along_batch(batch)  # [4., 5., 6.]

# ❌ BAD: Features first, batch second
batch = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])  # Wrong orientation
# Would give incorrect results
```

### Use Consistent Naming

Follow clear naming conventions for dimensions:

```python
import numpy as np

# ✅ GOOD: Clear dimensional names
batch_size = 32
seq_length = 100
num_features = 10
sequences = np.random.randn(batch_size, seq_length, num_features)

# ❌ BAD: Unclear names
n = 32
m = 100
k = 10
data = np.random.randn(n, m, k)  # What do n, m, k represent?
```

## Working with Batches

### Batch Size Considerations

Choose appropriate batch sizes for your use case:

```python
import numpy as np
from batcharray import nested

# ✅ GOOD: Process in reasonable batch sizes
large_dataset = {
    "X": np.random.randn(10000, 100),
    "y": np.random.randint(0, 10, 10000)
}

batch_size = 64  # Reasonable size for most cases
num_batches = (10000 + batch_size - 1) // batch_size

for i in range(num_batches):
    start = i * batch_size
    stop = min((i + 1) * batch_size, 10000)
    mini_batch = nested.slice_along_batch(large_dataset, start=start, stop=stop)
    # Process mini_batch

# ❌ BAD: Batch size too large (memory issues)
batch_size = 10000  # Might not fit in memory

# ❌ BAD: Batch size too small (inefficient)
batch_size = 1  # Very slow processing
```

### Avoid Redundant Operations

Chain operations efficiently to avoid creating unnecessary intermediate arrays:

```python
import numpy as np
from batcharray import array

data = np.random.randn(1000, 50)

# ✅ GOOD: Single operation
selected = array.index_select_along_batch(
    data,
    indices=np.array([0, 10, 20, 30])
)

# ❌ BAD: Multiple redundant operations
temp1 = array.slice_along_batch(data, start=0, stop=31)
temp2 = array.index_select_along_batch(temp1, indices=np.array([0, 10, 20, 30]))
```

### Preserve Data Relationships

When working with related arrays, always use nested operations:

```python
import numpy as np
from batcharray import nested, array

features = np.random.randn(100, 10)
labels = np.random.randint(0, 2, 100)

# ✅ GOOD: Keep features and labels synchronized
data = {"features": features, "labels": labels}
shuffled = nested.shuffle_along_batch(data)
# Features and labels remain aligned

# ❌ BAD: Shuffle separately (breaks alignment!)
shuffled_features = array.shuffle_along_batch(features)
shuffled_labels = array.shuffle_along_batch(labels)
# Now misaligned!
```

## Working with Sequences

### Sequence Dimension Guidelines

Keep the sequence dimension as the second axis:

```python
import numpy as np
from batcharray import array

# ✅ GOOD: (batch, sequence, features)
sequences = np.random.randn(32, 100, 10)  # 32 sequences, 100 steps, 10 features
mean_over_time = array.mean_along_seq(sequences)  # Shape: (32, 10)

# ❌ BAD: Non-standard ordering
sequences = np.random.randn(10, 32, 100)  # Confusing dimension order
```

### Variable-Length Sequences

Use masked arrays for variable-length sequences:

```python
import numpy as np
import numpy.ma as ma
from batcharray import array

# ✅ GOOD: Use masked arrays for padding
max_length = 50
sequences = []
masks = []

for length in [30, 45, 20]:  # Different lengths
    seq = np.random.randn(length, 5)
    # Pad to max_length
    padded = np.zeros((max_length, 5))
    padded[:length] = seq
    sequences.append(padded)
    
    # Create mask
    mask = np.ones((max_length, 5), dtype=bool)
    mask[:length] = False
    masks.append(mask)

batch = ma.array(sequences, mask=masks)
mean_seq = array.mean_along_seq(batch)  # Correctly handles variable lengths

# ❌ BAD: Include padding in calculations
# Simple padding without masks leads to wrong statistics
```

### Temporal Operations

Be mindful of temporal order in sequences:

```python
import numpy as np
from batcharray import array

sequences = np.random.randn(10, 50, 5)

# ✅ GOOD: Preserve temporal order when needed
historical = array.slice_along_seq(sequences, stop=40)  # First 40 steps
recent = array.slice_along_seq(sequences, start=40)     # Last 10 steps

# ✅ GOOD: Shuffle time when appropriate (data augmentation)
shuffled = array.shuffle_along_seq(sequences)
# OK if temporal order doesn't matter

# ❌ BAD: Shuffle time-series that depend on order
# Don't shuffle sequences where order matters (e.g., forecasting)
```

## Performance Optimization

### Use NumPy's Vectorization

Leverage NumPy's vectorized operations:

```python
import numpy as np
from batcharray import array

batch = np.random.randn(1000, 100)

# ✅ GOOD: Vectorized operations
normalized = (batch - batch.mean(axis=0)) / (batch.std(axis=0) + 1e-8)

# ❌ BAD: Python loops
normalized = np.zeros_like(batch)
for i in range(batch.shape[0]):
    for j in range(batch.shape[1]):
        normalized[i, j] = (batch[i, j] - batch[:, j].mean()) / (batch[:, j].std() + 1e-8)
```

### Preallocate Arrays When Possible

Avoid growing arrays dynamically:

```python
import numpy as np
from batcharray import array

# ✅ GOOD: Preallocate
num_samples = 1000
results = np.zeros((num_samples, 10))
for i in range(num_samples):
    results[i] = process_sample(i)

# ❌ BAD: Growing list (slower, more memory)
results = []
for i in range(1000):
    results.append(process_sample(i))
results = np.array(results)
```

### Minimize Data Copies

Be aware of operations that copy data:

```python
import numpy as np
from batcharray import array

batch = np.random.randn(1000, 100)

# ✅ GOOD: Slicing creates views (no copy)
view = batch[:500]  # Fast, shares memory with batch

# ⚠️ AWARE: Some operations create copies
sorted_batch = array.sort_along_batch(batch)  # Creates new array

# ✅ GOOD: Reuse arrays when possible
batch_normalized = batch  # In-place if possible
batch_normalized -= batch.mean(axis=0)
batch_normalized /= (batch.std(axis=0) + 1e-8)
```

## Error Handling and Validation

### Validate Input Shapes

Check shapes before processing:

```python
import numpy as np
from batcharray import array

def process_batch(batch):
    # ✅ GOOD: Validate inputs
    if batch.ndim != 2:
        raise ValueError(f"Expected 2D array, got {batch.ndim}D")
    
    if batch.shape[0] == 0:
        raise ValueError("Batch is empty")
    
    return array.mean_along_batch(batch)

# ❌ BAD: No validation (silent failures)
def process_batch_bad(batch):
    return array.mean_along_batch(batch)
```

### Handle Edge Cases

Consider empty batches and edge cases:

```python
import numpy as np
from batcharray import array

def safe_mean(batch):
    # ✅ GOOD: Handle empty batches
    if batch.shape[0] == 0:
        return np.array([])
    return array.mean_along_batch(batch)

# Test edge cases
empty_batch = np.array([]).reshape(0, 5)
result = safe_mean(empty_batch)  # Returns empty array

single_sample = np.array([[1, 2, 3]])
result = safe_mean(single_sample)  # Returns [1, 2, 3]
```

### Use Appropriate Data Types

Choose data types carefully:

```python
import numpy as np
from batcharray import array

# ✅ GOOD: Use float for numerical computations
data = np.array([[1, 2], [3, 4]], dtype=np.float32)
mean = array.mean_along_batch(data)  # Accurate results

# ❌ BAD: Integer division issues
data = np.array([[1, 2], [3, 4]], dtype=np.int32)
mean = array.mean_along_batch(data)  # May lose precision

# ✅ GOOD: Use appropriate precision
large_data = np.random.randn(10000, 1000).astype(np.float32)  # float32 for memory
small_data = np.random.randn(10, 10).astype(np.float64)  # float64 for precision
```

## Code Organization

### Organize Related Data

Group related arrays in dictionaries:

```python
import numpy as np
from batcharray import nested

# ✅ GOOD: Organized in meaningful structure
dataset = {
    "train": {
        "X": np.random.randn(800, 10),
        "y": np.random.randint(0, 2, 800),
        "weights": np.random.rand(800)
    },
    "val": {
        "X": np.random.randn(200, 10),
        "y": np.random.randint(0, 2, 200),
        "weights": np.random.rand(200)
    }
}

# ❌ BAD: Scattered variables
train_X = np.random.randn(800, 10)
train_y = np.random.randint(0, 2, 800)
train_weights = np.random.rand(800)
val_X = np.random.randn(200, 10)
val_y = np.random.randint(0, 2, 200)
val_weights = np.random.rand(200)
```

### Write Reusable Functions

Create functions for common operations:

```python
import numpy as np
from batcharray import nested

# ✅ GOOD: Reusable preprocessing function
def preprocess_batch(batch, normalize=True, shuffle=True):
    """Preprocess a batch of data."""
    if shuffle:
        batch = nested.shuffle_along_batch(batch)
    
    if normalize:
        mean = batch["features"].mean(axis=0)
        std = batch["features"].std(axis=0)
        batch["features"] = (batch["features"] - mean) / (std + 1e-8)
    
    return batch

# Use consistently
train_batch = preprocess_batch(train_data)
val_batch = preprocess_batch(val_data, shuffle=False)
```

### Document Your Code

Add clear documentation:

```python
import numpy as np
from batcharray import array

def compute_batch_statistics(batch: np.ndarray) -> dict:
    """Compute statistical summaries for a batch.
    
    Args:
        batch: Array of shape (batch_size, num_features)
            containing the batch data.
    
    Returns:
        Dictionary containing:
            - 'mean': Mean values for each feature
            - 'std': Standard deviation for each feature
            - 'min': Minimum values for each feature
            - 'max': Maximum values for each feature
    
    Example:
        >>> batch = np.array([[1, 2], [3, 4], [5, 6]])
        >>> stats = compute_batch_statistics(batch)
        >>> stats['mean']
        array([3., 4.])
    """
    return {
        'mean': array.mean_along_batch(batch),
        'std': batch.std(axis=0),
        'min': array.amin_along_batch(batch),
        'max': array.amax_along_batch(batch)
    }
```

## Testing

### Test Batch Operations

Write tests for batch processing:

```python
import numpy as np
from batcharray import array

def test_batch_slicing():
    # ✅ GOOD: Test expected behavior
    batch = np.array([[1, 2], [3, 4], [5, 6]])
    
    # Test basic slicing
    result = array.slice_along_batch(batch, stop=2)
    expected = np.array([[1, 2], [3, 4]])
    assert np.array_equal(result, expected)
    
    # Test edge cases
    empty_result = array.slice_along_batch(batch, start=5)
    assert empty_result.shape == (0, 2)
    
    # Test full batch
    full_result = array.slice_along_batch(batch)
    assert np.array_equal(full_result, batch)
```

### Verify Consistency

Test that nested operations maintain consistency:

```python
import numpy as np
from batcharray import nested

def test_nested_consistency():
    # ✅ GOOD: Verify array alignment
    batch = {
        "X": np.random.randn(100, 10),
        "y": np.random.randint(0, 2, 100)
    }
    
    shuffled = nested.shuffle_along_batch(batch)
    
    # Verify sizes match
    assert shuffled["X"].shape[0] == shuffled["y"].shape[0]
    
    # Verify no data loss
    assert shuffled["X"].shape == batch["X"].shape
    assert shuffled["y"].shape == batch["y"].shape
```

## Common Pitfalls to Avoid

### Don't Mix Operations Inconsistently

```python
import numpy as np
from batcharray import array, nested

data = {
    "features": np.array([[1, 2], [3, 4], [5, 6]]),
    "labels": np.array([0, 1, 0])
}

# ❌ BAD: Mixing nested and individual operations
features_sliced = array.slice_along_batch(data["features"], stop=2)
labels_sliced = data["labels"][:2]  # Direct indexing
# Inconsistent and error-prone

# ✅ GOOD: Use nested operations consistently
sliced_data = nested.slice_along_batch(data, stop=2)
```

### Don't Ignore Masked Values

```python
import numpy as np
import numpy.ma as ma
from batcharray import array

# ❌ BAD: Ignoring masks
masked_data = ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 1, 0], [1, 0, 0]])
mean = masked_data.mean(axis=0)  # Wrong - includes masked values

# ✅ GOOD: Use batcharray functions that handle masks
mean = array.mean_along_batch(masked_data)  # Correctly ignores masked values
```

### Don't Assume Batch Order

```python
import numpy as np
from batcharray import array

batch = np.array([[1, 2], [3, 4], [5, 6]])

# ❌ BAD: Assuming order after shuffle
shuffled = array.shuffle_along_batch(batch)
# Don't assume shuffled[0] has any relationship to batch[0]

# ✅ GOOD: Maintain explicit indices if needed
indices = np.arange(batch.shape[0])
data = {"features": batch, "indices": indices}
shuffled = nested.shuffle_along_batch(data)
# Now shuffled["indices"] tells you the original positions
```

## Summary

Key takeaways:
1. Use the right module (`array` vs `nested`)
2. Keep batch dimension first, sequence dimension second
3. Use nested operations to maintain consistency
4. Handle edge cases and validate inputs
5. Leverage NumPy's vectorization
6. Use masked arrays for variable-length data
7. Write reusable, documented functions
8. Test your batch processing code

Following these best practices will help you write more reliable, efficient, and maintainable code with `batcharray`.
