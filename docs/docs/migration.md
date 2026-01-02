# Migration Guide

This guide helps you migrate from plain NumPy code or other libraries to `batcharray`.

## Migrating from Plain NumPy

### Basic Array Slicing

**Before (NumPy):**
```python
import numpy as np

batch = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
sliced = batch[:2]  # Get first 2 samples
```

**After (batcharray):**
```python
import numpy as np
from batcharray import array

batch = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
sliced = array.slice_along_batch(batch, stop=2)
```

**Why migrate?** More explicit dimension handling and consistency with nested operations.

### Batch Statistics

**Before (NumPy):**
```python
import numpy as np

batch = np.array([[1, 2], [3, 4], [5, 6]])
mean = batch.mean(axis=0)  # Mean along first axis
max_val = batch.max(axis=0)
```

**After (batcharray):**
```python
import numpy as np
from batcharray import array

batch = np.array([[1, 2], [3, 4], [5, 6]])
mean = array.mean_along_batch(batch)
max_val = array.amax_along_batch(batch)
```

**Why migrate?** Self-documenting function names make intent clearer.

### Index Selection

**Before (NumPy):**
```python
import numpy as np

batch = np.array([[1, 2], [3, 4], [5, 6]])
indices = np.array([0, 2])
selected = batch[indices]  # Advanced indexing
```

**After (batcharray):**
```python
import numpy as np
from batcharray import array

batch = np.array([[1, 2], [3, 4], [5, 6]])
indices = np.array([0, 2])
selected = array.index_select_along_batch(batch, indices=indices)
```

**Why migrate?** Explicit dimension specification prevents errors.

## Migrating Multiple Related Arrays

### Manual Synchronization

**Before (Error-prone):**
```python
import numpy as np

features = np.array([[1, 2], [3, 4], [5, 6]])
labels = np.array([0, 1, 0])

# Manually slice both
features_train = features[:2]
labels_train = labels[:2]

# Easy to make mistakes!
features_test = features[2:]
labels_test = labels[:2]  # BUG: Wrong slice!
```

**After (Safe):**
```python
import numpy as np
from batcharray import nested

data = {
    "features": np.array([[1, 2], [3, 4], [5, 6]]),
    "labels": np.array([0, 1, 0])
}

# Automatically keeps arrays synchronized
train = nested.slice_along_batch(data, stop=2)
test = nested.slice_along_batch(data, start=2)
```

**Why migrate?** Eliminates synchronization bugs.

### Shuffling Related Arrays

**Before (Manual seed management):**
```python
import numpy as np

features = np.array([[1, 2], [3, 4], [5, 6]])
labels = np.array([0, 1, 0])

# Manual shuffling with same seed
np.random.seed(42)
perm = np.random.permutation(len(features))
features_shuffled = features[perm]

np.random.seed(42)  # Must remember to reset seed!
perm = np.random.permutation(len(labels))
labels_shuffled = labels[perm]
```

**After (Automatic):**
```python
import numpy as np
from batcharray import nested

data = {
    "features": np.array([[1, 2], [3, 4], [5, 6]]),
    "labels": np.array([0, 1, 0])
}

# Single shuffle keeps everything aligned
shuffled = nested.shuffle_along_batch(data)
```

**Why migrate?** Guaranteed consistency without manual seed management.

## Migrating from PyTorch DataLoader Patterns

### Mini-batch Processing

**Before (PyTorch-style):**
```python
import numpy as np

# Manual batch creation
dataset_X = np.random.randn(1000, 10)
dataset_y = np.random.randint(0, 2, 1000)

batch_size = 32
for i in range(0, len(dataset_X), batch_size):
    batch_X = dataset_X[i:i+batch_size]
    batch_y = dataset_y[i:i+batch_size]
    # Process batch
```

**After (batcharray):**
```python
import numpy as np
from batcharray import nested

dataset = {
    "X": np.random.randn(1000, 10),
    "y": np.random.randint(0, 2, 1000)
}

batch_size = 32
num_batches = (1000 + batch_size - 1) // batch_size

for i in range(num_batches):
    start = i * batch_size
    stop = min((i + 1) * batch_size, 1000)
    batch = nested.slice_along_batch(dataset, start=start, stop=stop)
    # Process batch
```

**Why migrate?** More flexible nested structure support.

### Train/Test Split

**Before (scikit-learn style):**
```python
import numpy as np
from sklearn.model_selection import train_test_split

X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**After (batcharray):**
```python
import numpy as np
from batcharray import nested

dataset = {
    "X": np.random.randn(1000, 10),
    "y": np.random.randint(0, 2, 1000)
}

# Shuffle first
dataset = nested.shuffle_along_batch(dataset)

# Split
train_size = int(0.8 * 1000)
train = nested.slice_along_batch(dataset, stop=train_size)
test = nested.slice_along_batch(dataset, start=train_size)
```

**Why migrate?** Works seamlessly with nested structures.

## Migrating Sequence Operations

### Time Series Slicing

**Before (NumPy):**
```python
import numpy as np

# Shape: (batch, time, features)
sequences = np.random.randn(10, 100, 5)

# Slice time dimension
window = sequences[:, 20:80, :]  # Manual indexing
```

**After (batcharray):**
```python
import numpy as np
from batcharray import array

sequences = np.random.randn(10, 100, 5)

# Explicit sequence slicing
window = array.slice_along_seq(sequences, start=20, stop=80)
```

**Why migrate?** Clearer intent, consistent API.

### Sequence Statistics

**Before (NumPy):**
```python
import numpy as np

sequences = np.random.randn(10, 100, 5)

# Mean over time
mean_over_time = sequences.mean(axis=1)  # Axis number might be unclear
```

**After (batcharray):**
```python
import numpy as np
from batcharray import array

sequences = np.random.randn(10, 100, 5)

# Self-documenting
mean_over_time = array.mean_along_seq(sequences)
```

**Why migrate?** Function names make the operation explicit.

## Migrating from Custom Batch Classes

### Custom Batch Container

**Before (Custom class):**
```python
import numpy as np

class Batch:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def slice(self, start, stop):
        return Batch(
            self.features[start:stop],
            self.labels[start:stop]
        )
    
    def shuffle(self):
        perm = np.random.permutation(len(self.features))
        return Batch(
            self.features[perm],
            self.labels[perm]
        )

# Usage
batch = Batch(
    features=np.random.randn(100, 10),
    labels=np.random.randint(0, 2, 100)
)
sliced = batch.slice(0, 50)
shuffled = batch.shuffle()
```

**After (batcharray):**
```python
import numpy as np
from batcharray import nested

# No custom class needed
batch = {
    "features": np.random.randn(100, 10),
    "labels": np.random.randint(0, 2, 100)
}

sliced = nested.slice_along_batch(batch, stop=50)
shuffled = nested.shuffle_along_batch(batch)
```

**Why migrate?** 
- No boilerplate code
- Works with any dict/list structure
- More flexible
- Better tested

## Handling Missing Data

### NaN Values to Masked Arrays

**Before (Manual NaN handling):**
```python
import numpy as np

data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])

# Manual NaN filtering
mean = np.nanmean(data, axis=0)
```

**After (batcharray with masked arrays):**
```python
import numpy as np
import numpy.ma as ma
from batcharray import array

# Convert NaN to masked array
data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
masked_data = ma.masked_invalid(data)

# Automatic handling
mean = array.mean_along_batch(masked_data)
```

**Why migrate?** More consistent handling of missing data across operations.

## Step-by-Step Migration Strategy

### 1. Identify Batch Operations

Find all code that:
- Slices along the first dimension
- Computes statistics over batches
- Processes multiple related arrays together

### 2. Group Related Arrays

Convert separate variables to dictionaries:

```python
# Before
features = np.random.randn(100, 10)
labels = np.random.randint(0, 2, 100)
weights = np.random.rand(100)

# After
data = {
    "features": features,
    "labels": labels,
    "weights": weights
}
```

### 3. Replace Operations Incrementally

Migrate one operation at a time:

```python
import numpy as np
from batcharray import nested

data = {
    "features": np.random.randn(100, 10),
    "labels": np.random.randint(0, 2, 100)
}

# Step 1: Replace manual slicing
# Before: train_features = data["features"][:80]
train = nested.slice_along_batch(data, stop=80)

# Step 2: Replace shuffling
# Before: manual permutation
shuffled = nested.shuffle_along_batch(data)

# Step 3: Replace splitting
# Before: manual index calculations
batches = nested.split_along_batch(data, split_size_or_sections=10)
```

### 4. Test Incrementally

Add tests to verify migration:

```python
import numpy as np
from batcharray import nested

def test_migration():
    # Original code result
    features = np.array([[1, 2], [3, 4], [5, 6]])
    labels = np.array([0, 1, 0])
    original_features = features[:2]
    original_labels = labels[:2]
    
    # Migrated code result
    data = {"features": features, "labels": labels}
    migrated = nested.slice_along_batch(data, stop=2)
    
    # Verify equivalence
    assert np.array_equal(original_features, migrated["features"])
    assert np.array_equal(original_labels, migrated["labels"])

test_migration()
```

## Common Migration Patterns

### Pattern 1: Multiple Array Slicing

```python
# Before
train_X = X[:800]
train_y = y[:800]
val_X = X[800:]
val_y = y[800:]

# After
data = {"X": X, "y": y}
train = nested.slice_along_batch(data, stop=800)
val = nested.slice_along_batch(data, start=800)
```

### Pattern 2: Batch Processing Loop

```python
# Before
for i in range(0, len(X), batch_size):
    batch_X = X[i:i+batch_size]
    batch_y = y[i:i+batch_size]
    process(batch_X, batch_y)

# After
data = {"X": X, "y": y}
batches = nested.split_along_batch(data, split_size_or_sections=batch_size)
for batch in batches:
    process(batch["X"], batch["y"])
```

### Pattern 3: Random Sampling

```python
# Before
indices = np.random.choice(len(X), size=100, replace=False)
sample_X = X[indices]
sample_y = y[indices]

# After
data = {"X": X, "y": y}
indices = np.random.choice(data["X"].shape[0], size=100, replace=False)
sample = nested.index_select_along_batch(data, indices=indices)
```

## Compatibility Considerations

### NumPy Compatibility

`batcharray` is fully compatible with NumPy:

```python
import numpy as np
from batcharray import array

# All NumPy arrays work
batch = np.array([[1, 2], [3, 4]])
result = array.slice_along_batch(batch, stop=1)

# Result is still a NumPy array
assert isinstance(result, np.ndarray)
```

### Type Hints

`batcharray` supports type hints:

```python
import numpy as np
from typing import Dict
from batcharray import nested

def process_batch(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Process a batch of data."""
    return nested.shuffle_along_batch(batch)
```

## Performance Impact

Migration to `batcharray` has minimal performance impact:

```python
import numpy as np
from batcharray import array
import timeit

batch = np.random.randn(1000, 100)

# NumPy direct
time_numpy = timeit.timeit(lambda: batch[:500], number=10000)

# batcharray
time_batcharray = timeit.timeit(
    lambda: array.slice_along_batch(batch, stop=500),
    number=10000
)

overhead = (time_batcharray - time_numpy) / time_numpy * 100
print(f"Overhead: {overhead:.1f}%")  # Typically < 5%
```

## Summary

Migration to `batcharray` provides:

1. ✅ **Safer code**: Automatic array synchronization
2. ✅ **Clearer intent**: Self-documenting function names
3. ✅ **Consistent API**: Same patterns for all operations
4. ✅ **Nested support**: Handle complex structures easily
5. ✅ **Minimal overhead**: Similar performance to NumPy
6. ✅ **Incremental migration**: Adopt gradually

Start by migrating your most error-prone code (manual array synchronization) and expand from there.
