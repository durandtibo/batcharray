# Performance Tips

This guide provides tips and techniques to optimize performance when using `batcharray`.

## General Performance Principles

### Use Vectorized Operations

NumPy's vectorized operations are significantly faster than Python loops:

```python
import numpy as np
from batcharray import array
import time

batch = np.random.randn(10000, 100)

# ✅ FAST: Vectorized operations
start = time.time()
normalized = (batch - batch.mean(axis=0)) / (batch.std(axis=0) + 1e-8)
vectorized_time = time.time() - start

# ❌ SLOW: Python loops
start = time.time()
normalized_slow = np.zeros_like(batch)
for i in range(batch.shape[0]):
    for j in range(batch.shape[1]):
        mean = batch[:, j].mean()
        std = batch[:, j].std()
        normalized_slow[i, j] = (batch[i, j] - mean) / (std + 1e-8)
loop_time = time.time() - start

print(f"Vectorized: {vectorized_time:.4f}s")
print(f"Loops: {loop_time:.4f}s")
print(f"Speedup: {loop_time / vectorized_time:.1f}x")
```

### Minimize Memory Allocations

Reuse arrays when possible:

```python
import numpy as np

batch = np.random.randn(1000, 100)

# ✅ GOOD: In-place operations (when safe)
batch -= batch.mean(axis=0)
batch /= (batch.std(axis=0) + 1e-8)

# ❌ LESS EFFICIENT: Creates intermediate arrays
normalized = (batch - batch.mean(axis=0)) / (batch.std(axis=0) + 1e-8)
```

### Choose Appropriate Data Types

Use the smallest data type that meets your precision requirements:

```python
import numpy as np

# For large datasets where precision isn't critical
data_float32 = np.random.randn(10000, 1000).astype(np.float32)  # 40MB
print(f"float32 memory: {data_float32.nbytes / 1e6:.1f} MB")

# More precise but uses more memory
data_float64 = np.random.randn(10000, 1000).astype(np.float64)  # 80MB
print(f"float64 memory: {data_float64.nbytes / 1e6:.1f} MB")

# For integer data
labels = np.random.randint(0, 10, 10000, dtype=np.int8)  # 10KB
print(f"int8 memory: {labels.nbytes / 1e3:.1f} KB")
```

## Batch Processing Optimization

### Optimal Batch Sizes

Choose batch sizes based on your memory and compute constraints:

```python
import numpy as np
from batcharray import nested

def optimal_batch_size(total_samples, feature_size, available_memory_gb=4):
    """Calculate optimal batch size based on available memory."""
    bytes_per_sample = feature_size * 8  # float64
    bytes_per_batch = bytes_per_sample
    
    # Use ~80% of available memory for batches
    max_batch_bytes = available_memory_gb * 1e9 * 0.8
    
    batch_size = int(max_batch_bytes / bytes_per_batch)
    
    # Common batch sizes (powers of 2)
    common_sizes = [16, 32, 64, 128, 256, 512, 1024]
    batch_size = max([s for s in common_sizes if s <= batch_size], default=32)
    
    return batch_size

# Example
dataset_size = 100000
features = 784
batch_size = optimal_batch_size(dataset_size, features)
print(f"Recommended batch size: {batch_size}")
```

### Efficient Batch Iteration

Use generators to avoid loading all data at once:

```python
import numpy as np
from batcharray import nested

def batch_generator(data, batch_size):
    """Generate batches efficiently."""
    num_samples = data["X"].shape[0]
    indices = np.arange(num_samples)
    
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_indices = indices[start:end]
        
        yield nested.index_select_along_batch(data, indices=batch_indices)

# Usage
large_data = {
    "X": np.random.randn(100000, 100),
    "y": np.random.randint(0, 10, 100000)
}

for batch in batch_generator(large_data, batch_size=256):
    # Process batch
    pass
```

## Array Operations Performance

### Slicing vs Copying

Understand when operations create copies:

```python
import numpy as np
from batcharray import array

batch = np.random.randn(10000, 100)

# ✅ FAST: Slicing creates view (no copy)
view = batch[:5000]  # Very fast, shares memory

# ⚠️ SLOWER: Some operations create copies
sorted_batch = array.sort_along_batch(batch)  # Creates copy
shuffled = array.shuffle_along_batch(batch)  # Creates copy

# Check if array is a view
print(f"Is view: {view.base is batch}")  # True
print(f"Is copy: {sorted_batch.base is None}")  # True
```

### Preallocate Arrays

Preallocate when building arrays in loops:

```python
import numpy as np

num_samples = 10000

# ✅ GOOD: Preallocate
results = np.zeros((num_samples, 10))
for i in range(num_samples):
    results[i] = process(i)

# ❌ BAD: Growing list
results = []
for i in range(num_samples):
    results.append(process(i))
results = np.array(results)  # Slow conversion
```

## Nested Structure Performance

### Shallow vs Deep Nesting

Prefer shallow nesting when possible:

```python
import numpy as np
from batcharray import nested

# ✅ BETTER: Shallow nesting
data_shallow = {
    "train_X": np.random.randn(1000, 10),
    "train_y": np.random.randint(0, 2, 1000),
    "val_X": np.random.randn(200, 10),
    "val_y": np.random.randint(0, 2, 200)
}

# ⚠️ SLOWER: Deep nesting
data_deep = {
    "train": {
        "features": {
            "raw": np.random.randn(1000, 10),
            "normalized": np.random.randn(1000, 10)
        },
        "labels": np.random.randint(0, 2, 1000)
    },
    "val": {
        "features": {
            "raw": np.random.randn(200, 10),
            "normalized": np.random.randn(200, 10)
        },
        "labels": np.random.randint(0, 2, 200)
    }
}

# Operations on deep structures are slower due to recursion
```

### Batch Nested Operations

Group operations on nested structures:

```python
import numpy as np
from batcharray import nested

data = {
    "features": np.random.randn(1000, 50),
    "labels": np.random.randint(0, 10, 1000)
}

# ✅ GOOD: Single nested operation
sliced = nested.slice_along_batch(data, stop=800)
shuffled = nested.shuffle_along_batch(sliced)

# ❌ LESS EFFICIENT: Separate operations
features_sliced = data["features"][:800]
labels_sliced = data["labels"][:800]
features_shuffled = array.shuffle_along_batch(features_sliced)
labels_shuffled = array.shuffle_along_batch(labels_sliced)
```

## Masked Array Performance

### Minimize Mask Operations

Masked arrays have overhead; use only when necessary:

```python
import numpy as np
import numpy.ma as ma
from batcharray import array
import time

data = np.random.randn(10000, 100)
mask = np.random.random((10000, 100)) < 0.1  # 10% masked

# Regular array (faster)
start = time.time()
mean_regular = data.mean(axis=0)
regular_time = time.time() - start

# Masked array (slower but handles missing data)
masked_data = ma.array(data, mask=mask)
start = time.time()
mean_masked = array.mean_along_batch(masked_data)
masked_time = time.time() - start

print(f"Regular: {regular_time:.4f}s")
print(f"Masked: {masked_time:.4f}s")
print(f"Overhead: {masked_time / regular_time:.1f}x")
```

### Compress Masked Arrays

Remove masked values when possible:

```python
import numpy as np
import numpy.ma as ma

# Masked array with many masked values
data = ma.array(
    np.random.randn(1000, 100),
    mask=np.random.random((1000, 100)) < 0.5  # 50% masked
)

# ✅ GOOD: Compress to remove masked values (when appropriate)
compressed = data.compressed()  # 1D array of non-masked values
# Faster operations on compressed data

# Process compressed data
result = compressed.mean()
```

## Memory Management

### Monitor Memory Usage

Track memory usage to avoid issues:

```python
import numpy as np
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Monitor memory during processing
initial_memory = get_memory_usage()
print(f"Initial memory: {initial_memory:.1f} MB")

# Create large array
large_data = np.random.randn(10000, 10000)
after_create = get_memory_usage()
print(f"After creation: {after_create:.1f} MB")
print(f"Data size: {large_data.nbytes / 1024 / 1024:.1f} MB")

# Clean up
del large_data
final_memory = get_memory_usage()
print(f"After deletion: {final_memory:.1f} MB")
```

### Use Memory Views

Views avoid unnecessary copies:

```python
import numpy as np

large_array = np.random.randn(10000, 1000)

# ✅ GOOD: Use views for subsets
subset = large_array[:5000]  # View, no copy
print(f"Is view: {subset.base is large_array}")

# Only copy when necessary
if need_independent_copy:
    subset = large_array[:5000].copy()
```

## Computation Model Performance

### Choose the Right Model

Use specific models when possible:

```python
import numpy as np
import numpy.ma as ma
from batcharray.computation import ArrayComputationModel, MaskedArrayComputationModel, AutoComputationModel

# Regular arrays
regular_data = np.random.randn(1000, 100)

# ✅ FASTEST: Specific model
array_model = ArrayComputationModel()
result = array_model.max(regular_data, axis=0)

# ⚠️ SLOWER: Auto model (needs to check type)
auto_model = AutoComputationModel()
result = auto_model.max(regular_data, axis=0)

# For masked arrays, use MaskedArrayComputationModel
masked_data = ma.array(regular_data, mask=np.random.random((1000, 100)) < 0.1)
masked_model = MaskedArrayComputationModel()
result = masked_model.max(masked_data, axis=0)
```

## Profiling and Benchmarking

### Profile Your Code

Use profiling to find bottlenecks:

```python
import numpy as np
from batcharray import nested
import cProfile
import pstats

def process_pipeline():
    """Example processing pipeline."""
    data = {
        "X": np.random.randn(10000, 100),
        "y": np.random.randint(0, 10, 10000)
    }
    
    # Shuffle
    data = nested.shuffle_along_batch(data)
    
    # Split
    train = nested.slice_along_batch(data, stop=8000)
    val = nested.slice_along_batch(data, start=8000)
    
    # Process
    train_mean = nested.mean_along_batch(train)
    val_mean = nested.mean_along_batch(val)
    
    return train_mean, val_mean

# Profile
profiler = cProfile.Profile()
profiler.enable()
process_pipeline()
profiler.disable()

# Print results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

### Benchmark Different Approaches

Compare different implementations:

```python
import numpy as np
from batcharray import array, nested
import timeit

def benchmark(func, *args, number=100):
    """Benchmark a function."""
    time = timeit.timeit(lambda: func(*args), number=number)
    return time / number

# Compare approaches
batch = np.random.randn(1000, 100)

# Approach 1: Direct slicing
time1 = benchmark(lambda: batch[:500])

# Approach 2: Using batcharray
time2 = benchmark(lambda: array.slice_along_batch(batch, stop=500))

print(f"Direct slicing: {time1*1e6:.2f} μs")
print(f"Batcharray: {time2*1e6:.2f} μs")
print(f"Overhead: {(time2-time1)/time1*100:.1f}%")
```

## Summary of Best Practices

1. **Use vectorized operations** instead of Python loops
2. **Choose appropriate batch sizes** based on memory constraints
3. **Minimize memory allocations** by reusing arrays
4. **Use appropriate data types** (float32 vs float64, int8 vs int64)
5. **Prefer views over copies** when possible
6. **Use masked arrays only when needed** (they have overhead)
7. **Preallocate arrays** when building incrementally
8. **Profile your code** to find actual bottlenecks
9. **Use specific computation models** when type is known
10. **Keep nesting shallow** for better performance

## Platform-Specific Optimizations

### Use NumPy with Optimized BLAS

Ensure NumPy uses optimized BLAS libraries:

```python
import numpy as np

# Check NumPy configuration
print(np.__config__.show())

# Look for optimized libraries like:
# - MKL (Intel Math Kernel Library)
# - OpenBLAS
# - ATLAS
```

### Parallel Processing

For very large datasets, consider parallel processing:

```python
import numpy as np
from multiprocessing import Pool
from batcharray import nested

def process_batch(batch):
    """Process a single batch."""
    return nested.mean_along_batch(batch)

# Split data into chunks for parallel processing
data = {
    "X": np.random.randn(100000, 100),
    "y": np.random.randint(0, 10, 100000)
}

chunks = nested.split_along_batch(data, split_size_or_sections=10)

# Process in parallel
with Pool(processes=4) as pool:
    results = pool.map(process_batch, chunks)
```

Following these performance tips will help you get the most out of `batcharray` for efficient data processing.
