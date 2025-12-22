# Troubleshooting Guide

This guide helps you diagnose and fix common issues when using `batcharray`.

## Common Errors

### IndexError: axis X is out of bounds for array of dimension Y

**Problem**: You're trying to perform an operation on an axis that doesn't exist in your array.

**Cause**: The array doesn't have the expected number of dimensions.

**Solution**:

```python
import numpy as np
from batcharray import array

# Error: 1D array, but trying batch operation
arr = np.array([1, 2, 3])  # Shape: (3,)
# array.slice_along_batch(arr, stop=2)  # Would fail!

# Fix: Reshape to have batch dimension
arr = arr.reshape(-1, 1)  # Shape: (3, 1) - now has batch dimension
result = array.slice_along_batch(arr, stop=2)  # Works!
```

For sequences, you need at least 2D arrays:

```python
import numpy as np
from batcharray import array

# Error: Need at least 2D for sequence operations
arr = np.array([1, 2, 3])

# Fix: Add batch and sequence dimensions
arr = arr.reshape(1, -1)  # Shape: (1, 3) - batch_size=1, seq_len=3
result = array.slice_along_seq(arr, stop=2)
```

### ValueError: all the input arrays must have same number of dimensions

**Problem**: When using nested operations, arrays in your structure have incompatible shapes.

**Cause**: Arrays don't have the same batch size or incompatible dimensions.

**Solution**:

```python
import numpy as np
from batcharray import nested

# Error: Incompatible batch sizes
bad_data = {
    "a": np.array([[1, 2], [3, 4]]),  # batch_size=2
    "b": np.array([5, 6, 7]),  # batch_size=3
}
# nested.slice_along_batch(bad_data, stop=1)  # Fails!

# Fix: Ensure consistent batch sizes
good_data = {
    "a": np.array([[1, 2], [3, 4], [5, 6]]),  # batch_size=3
    "b": np.array([7, 8, 9]),  # batch_size=3
}
result = nested.slice_along_batch(good_data, stop=1)  # Works!
```

### TypeError: unsupported operand type(s)

**Problem**: Trying to use operations on incompatible types.

**Cause**: Mixing arrays with non-array types in places where arrays are expected.

**Solution**:

```python
import numpy as np
from batcharray import nested

# Error: Non-array value
bad_data = {"array": np.array([1, 2, 3]), "scalar": 42}  # Not an array!
# nested.mean_along_batch(bad_data)  # Fails on scalar

# Fix: Ensure all values are arrays
good_data = {
    "array": np.array([1, 2, 3]),
    "scalar": np.array([42, 42, 42]),  # Now an array
}
result = nested.mean_along_batch(good_data)  # Works!
```

### AttributeError: 'dict' object has no attribute 'shape'

**Problem**: Passing a dictionary to an `array` function instead of `nested`.

**Cause**: Using wrong module for your data type.

**Solution**:

```python
import numpy as np
from batcharray import array, nested

data = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}

# Error: Using array module with dict
# result = array.slice_along_batch(data, stop=2)  # Fails!

# Fix: Use nested module for dictionaries
result = nested.slice_along_batch(data, stop=2)  # Works!

# For single arrays, use array module
single_array = np.array([1, 2, 3])
result = array.slice_along_batch(single_array, stop=2)  # Works!
```

## Performance Issues

### Operations are slower than expected

**Problem**: Operations take longer than anticipated.

**Diagnosis**:

```python
import numpy as np
import time
from batcharray import nested

# Time your operation
data = {f"arr{i}": np.random.randn(1000, 100) for i in range(100)}

start = time.time()
result = nested.slice_along_batch(data, stop=500)
elapsed = time.time() - start
print(f"Operation took {elapsed:.3f} seconds")
```

**Common causes and solutions**:

1. **Large nested structures**: Consider flattening or reducing nesting depth
2. **Unnecessary copies**: Use views when possible
3. **Type conversions**: Avoid repeated conversions between types

```python
import numpy as np
from batcharray import array

# Slow: Creating copies
large_array = np.random.randn(10000, 1000)
for i in range(100):
    copy = large_array.copy()  # Expensive!
    result = array.mean_along_batch(copy)

# Fast: Use views
for i in range(100):
    view = large_array  # No copy
    result = array.mean_along_batch(view)
```

### High memory usage

**Problem**: Your program uses more memory than expected.

**Diagnosis**:

```python
import numpy as np
from batcharray.utils import bfs_array


def analyze_memory(data):
    """Check memory usage of nested structure."""
    total_bytes = sum(arr.nbytes for arr in bfs_array(data))
    print(f"Total memory: {total_bytes / (1024**2):.2f} MB")

    # Check for large arrays
    for i, arr in enumerate(bfs_array(data)):
        if arr.nbytes > 1024**2:  # > 1 MB
            print(f"Array {i}: {arr.nbytes / (1024**2):.2f} MB, shape={arr.shape}")


data = {
    "features": np.random.randn(10000, 1000),
    "labels": np.random.randint(0, 10, 10000),
}
analyze_memory(data)
```

**Solutions**:

1. **Use smaller dtypes**:

```python
import numpy as np

# Wasteful: Using float64 when float32 is sufficient
large = np.random.randn(10000, 1000).astype(np.float64)  # 80 MB

# Efficient: Use float32
smaller = np.random.randn(10000, 1000).astype(np.float32)  # 40 MB
```

2. **Delete unused arrays**:

```python
import numpy as np
from batcharray import array

data = np.random.randn(10000, 1000)
result = array.mean_along_batch(data)

# Free memory if you don't need original
del data
```

3. **Process in chunks**:

```python
import numpy as np
from batcharray import array

large_data = np.random.randn(100000, 100)

# Process in chunks to reduce memory
chunks = array.chunk_along_batch(large_data, chunks=10)
results = []
for chunk in chunks:
    result = array.mean_along_batch(chunk)
    results.append(result)
```

## Data Issues

### NaN or Inf values in results

**Problem**: Unexpected NaN or Inf values appear in results.

**Diagnosis**:

```python
import numpy as np


def check_array(arr):
    """Check for problematic values."""
    print(f"Shape: {arr.shape}")
    print(f"Contains NaN: {np.isnan(arr).any()}")
    print(f"Contains Inf: {np.isinf(arr).any()}")
    print(f"Min: {np.nanmin(arr)}, Max: {np.nanmax(arr)}")


data = np.array([1.0, 2.0, np.nan, 4.0])
check_array(data)
```

**Solutions**:

1. **Use masked arrays**:

```python
import numpy as np
import numpy.ma as ma
from batcharray import array

# Replace NaN with masked values
data = np.array([1.0, 2.0, np.nan, 4.0])
masked_data = ma.masked_invalid(data)

mean = array.mean_along_batch(masked_data)  # Ignores NaN
```

2. **Clean data before processing**:

```python
import numpy as np

data = np.array([1.0, 2.0, np.nan, 4.0, np.inf])

# Remove invalid values
clean_data = data[np.isfinite(data)]

# Or replace with specific value
data[~np.isfinite(data)] = 0
```

### Type mismatches

**Problem**: Operations fail due to incompatible types.

**Diagnosis**:

```python
import numpy as np
from batcharray.utils import bfs_array


def check_types(data):
    """Check types of all arrays in nested structure."""
    for i, arr in enumerate(bfs_array(data)):
        print(f"Array {i}: dtype={arr.dtype}, type={type(arr)}")


data = {
    "int": np.array([1, 2, 3], dtype=np.int32),
    "float": np.array([1.0, 2.0, 3.0], dtype=np.float64),
}
check_types(data)
```

**Solutions**:

1. **Convert to consistent type**:

```python
import numpy as np
from batcharray.recursive import recursive_apply


def ensure_float32(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    return x


data = {
    "int_array": np.array([1, 2, 3], dtype=np.int64),
    "float_array": np.array([1.0, 2.0], dtype=np.float64),
}

# Convert all to float32
data = recursive_apply(data, ensure_float32)
```

2. **Handle types explicitly**:

```python
import numpy as np
from batcharray import nested

data = {
    "a": np.array([1, 2, 3], dtype=np.int32),
    "b": np.array([1.0, 2.0, 3.0], dtype=np.float32),
}

# Operations handle mixed types
result = nested.sum_along_batch(data)
# Result dtypes preserved
```

## Installation Issues

### Import errors

**Problem**: `ImportError: cannot import name 'X' from 'batcharray'`

**Solution**:

1. **Check version**:

```bash
python -c "import batcharray; print(batcharray.__version__)"
```

2. **Reinstall**:

```bash
pip uninstall batcharray
pip install batcharray
```

3. **Check dependencies**:

```bash
pip show batcharray
pip install --upgrade numpy coola
```

### Conflicting dependencies

**Problem**: Dependency conflicts during installation.

**Solution**:

1. **Use virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Unix
# or
venv\Scripts\activate  # On Windows

pip install batcharray
```

2. **Check for conflicts**:

```bash
pip check
```

3. **Update pip**:

```bash
pip install --upgrade pip
pip install batcharray
```

## Debugging Strategies

### Enable verbose logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Your batcharray code here
```

### Validate data structure

```python
import numpy as np
from batcharray.utils import bfs_array


def validate_structure(data, expected_batch_size=None):
    """Validate nested data structure."""
    arrays = list(bfs_array(data))

    if not arrays:
        print("Warning: No arrays found in structure")
        return False

    print(f"Found {len(arrays)} arrays")

    # Check batch sizes
    batch_sizes = [arr.shape[0] if arr.ndim > 0 else None for arr in arrays]
    unique_sizes = set(batch_sizes)

    if len(unique_sizes) > 1:
        print(f"Warning: Inconsistent batch sizes: {unique_sizes}")
        return False

    if expected_batch_size and unique_sizes.pop() != expected_batch_size:
        print(f"Warning: Expected batch size {expected_batch_size}, got {unique_sizes}")
        return False

    print("✓ Structure is valid")
    return True


data = {"a": np.array([[1, 2], [3, 4]]), "b": np.array([5, 6])}

validate_structure(data, expected_batch_size=2)
```

### Minimal reproduction

Create minimal example to isolate the issue:

```python
import numpy as np
from batcharray import array

# Minimal example
data = np.array([[1, 2], [3, 4], [5, 6]])
result = array.slice_along_batch(data, stop=2)

print("Input shape:", data.shape)
print("Output shape:", result.shape)
print("Result:", result)
```

## Getting Help

If you can't resolve your issue:

1. **Check documentation**: [https://durandtibo.github.io/batcharray/](https://durandtibo.github.io/batcharray/)
2. **Search existing issues**: [GitHub Issues](https://github.com/durandtibo/batcharray/issues)
3. **Ask for help**: Open a new issue with:
   - Python version
   - batcharray version
   - NumPy version
   - Minimal code to reproduce
   - Full error traceback
   - Expected vs actual behavior

## Reporting Bugs

When reporting bugs, include:

```python
import sys
import numpy as np
import batcharray

print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"batcharray: {batcharray.__version__}")

# Your code that causes the issue
# ...
```

This information helps maintainers reproduce and fix the issue quickly.
