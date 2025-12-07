# Utility Functions

The `batcharray.utils` module provides utility functions for traversing and searching through nested data structures containing arrays.

## Overview

When working with complex nested data structures, you may need to:

- Find all arrays in a nested structure
- Visit arrays in a specific order
- Search for arrays meeting certain criteria
- Analyze the structure of your data

The utils module provides two traversal strategies: breadth-first search (BFS) and depth-first search (DFS).

## Breadth-First Search (BFS)

Breadth-first search visits data structures level by level, processing all items at one depth before moving to the next depth.

### Basic Usage

```python
import numpy as np
from batcharray.utils import bfs_array

# Simple nested structure
data = {
    "a": np.array([1, 2, 3]),
    "b": {
        "c": np.array([4, 5, 6]),
        "d": np.array([7, 8, 9])
    }
}

# Find all arrays using BFS
arrays = list(bfs_array(data))
# Returns arrays in breadth-first order:
# [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
```

### Traversal Order

BFS processes structures level by level:

```python
import numpy as np
from batcharray.utils import bfs_array

data = {
    "level1_a": np.array([1]),           # Depth 1
    "level1_b": {
        "level2_a": np.array([2]),       # Depth 2
        "level2_b": {
            "level3": np.array([3])      # Depth 3
        }
    }
}

arrays = list(bfs_array(data))
# Order: [1], [2], [3]
# Visits all level 1 arrays, then level 2, then level 3
```

### Use Cases for BFS

BFS is useful when you want to:

1. **Process by hierarchy level**: Handle all top-level data before deeper nested data
2. **Find shallow arrays first**: Locate arrays close to the root quickly
3. **Memory-efficient wide structures**: Better for wide, shallow structures

```python
import numpy as np
from batcharray.utils import bfs_array

# Configuration with nested parameters
config = {
    "model": {
        "layer1": {"weights": np.random.randn(10, 5)},
        "layer2": {"weights": np.random.randn(5, 3)}
    },
    "optimizer": {
        "momentum": np.array([0.9])
    }
}

# Collect all parameter arrays
params = list(bfs_array(config))
print(f"Found {len(params)} parameter arrays")
```

## Depth-First Search (DFS)

Depth-first search follows each branch to its deepest point before backtracking to explore other branches.

### Basic Usage

```python
import numpy as np
from batcharray.utils import dfs_array

data = {
    "a": np.array([1, 2, 3]),
    "b": {
        "c": np.array([4, 5, 6]),
        "d": np.array([7, 8, 9])
    }
}

# Find all arrays using DFS
arrays = list(dfs_array(data))
# Returns arrays in depth-first order
```

### Traversal Order

DFS explores deeply before moving to siblings:

```python
import numpy as np
from batcharray.utils import dfs_array

data = {
    "branch1": {
        "deep": {
            "deeper": np.array([1])      # Visited 2nd
        }
    },
    "branch2": np.array([2])             # Visited 3rd
}

arrays = list(dfs_array(data))
# Order: explores branch1 completely, then branch2
```

### Use Cases for DFS

DFS is useful when you want to:

1. **Process complete branches**: Handle entire subtrees before moving to others
2. **Memory-efficient for deep structures**: Better for deep, narrow structures
3. **Path-dependent operations**: When you need to track the path from root to leaf

```python
import numpy as np
from batcharray.utils import dfs_array

# Neural network structure
network = {
    "encoder": {
        "layer1": np.random.randn(100, 50),
        "layer2": np.random.randn(50, 25)
    },
    "decoder": {
        "layer1": np.random.randn(25, 50),
        "layer2": np.random.randn(50, 100)
    }
}

# Process all encoder layers before decoder layers
for array in dfs_array(network):
    print(f"Array shape: {array.shape}")
```

## Comparison: BFS vs DFS

### Example Structure

```python
import numpy as np
from batcharray.utils import bfs_array, dfs_array

data = {
    "A": np.array([1]),
    "B": {
        "B1": np.array([2]),
        "B2": {
            "B2a": np.array([3])
        }
    },
    "C": np.array([4])
}

bfs_order = [arr[0] for arr in bfs_array(data)]
# BFS: [1, 4, 2, 3] - visits A, C (same level), then B1, then B2a

dfs_order = [arr[0] for arr in dfs_array(data)]
# DFS: [1, 2, 3, 4] - completes A, then entire B branch, then C
```

### When to Use Each

| Scenario | Prefer BFS | Prefer DFS |
|----------|-----------|-----------|
| Wide, shallow structures | ✓ | |
| Deep, narrow structures | | ✓ |
| Need to process by level | ✓ | |
| Need to process complete branches | | ✓ |
| Searching for something close to root | ✓ | |
| Path-dependent processing | | ✓ |

## Advanced Usage

### Filtering Arrays

Combine with filters to find specific arrays:

```python
import numpy as np
from batcharray.utils import bfs_array

data = {
    "float_data": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    "int_data": np.array([1, 2, 3], dtype=np.int32),
    "nested": {
        "more_floats": np.array([4.0, 5.0], dtype=np.float32)
    }
}

# Find only float32 arrays
float_arrays = [
    arr for arr in bfs_array(data) 
    if arr.dtype == np.float32
]
```

### Computing Statistics

Use traversal to compute statistics across all arrays:

```python
import numpy as np
from batcharray.utils import bfs_array

data = {
    "train": {
        "features": np.random.randn(100, 10),
        "labels": np.random.randint(0, 5, 100)
    },
    "val": {
        "features": np.random.randn(20, 10),
        "labels": np.random.randint(0, 5, 20)
    }
}

# Count total number of arrays
num_arrays = sum(1 for _ in bfs_array(data))

# Compute total memory usage
total_bytes = sum(arr.nbytes for arr in bfs_array(data))
total_mb = total_bytes / (1024 * 1024)
print(f"Total memory: {total_mb:.2f} MB")

# Find shape statistics
shapes = [arr.shape for arr in bfs_array(data)]
print(f"Array shapes: {shapes}")
```

### Validation

Validate all arrays in a structure:

```python
import numpy as np
from batcharray.utils import dfs_array

def validate_structure(data):
    """Validate all arrays in a nested structure."""
    issues = []
    
    for i, arr in enumerate(dfs_array(data)):
        # Check for NaN values
        if np.isnan(arr).any():
            issues.append(f"Array {i} contains NaN values")
        
        # Check for infinite values
        if np.isinf(arr).any():
            issues.append(f"Array {i} contains infinite values")
        
        # Check for empty arrays
        if arr.size == 0:
            issues.append(f"Array {i} is empty")
    
    return issues

data = {
    "good": np.array([1, 2, 3]),
    "bad": {
        "nan_array": np.array([1, np.nan, 3]),
        "inf_array": np.array([1, np.inf, 3])
    }
}

issues = validate_structure(data)
for issue in issues:
    print(f"Warning: {issue}")
```

### Transformation with Context

Apply transformations while maintaining context:

```python
import numpy as np
from batcharray.utils import bfs_array

def normalize_all_arrays(data):
    """Normalize all arrays to [0, 1] range."""
    
    # First pass: find min and max across all arrays
    all_min = min(arr.min() for arr in bfs_array(data))
    all_max = max(arr.max() for arr in bfs_array(data))
    
    # Define normalization function
    def normalize(x):
        if isinstance(x, np.ndarray):
            return (x - all_min) / (all_max - all_min + 1e-8)
        return x
    
    # Apply to structure (would need recursive_apply from recursive module)
    # This is just an example of gathering context first
    print(f"Normalizing with range [{all_min}, {all_max}]")

data = {
    "a": np.array([1, 2, 3]),
    "b": np.array([10, 20, 30])
}

normalize_all_arrays(data)
```

## Integration with Other Modules

The utils module works well with other batcharray modules:

```python
import numpy as np
from batcharray.utils import bfs_array
from batcharray.recursive import recursive_apply

# Find all arrays
data = {
    "features": np.array([1, 2, 3]),
    "nested": {
        "more": np.array([4, 5, 6])
    }
}

# Count arrays
num_arrays = sum(1 for _ in bfs_array(data))

# Transform all arrays
transformed = recursive_apply(data, lambda x: x * 2 if isinstance(x, np.ndarray) else x)

# Verify transformation
original_sum = sum(arr.sum() for arr in bfs_array(data))
transformed_sum = sum(arr.sum() for arr in bfs_array(transformed))
assert transformed_sum == original_sum * 2
```

## Best Practices

1. **Choose appropriate traversal**: Use BFS for wide structures, DFS for deep structures
2. **Generator efficiency**: Both functions return generators, so they're memory-efficient
3. **Combine with filters**: Use list comprehensions or generator expressions to filter results
4. **Type checking**: Always check `isinstance(x, np.ndarray)` when processing mixed structures
5. **Immutability**: These functions don't modify data; use with `recursive_apply` for transformations

## Common Patterns

### Debugging Structure

```python
import numpy as np
from batcharray.utils import bfs_array

def debug_structure(data):
    """Print information about all arrays in a structure."""
    arrays = list(bfs_array(data))
    print(f"Total arrays: {len(arrays)}")
    
    for i, arr in enumerate(arrays):
        print(f"Array {i}: shape={arr.shape}, dtype={arr.dtype}, "
              f"min={arr.min():.3f}, max={arr.max():.3f}")

data = {
    "features": np.random.randn(100, 10),
    "labels": np.random.randint(0, 5, 100)
}

debug_structure(data)
```

### Memory Analysis

```python
import numpy as np
from batcharray.utils import dfs_array

def analyze_memory(data):
    """Analyze memory usage of nested structure."""
    arrays = list(dfs_array(data))
    
    total_bytes = sum(arr.nbytes for arr in arrays)
    by_dtype = {}
    
    for arr in arrays:
        dtype = str(arr.dtype)
        by_dtype[dtype] = by_dtype.get(dtype, 0) + arr.nbytes
    
    print(f"Total memory: {total_bytes / 1024**2:.2f} MB")
    print("By dtype:")
    for dtype, bytes in by_dtype.items():
        print(f"  {dtype}: {bytes / 1024**2:.2f} MB")

data = {
    "int_data": np.random.randint(0, 100, 10000, dtype=np.int32),
    "float_data": np.random.randn(10000).astype(np.float64)
}

analyze_memory(data)
```
