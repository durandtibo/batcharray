# Computation Models

The `batcharray.computation` module provides a flexible computation abstraction that allows operations to work with different array types (regular arrays, masked arrays, etc.) through a common interface.

## Overview

Computation models abstract away the details of different array types, allowing you to write code that works with:

- Standard NumPy arrays
- NumPy masked arrays
- Future array types

The computation model automatically selects the appropriate implementation based on the input array type.

## Basic Usage

### Automatic Model Selection

The easiest way to use computation models is through the interface functions with `AutoComputationModel`:

```python
import numpy as np
from batcharray import computation

# Works with regular arrays
arr = np.array([[1, 2, 3], [4, 5, 6]])
max_val = computation.max(arr, axis=0)  # [4, 5, 6]

# Automatically works with masked arrays too
import numpy.ma as ma

masked_arr = ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 1, 0], [1, 0, 0]])
max_val = computation.max(masked_arr, axis=0)  # [4, --, 6]
```

### Available Operations

The computation module provides several common operations:

```python
import numpy as np
from batcharray import computation

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Statistical operations
max_vals = computation.max(data, axis=0)  # [7, 8, 9]
min_vals = computation.min(data, axis=0)  # [1, 2, 3]
mean_vals = computation.mean(data, axis=0)  # [4., 5., 6.]
median_vals = computation.median(data, axis=0)  # [4., 5., 6.]

# Indexing operations
max_indices = computation.argmax(data, axis=0)  # [2, 2, 2]
min_indices = computation.argmin(data, axis=0)  # [0, 0, 0]

# Sorting
sorted_data = computation.sort(data, axis=0)
sort_indices = computation.argsort(data, axis=0)

# Concatenation
other = np.array([[10, 11, 12]])
combined = computation.concatenate([data, other], axis=0)
```

## Computation Models

### ArrayComputationModel

Handles standard NumPy arrays:

```python
import numpy as np
from batcharray.computation import ArrayComputationModel

model = ArrayComputationModel()

arr = np.array([[1, 2, 3], [4, 5, 6]])

# Use model methods
max_val = model.max(arr, axis=0)
mean_val = model.mean(arr, axis=1)
sorted_arr = model.sort(arr, axis=0)
```

### MaskedArrayComputationModel

Handles NumPy masked arrays with special consideration for masked values:

```python
import numpy as np
import numpy.ma as ma
from batcharray.computation import MaskedArrayComputationModel

model = MaskedArrayComputationModel()

# Create masked array
masked_arr = ma.array(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    mask=[[False, True, False], [False, False, True], [True, False, False]],
)

# Operations handle masked values appropriately
max_val = model.max(masked_arr, axis=0)  # Ignores masked values
mean_val = model.mean(masked_arr, axis=0)  # Computes mean of non-masked values
```

### AutoComputationModel

Automatically selects the appropriate model based on input type:

```python
import numpy as np
import numpy.ma as ma
from batcharray.computation import AutoComputationModel

auto_model = AutoComputationModel()

# Works with regular arrays
regular_arr = np.array([[1, 2, 3], [4, 5, 6]])
result1 = auto_model.max(regular_arr, axis=0)

# Automatically switches to masked array handling
masked_arr = ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 1, 0], [1, 0, 0]])
result2 = auto_model.max(masked_arr, axis=0)
```

## Custom Computation Models

You can create custom computation models by extending `BaseComputationModel`:

```python
import numpy as np
from batcharray.computation import BaseComputationModel, register_computation_models


class CustomArrayComputationModel(BaseComputationModel):
    """Custom computation model for special array types."""

    def max(
        self, array: np.ndarray, axis: int | None = None, keepdims: bool = False
    ) -> np.ndarray:
        # Custom max implementation
        return np.amax(array, axis=axis, keepdims=keepdims)

    def min(
        self, array: np.ndarray, axis: int | None = None, keepdims: bool = False
    ) -> np.ndarray:
        # Custom min implementation
        return np.amin(array, axis=axis, keepdims=keepdims)

    # Implement other required methods...
```

## Working with Different Array Types

### Regular NumPy Arrays

```python
import numpy as np
from batcharray import computation

# Standard operations
data = np.random.randn(100, 10)
max_vals = computation.max(data, axis=0)
mean_vals = computation.mean(data, axis=0)
```

### Masked Arrays

Masked arrays are useful for handling missing or invalid data:

```python
import numpy as np
import numpy.ma as ma
from batcharray import computation

# Create data with some missing values
data = ma.array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    mask=[
        [False, True, False],  # 2nd value is masked
        [False, False, True],  # 3rd value is masked
        [True, False, False],
    ],  # 1st value is masked
)

# Operations automatically handle masked values
max_vals = computation.max(data, axis=0)
# Result: [4.0, 8.0, 6.0] - ignoring masked values

mean_vals = computation.mean(data, axis=0)
# Result: [2.5, 5.0, 4.5] - mean of non-masked values only
```

## Advanced Features

### Axis Operations

All operations support axis parameters:

```python
import numpy as np
from batcharray import computation

data = np.random.randn(4, 5, 6)

# Operate on different axes
max_0 = computation.max(data, axis=0)  # Shape: (5, 6)
max_1 = computation.max(data, axis=1)  # Shape: (4, 6)
max_all = computation.max(data, axis=None)  # Scalar
```

### Keepdims

Preserve dimensions after reduction:

```python
import numpy as np
from batcharray import computation

data = np.array([[1, 2, 3], [4, 5, 6]])

# Without keepdims
result1 = computation.max(data, axis=0)  # Shape: (3,)

# With keepdims
result2 = computation.max(data, axis=0, keepdims=True)  # Shape: (1, 3)
```

## Integration with Other Modules

Computation models integrate seamlessly with other batcharray modules:

```python
import numpy as np
import numpy.ma as ma
from batcharray import array, computation

# Create masked array batch
batch = ma.array(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mask=[[0, 1, 0], [1, 0, 1], [0, 0, 0]]
)

# Use array operations (they use computation models internally)
sliced = array.slice_along_batch(batch, stop=2)
max_vals = array.amax_along_batch(batch)  # Uses computation.max internally
```

## Common Patterns

### Data Validation

```python
import numpy as np
import numpy.ma as ma
from batcharray import computation

# Load data with potential invalid values
data = np.array([[1.0, -999.0, 3.0], [4.0, 5.0, -999.0]])

# Mask invalid values
masked_data = ma.masked_equal(data, -999.0)

# Compute statistics safely
mean = computation.mean(masked_data, axis=0)
max_val = computation.max(masked_data, axis=0)
```

### Batch Processing with Missing Data

```python
import numpy as np
import numpy.ma as ma
from batcharray import computation, array

# Batch with some missing values
batch = ma.array(
    np.random.randn(100, 50), mask=np.random.random((100, 50)) < 0.1  # 10% missing
)

# Process batch
batch_means = computation.mean(batch, axis=1)  # Mean per sample
sorted_batch = computation.sort(batch, axis=1)  # Sort each sample
```
