# Recursive Operations

The `batcharray.recursive` module provides functionality to recursively apply functions to nested data structures, handling complex hierarchies of dictionaries, lists, tuples, and other containers.

## Overview

When working with deeply nested data structures, you often need to apply the same transformation to all values at a certain depth or of a certain type. The recursive module makes this easy by traversing the structure and applying your function intelligently.

## Basic Usage

### Recursive Apply

The main entry point is `recursive_apply`:

```python
from batcharray.recursive import recursive_apply

# Simple transformation
data = {"a": 1, "b": 2, "c": 3}
result = recursive_apply(data, lambda x: x * 2)
# Result: {"a": 2, "b": 4, "c": 6}
```

### Nested Structures

Works with deeply nested structures:

```python
from batcharray.recursive import recursive_apply

data = {
    "level1": {
        "level2a": [1, 2, 3],
        "level2b": {"level3": 4}
    },
    "other": 5
}

# Convert all numbers to strings
result = recursive_apply(data, str)
# Result: {
#     "level1": {
#         "level2a": ["1", "2", "3"],
#         "level2b": {"level3": "4"}
#     },
#     "other": "5"
# }
```

## Working with NumPy Arrays

A common use case is applying functions to all NumPy arrays in a nested structure:

```python
import numpy as np
from batcharray.recursive import recursive_apply

data = {
    "features": {
        "image": np.array([[1, 2], [3, 4]]),
        "embedding": np.array([0.1, 0.2, 0.3])
    },
    "label": np.array([1, 0])
}

# Apply function only to arrays
def double_arrays(x):
    if isinstance(x, np.ndarray):
        return x * 2
    return x

result = recursive_apply(data, double_arrays)
# All arrays are doubled, structure is preserved
```

## Applier Classes

The module provides several applier classes for different data types:

### DefaultApplier

Handles basic Python objects (strings, numbers, etc.):

```python
from batcharray.recursive import DefaultApplier

applier = DefaultApplier()

# Apply to simple values
result = applier.apply(data=42, func=lambda x: x * 2, state=...)
# Result: 84
```

### MappingApplier

Handles dictionaries and other mapping types:

```python
from batcharray.recursive import MappingApplier, ApplyState, AutoApplier

applier = MappingApplier()
state = ApplyState(AutoApplier())

data = {"a": 1, "b": 2, "c": 3}
result = applier.apply(data=data, func=lambda x: x * 2, state=state)
# Result: {"a": 2, "b": 4, "c": 6}
```

### SequenceApplier

Handles lists, tuples, and other sequences:

```python
from batcharray.recursive import SequenceApplier, ApplyState, AutoApplier

applier = SequenceApplier()
state = ApplyState(AutoApplier())

data = [1, 2, 3, 4, 5]
result = applier.apply(data=data, func=lambda x: x * 2, state=state)
# Result: [2, 4, 6, 8, 10]
```

### AutoApplier

Automatically selects the appropriate applier based on data type:

```python
from batcharray.recursive import AutoApplier, ApplyState

applier = AutoApplier()
state = ApplyState(applier)

# Works with different types
dict_result = applier.apply({"a": 1}, lambda x: x * 2, state)
list_result = applier.apply([1, 2, 3], lambda x: x * 2, state)
value_result = applier.apply(42, lambda x: x * 2, state)
```

## Advanced Usage

### Custom Transformations

Apply complex transformations based on data type:

```python
import numpy as np
from batcharray.recursive import recursive_apply

def smart_transform(x):
    if isinstance(x, np.ndarray):
        return x.mean()  # Compute mean for arrays
    elif isinstance(x, str):
        return x.upper()  # Uppercase for strings
    elif isinstance(x, (int, float)):
        return x ** 2  # Square for numbers
    return x

data = {
    "arrays": {
        "a": np.array([1, 2, 3]),
        "b": np.array([4, 5, 6])
    },
    "text": "hello",
    "number": 5
}

result = recursive_apply(data, smart_transform)
# Result: {
#     "arrays": {"a": 2.0, "b": 5.0},  # Array means
#     "text": "HELLO",
#     "number": 25
# }
```

### Type Filtering

Apply transformations only to specific types:

```python
import numpy as np
from batcharray.recursive import recursive_apply

def normalize_arrays(x):
    if isinstance(x, np.ndarray) and x.dtype in [np.float32, np.float64]:
        # Normalize float arrays
        return (x - x.mean()) / (x.std() + 1e-8)
    return x

data = {
    "features": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
    "indices": np.array([0, 1, 2, 3, 4], dtype=np.int32),
    "label": "class_A"
}

result = recursive_apply(data, normalize_arrays)
# Only float arrays are normalized, indices and strings are unchanged
```

### Conditional Transformations

Apply different transformations based on context:

```python
import numpy as np
from batcharray.recursive import recursive_apply

def conditional_transform(x):
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            # Reshape 1D arrays to 2D
            return x.reshape(-1, 1)
        elif x.ndim == 2:
            # Transpose 2D arrays
            return x.T
    return x

data = {
    "vector": np.array([1, 2, 3]),
    "matrix": np.array([[1, 2], [3, 4]])
}

result = recursive_apply(data, conditional_transform)
# "vector" becomes shape (3, 1), "matrix" becomes shape (2, 2) transposed
```

## Integration with Other Modules

The recursive module is used internally by the nested module:

```python
import numpy as np
from batcharray import nested

# These nested operations use recursive_apply internally
data = {
    "a": np.array([1, 2, 3]),
    "b": np.array([4, 5, 6])
}

# Slice operation uses recursive_apply to handle the dictionary
sliced = nested.slice_along_batch(data, stop=2)
```

## State Management

The `ApplyState` class tracks the state during recursive traversal:

```python
from batcharray.recursive import AutoApplier, ApplyState

applier = AutoApplier()
state = ApplyState(applier)

# State can be used to track depth, visited objects, etc.
data = {"level1": {"level2": [1, 2, 3]}}

def func_with_state(x):
    # You can use state information here if needed
    return x * 2

result = applier.apply(data, func_with_state, state)
```

## Custom Appliers

You can create custom appliers for special data types:

```python
from batcharray.recursive import BaseApplier, ApplyState
from typing import Any
from collections.abc import Callable

class CustomDataApplier(BaseApplier):
    """Applier for custom data structures."""
    
    def apply(self, data: Any, func: Callable, state: ApplyState) -> Any:
        # Check if this applier can handle the data
        if not self.can_apply(data):
            return func(data)
        
        # Custom transformation logic
        # For example, handle a custom class
        if isinstance(data, MyCustomClass):
            # Transform the custom class's data
            transformed_data = state.applier.apply(
                data.get_data(), 
                func, 
                state
            )
            return MyCustomClass(transformed_data)
        
        return func(data)
    
    def can_apply(self, data: Any) -> bool:
        """Check if this applier can handle the data."""
        return isinstance(data, MyCustomClass)
```

## Best Practices

1. **Pure functions**: Use pure functions (no side effects) with `recursive_apply` for predictable behavior
2. **Type checking**: Always check the type before transforming to avoid unexpected behavior
3. **Identity for unknown types**: Return the input unchanged for types you don't want to transform
4. **Performance**: For very deep structures, consider iterative approaches or limit recursion depth
5. **Immutability**: `recursive_apply` creates new structures; original data is not modified

## Common Use Cases

### Data Preprocessing

```python
import numpy as np
from batcharray.recursive import recursive_apply

def preprocess(x):
    if isinstance(x, np.ndarray) and x.dtype == np.float64:
        # Normalize to [0, 1]
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x

raw_data = {
    "train": {
        "images": np.random.randn(100, 28, 28),
        "labels": np.array([0, 1, 2] * 33 + [0])
    },
    "val": {
        "images": np.random.randn(20, 28, 28),
        "labels": np.array([0, 1, 2] * 6 + [0, 1])
    }
}

preprocessed = recursive_apply(raw_data, preprocess)
```

### Type Conversion

```python
import numpy as np
from batcharray.recursive import recursive_apply

def convert_to_float32(x):
    if isinstance(x, np.ndarray) and x.dtype != np.float32:
        return x.astype(np.float32)
    return x

data = {
    "features": np.array([1, 2, 3], dtype=np.int64),
    "weights": np.array([0.1, 0.2, 0.3], dtype=np.float64)
}

converted = recursive_apply(data, convert_to_float32)
# All arrays are now float32
```

### Structure Validation

```python
import numpy as np
from batcharray.recursive import recursive_apply

def validate_arrays(x):
    if isinstance(x, np.ndarray):
        if not np.isfinite(x).all():
            raise ValueError(f"Array contains non-finite values: {x}")
    return x

data = {
    "values": np.array([1.0, 2.0, 3.0]),
    "scores": np.array([0.5, 0.7, 0.9])
}

# Validates all arrays in the structure
validated = recursive_apply(data, validate_arrays)
```
