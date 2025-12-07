"""Working with nested data structures.

This example demonstrates how to manipulate dictionaries and lists
of arrays using the batcharray.nested module.
"""

import numpy as np
from batcharray import nested

print("=" * 60)
print("Nested Data Structures Example")
print("=" * 60)

# Create a nested batch structure
print("\n1. Creating nested batch data")
print("-" * 40)
batch = {
    "features": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
    "labels": np.array([0, 1, 0, 1, 0]),
    "weights": np.array([1.0, 2.0, 1.5, 2.5, 1.0]),
    "metadata": {
        "ids": np.array([100, 101, 102, 103, 104]),
        "timestamps": np.array([1, 2, 3, 4, 5])
    }
}

print("Batch structure:")
for key, value in batch.items():
    if isinstance(value, dict):
        print(f"  {key}: (nested dict)")
        for k, v in value.items():
            print(f"    {k}: shape {v.shape}")
    else:
        print(f"  {key}: shape {value.shape}")

# Slicing all arrays together
print("\n2. Slicing nested structure")
print("-" * 40)
sliced = nested.slice_along_batch(batch, stop=3)
print(f"Sliced batch (first 3 items):")
print(f"  features shape: {sliced['features'].shape}")
print(f"  labels: {sliced['labels']}")
print(f"  metadata.ids: {sliced['metadata']['ids']}")

# Splitting into multiple batches
print("\n3. Splitting into batches")
print("-" * 40)
batches = nested.split_along_batch(batch, split_size_or_sections=2)
print(f"Number of mini-batches: {len(batches)}")
for i, mini_batch in enumerate(batches):
    print(f"  Batch {i}: features shape {mini_batch['features'].shape}")

# Statistical operations
print("\n4. Statistical operations on nested data")
print("-" * 40)
mean_vals = nested.mean_along_batch(batch)
print(f"Mean features: {mean_vals['features']}")
print(f"Mean weights: {mean_vals['weights']}")

max_vals = nested.amax_along_batch(batch)
print(f"Max features: {max_vals['features']}")

# Shuffling while maintaining relationships
print("\n5. Shuffling (maintaining relationships)")
print("-" * 40)
np.random.seed(42)
shuffled = nested.shuffle_along_batch(batch)
print("After shuffling:")
print(f"  Labels: {shuffled['labels']}")
print(f"  IDs: {shuffled['metadata']['ids']}")
print("Note: Labels and IDs remain synchronized!")

# Concatenating batches
print("\n6. Concatenating batches")
print("-" * 40)
batch1 = {
    "x": np.array([[1, 2], [3, 4]]),
    "y": np.array([0, 1])
}
batch2 = {
    "x": np.array([[5, 6], [7, 8]]),
    "y": np.array([1, 0])
}
combined = nested.concatenate_along_batch([batch1, batch2])
print(f"Combined x shape: {combined['x'].shape}")
print(f"Combined x:\n{combined['x']}")
print(f"Combined y: {combined['y']}")

# Mathematical operations
print("\n7. Mathematical operations")
print("-" * 40)
data = {
    "values": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    "scores": np.array([[-1.0, -2.0], [3.0, 4.0]])
}

absolute = nested.abs(data)
print(f"Absolute values:")
print(f"  values: {absolute['values'][0]}")
print(f"  scores: {absolute['scores'][0]}")

exponential = nested.exp(data)
print(f"Exponential:")
print(f"  values (first row): {exponential['values'][0]}")

# Selection operations
print("\n8. Index selection")
print("-" * 40)
batch = {
    "data": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    "mask": np.array([True, False, True])
}
indices = np.array([0, 2])
selected = nested.index_select_along_batch(batch, indices=indices)
print(f"Selected data:\n{selected['data']}")
print(f"Selected mask: {selected['mask']}")

# Sequence operations on nested data
print("\n9. Sequence operations")
print("-" * 40)
sequences = {
    "inputs": np.array([
        [[1, 2], [3, 4], [5, 6]],
        [[7, 8], [9, 10], [11, 12]]
    ]),
    "targets": np.array([
        [[0], [1], [0]],
        [[1], [1], [0]]
    ])
}
print(f"Original sequence lengths: {sequences['inputs'].shape[1]}")

sliced_seq = nested.slice_along_seq(sequences, start=1)
print(f"Sliced sequences (timesteps 1-2):")
print(f"  inputs shape: {sliced_seq['inputs'].shape}")

chunked = nested.chunk_along_seq(sequences, chunks=3)
print(f"Chunked into {len(chunked)} sequence chunks")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)
