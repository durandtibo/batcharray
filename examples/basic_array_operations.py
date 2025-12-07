"""Basic array operations with batcharray.

This example demonstrates fundamental operations on NumPy arrays
using the batcharray.array module.
"""

import numpy as np
from batcharray import array

print("=" * 60)
print("Basic Array Operations Example")
print("=" * 60)

# Create a batch of data
print("\n1. Creating a batch of data")
print("-" * 40)
batch = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
print(f"Original batch shape: {batch.shape}")
print(f"Batch:\n{batch}")

# Slicing operations
print("\n2. Slicing along batch dimension")
print("-" * 40)
sliced = array.slice_along_batch(batch, start=1, stop=4)
print(f"Sliced batch (indices 1-3):\n{sliced}")

# Splitting into chunks
print("\n3. Splitting into chunks")
print("-" * 40)
chunks = array.chunk_along_batch(batch, chunks=3)
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i} shape: {chunk.shape}")

# Statistical operations
print("\n4. Statistical reductions")
print("-" * 40)
mean_vals = array.mean_along_batch(batch)
max_vals = array.amax_along_batch(batch)
min_vals = array.amin_along_batch(batch)
sum_vals = array.sum_along_batch(batch)

print(f"Mean along batch: {mean_vals}")
print(f"Max along batch: {max_vals}")
print(f"Min along batch: {min_vals}")
print(f"Sum along batch: {sum_vals}")

# Sorting operations
print("\n5. Sorting operations")
print("-" * 40)
unsorted = np.array([[5, 2, 8], [1, 9, 3], [7, 4, 6]])
sorted_batch = array.sort_along_batch(unsorted)
print(f"Original:\n{unsorted}")
print(f"Sorted along batch:\n{sorted_batch}")

# Indexing operations
print("\n6. Indexing and selection")
print("-" * 40)
indices = np.array([0, 2, 4])
selected = array.index_select_along_batch(batch, indices=indices)
print(f"Selected indices {indices}:\n{selected}")

# Shuffling
print("\n7. Shuffling")
print("-" * 40)
np.random.seed(42)
shuffled = array.shuffle_along_batch(batch.copy())
print(f"Shuffled batch:\n{shuffled}")

# Sequence operations
print("\n8. Sequence operations")
print("-" * 40)
sequences = np.array([
    [[1, 2], [3, 4], [5, 6], [7, 8]],
    [[9, 10], [11, 12], [13, 14], [15, 16]]
])
print(f"Sequences shape: {sequences.shape} (batch=2, seq_len=4, features=2)")

sliced_seq = array.slice_along_seq(sequences, start=1, stop=3)
print(f"Sliced sequences (timesteps 1-2) shape: {sliced_seq.shape}")
print(f"Sliced:\n{sliced_seq}")

mean_seq = array.mean_along_seq(sequences)
print(f"Mean along sequence dimension:\n{mean_seq}")

# Cumulative operations
print("\n9. Cumulative operations")
print("-" * 40)
data = np.array([[1, 2, 3], [4, 5, 6]])
cumsum = array.cumsum_along_batch(data)
print(f"Original:\n{data}")
print(f"Cumulative sum along batch:\n{cumsum}")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)
