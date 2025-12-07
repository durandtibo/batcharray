"""Machine Learning data pipeline example.

This example demonstrates a complete ML data pipeline using batcharray,
including data loading, preprocessing, batching, shuffling, and splitting.
"""

import numpy as np
from batcharray import nested

print("=" * 60)
print("Machine Learning Data Pipeline Example")
print("=" * 60)

# Simulate loading data
print("\n1. Loading and preparing data")
print("-" * 40)
np.random.seed(42)

# Generate synthetic dataset (e.g., image classification)
n_samples = 1000
n_features = 784  # 28x28 images flattened
n_classes = 10

data = {
    "images": np.random.randn(n_samples, n_features).astype(np.float32),
    "labels": np.random.randint(0, n_classes, n_samples),
    "sample_weights": np.random.rand(n_samples).astype(np.float32)
}

print(f"Dataset size: {n_samples} samples")
print(f"Images shape: {data['images'].shape}")
print(f"Labels shape: {data['labels'].shape}")

# Data preprocessing
print("\n2. Preprocessing data")
print("-" * 40)

# Normalize images to [0, 1] range
images = data['images']
images_min = images.min()
images_max = images.max()
data['images'] = (images - images_min) / (images_max - images_min)
print(f"Images normalized to range [{data['images'].min():.3f}, {data['images'].max():.3f}]")

# Shuffle the dataset
print("\n3. Shuffling dataset")
print("-" * 40)
shuffled_data = nested.shuffle_along_batch(data)
print(f"Data shuffled. First 5 labels: {shuffled_data['labels'][:5]}")

# Split into train/validation/test sets
print("\n4. Splitting into train/val/test sets")
print("-" * 40)
train_size = 700
val_size = 150
test_size = 150

train_data = nested.slice_along_batch(shuffled_data, stop=train_size)
val_data = nested.slice_along_batch(shuffled_data, start=train_size, stop=train_size + val_size)
test_data = nested.slice_along_batch(shuffled_data, start=train_size + val_size)

print(f"Train set: {train_data['images'].shape[0]} samples")
print(f"Val set: {val_data['images'].shape[0]} samples")
print(f"Test set: {test_data['images'].shape[0]} samples")

# Create mini-batches for training
print("\n5. Creating mini-batches")
print("-" * 40)
batch_size = 32
n_batches = (train_size + batch_size - 1) // batch_size

train_batches = nested.split_along_batch(train_data, split_size_or_sections=batch_size)
print(f"Number of training mini-batches: {len(train_batches)}")
print(f"First batch size: {train_batches[0]['images'].shape[0]}")
print(f"Last batch size: {train_batches[-1]['images'].shape[0]}")

# Simulate training loop
print("\n6. Simulating training loop")
print("-" * 40)
n_epochs = 2

for epoch in range(n_epochs):
    print(f"\nEpoch {epoch + 1}/{n_epochs}")
    
    # Shuffle training data each epoch
    epoch_train_data = nested.shuffle_along_batch(train_data)
    epoch_batches = nested.split_along_batch(epoch_train_data, split_size_or_sections=batch_size)
    
    # Process batches
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(epoch_batches[:5]):  # Show first 5 batches
        # Simulate forward pass
        batch_images = batch['images']
        batch_labels = batch['labels']
        
        # Simulate loss computation
        batch_loss = np.random.rand()  # Placeholder
        epoch_loss += batch_loss
        
        if batch_idx < 3:  # Show first 3 batches
            print(f"  Batch {batch_idx + 1}: size={batch_images.shape[0]}, loss={batch_loss:.4f}")
    
    # Validation
    val_loss = np.random.rand()
    print(f"  Validation loss: {val_loss:.4f}")

# Data augmentation simulation
print("\n7. Data augmentation (simulation)")
print("-" * 40)

def augment_batch(batch):
    """Simulate data augmentation."""
    augmented = {}
    
    # Add noise to images
    augmented['images'] = batch['images'] + np.random.randn(*batch['images'].shape) * 0.01
    
    # Keep labels and weights unchanged
    augmented['labels'] = batch['labels']
    augmented['sample_weights'] = batch['sample_weights']
    
    return augmented

# Augment a batch
sample_batch = train_batches[0]
augmented_batch = augment_batch(sample_batch)
print(f"Original batch image mean: {sample_batch['images'].mean():.4f}")
print(f"Augmented batch image mean: {augmented_batch['images'].mean():.4f}")

# Test set evaluation
print("\n8. Test set evaluation")
print("-" * 40)
test_batches = nested.split_along_batch(test_data, split_size_or_sections=batch_size)
print(f"Test batches: {len(test_batches)}")

# Simulate predictions
all_predictions = []
for batch in test_batches:
    # Simulate model predictions
    batch_preds = np.random.randint(0, n_classes, batch['labels'].shape[0])
    all_predictions.append(batch_preds)

all_predictions = np.concatenate(all_predictions)
print(f"Total predictions: {len(all_predictions)}")

# Compute accuracy
accuracy = (all_predictions == test_data['labels']).mean()
print(f"Test accuracy: {accuracy:.2%}")

# Class-wise statistics
print("\n9. Computing class-wise statistics")
print("-" * 40)
for class_id in range(3):  # Show first 3 classes
    class_mask = test_data['labels'] == class_id
    class_count = class_mask.sum()
    class_accuracy = (all_predictions[class_mask] == class_id).mean() if class_count > 0 else 0
    print(f"Class {class_id}: {class_count} samples, accuracy: {class_accuracy:.2%}")

print("\n" + "=" * 60)
print("ML Pipeline Example completed successfully!")
print("=" * 60)
