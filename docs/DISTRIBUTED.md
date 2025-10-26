# Distributed Training Guide

This guide covers DPA's distributed training capabilities, including rank-aware parameter generation, range splitting, and multi-process coordination.

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Quick Start](#quick-start)
- [Rank-Aware Generation](#rank-aware-generation)
- [Range Splitting](#range-splitting)
- [Multi-Process Coordination](#multi-process-coordination)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

DPA's distributed training support ensures that each rank in a distributed training setup receives unique, deterministic augmentation parameters while maintaining reproducibility across runs. This prevents data duplication and ensures optimal training efficiency.

### Key Benefits

- **Unique Parameters**: Each rank generates different parameters for the same sample_id
- **Deterministic**: Same rank always produces identical parameters across runs
- **Scalable**: Linear scaling across any number of ranks
- **Reproducible**: Consistent results regardless of distributed setup changes

## Core Concepts

### Rank-Aware Seeding

Each rank uses a unique seed derived from:
- Base seed (shared across all ranks)
- Sample ID
- Rank number
- World size

This ensures deterministic yet unique parameter generation per rank.

### Range Splitting

Automatically distributes samples across ranks:
- Even distribution when possible
- Handles remainder samples fairly
- Validates no overlap between ranks
- Supports any world size configuration

## Quick Start

### Basic Distributed Generation

```python
from src.distributed import gen_distributed_augmentation_params
from src.dpa import get_preset

config = get_preset("moderate")

# Generate parameters for rank 0 in a 4-rank setup
params = gen_distributed_augmentation_params(
    sample_id=42,
    rank=0,
    config=config,
    world_size=4
)

print(f"Rotation: {params['rotation']:.2f}")
print(f"Brightness: {params['brightness']:.3f}")
```

### Range-Based Processing

```python
from src.distributed import DistributedRangeSplitter, stream_distributed_augmentation_chain

total_samples = 1000
world_size = 4
rank = 0

# Get this rank's sample range
splitter = DistributedRangeSplitter(total_samples, world_size)
start_id, end_id = splitter.get_rank_range(rank)

print(f"Rank {rank} processes samples {start_id} to {end_id-1}")

# Stream parameters for this rank's range
param_stream = stream_distributed_augmentation_chain(
    num_samples=total_samples,
    rank=rank,
    world_size=world_size,
    config=config
)

for params in param_stream:
    # Process each parameter set
    pass
```

## Rank-Aware Generation

### Function: `gen_distributed_augmentation_params`

Generates unique augmentation parameters for each rank while maintaining determinism.

```python
def gen_distributed_augmentation_params(
    sample_id: int,
    rank: int,
    config: AugmentationConfig | None = None,
    base_seed: int = 0,
    world_size: int = 1
) -> dict[str, Any]
```

#### Parameters

- **sample_id**: Unique identifier for the sample
- **rank**: Process rank (0-based)
- **config**: Augmentation configuration (uses default if None)
- **base_seed**: Base seed for reproducibility
- **world_size**: Total number of processes

#### Example: Multi-Rank Comparison

```python
from src.distributed import gen_distributed_augmentation_params
from src.dpa import get_preset

config = get_preset("mild")
sample_id = 100
world_size = 3

print("Parameters for sample_id 100 across ranks:")
for rank in range(world_size):
    params = gen_distributed_augmentation_params(
        sample_id=sample_id,
        rank=rank,
        config=config,
        world_size=world_size
    )
    print(f"Rank {rank}: rotation={params['rotation']:.2f}, "
          f"brightness={params['brightness']:.3f}")
```

Output:
```
Parameters for sample_id 100 across ranks:
Rank 0: rotation=-15.02, brightness=0.933
Rank 1: rotation=23.47, brightness=1.129
Rank 2: rotation=-7.44, brightness=0.829
```

### Reproducibility Verification

```python
# Same rank should produce identical results
params1 = gen_distributed_augmentation_params(42, rank=1, config=config, world_size=4)
params2 = gen_distributed_augmentation_params(42, rank=1, config=config, world_size=4)

assert params1 == params2  # Always True
print("✓ Reproducibility verified!")
```

## Range Splitting

### Class: `DistributedRangeSplitter`

Handles automatic distribution of sample ranges across ranks.

```python
class DistributedRangeSplitter:
    def __init__(self, total_samples: int, world_size: int)
    def get_rank_range(self, rank: int) -> tuple[int, int]
    def get_all_ranges(self) -> list[tuple[int, int]]
    def validate_ranges(self) -> bool
```

#### Basic Usage

```python
from src.distributed import DistributedRangeSplitter

# Split 100 samples across 4 ranks
splitter = DistributedRangeSplitter(total_samples=100, world_size=4)

for rank in range(4):
    start, end = splitter.get_rank_range(rank)
    samples_count = end - start
    print(f"Rank {rank}: samples {start}-{end-1} ({samples_count} samples)")
```

Output:
```
Rank 0: samples 0-24 (25 samples)
Rank 1: samples 25-49 (25 samples)
Rank 2: samples 50-74 (25 samples)
Rank 3: samples 75-99 (25 samples)
```

#### Handling Uneven Distribution

```python
# 23 samples across 4 ranks
splitter = DistributedRangeSplitter(total_samples=23, world_size=4)

for rank in range(4):
    start, end = splitter.get_rank_range(rank)
    if end > start:
        print(f"Rank {rank}: samples {start}-{end-1} ({end-start} samples)")
    else:
        print(f"Rank {rank}: no samples assigned")
```

Output:
```
Rank 0: samples 0-5 (6 samples)
Rank 1: samples 6-11 (6 samples)
Rank 2: samples 12-17 (6 samples)
Rank 3: samples 18-22 (5 samples)
```

#### Validation

```python
# Verify ranges are valid
is_valid = splitter.validate_ranges()
print(f"Ranges valid: {is_valid}")

# Get all ranges at once
all_ranges = splitter.get_all_ranges()
print(f"All ranges: {all_ranges}")
```

## Multi-Process Coordination

### Streaming with Distributed Ranges

The `stream_distributed_augmentation_chain` function combines range splitting with parameter generation:

```python
def stream_distributed_augmentation_chain(
    num_samples: int,
    rank: int,
    world_size: int,
    config: AugmentationConfig | None = None,
    chunk_size: int = 1000,
    base_seed: int = 0
) -> Generator[dict[str, Any], None, None]
```

#### Example: Complete Distributed Workflow

```python
import multiprocessing as mp
from src.distributed import stream_distributed_augmentation_chain
from src.dpa import get_preset

def worker_process(rank, world_size, total_samples, config_dict, results_queue):
    """Worker process for distributed training simulation."""
    # Reconstruct config from dictionary
    config = AugmentationConfig(**config_dict)
    
    # Process this rank's samples
    processed_count = 0
    param_stream = stream_distributed_augmentation_chain(
        num_samples=total_samples,
        rank=rank,
        world_size=world_size,
        config=config
    )
    
    for params in param_stream:
        # Simulate processing
        processed_count += 1
    
    results_queue.put({
        'rank': rank,
        'processed': processed_count
    })

# Main process coordination
def run_distributed_training():
    world_size = 4
    total_samples = 1000
    config = get_preset("moderate")
    
    # Convert config to dictionary for multiprocessing
    config_dict = {
        'rotation_range': config.rotation_range,
        'brightness_range': config.brightness_range,
        'noise_range': config.noise_range,
        'scale_range': config.scale_range,
        'contrast_range': config.contrast_range,
        'augmentation_depth': config.augmentation_depth
    }
    
    # Launch worker processes
    results_queue = mp.Queue()
    processes = []
    
    for rank in range(world_size):
        p = mp.Process(
            target=worker_process,
            args=(rank, world_size, total_samples, config_dict, results_queue)
        )
        p.start()
        processes.append(p)
    
    # Collect results
    results = []
    for _ in range(world_size):
        result = results_queue.get()
        results.append(result)
    
    # Wait for completion
    for p in processes:
        p.join()
    
    # Analyze results
    total_processed = sum(r['processed'] for r in results)
    print(f"Total samples processed: {total_processed}/{total_samples}")
    
    for result in sorted(results, key=lambda x: x['rank']):
        print(f"Rank {result['rank']}: {result['processed']} samples")

if __name__ == "__main__":
    run_distributed_training()
```

## Best Practices

### 1. Consistent Configuration

Always use the same configuration across all ranks:

```python
# ✓ Good: Same config for all ranks
config = get_preset("moderate")

# ✗ Bad: Different configs per rank
config = get_preset("mild" if rank == 0 else "aggressive")
```

### 2. Proper Error Handling

```python
try:
    params = gen_distributed_augmentation_params(
        sample_id=sample_id,
        rank=rank,
        config=config,
        world_size=world_size
    )
except ValueError as e:
    print(f"Invalid parameters: {e}")
    # Handle error appropriately
```

### 3. Memory Management

For large datasets, use streaming to avoid memory issues:

```python
# ✓ Good: Streaming for large datasets
param_stream = stream_distributed_augmentation_chain(
    num_samples=1000000,  # Large dataset
    rank=rank,
    world_size=world_size,
    config=config,
    chunk_size=1000  # Process in chunks
)

# ✗ Bad: Loading all parameters at once
# params_list = list(param_stream)  # Don't do this for large datasets
```

### 4. Validation

Always validate your distributed setup:

```python
def validate_distributed_setup(total_samples, world_size):
    """Validate distributed training setup."""
    splitter = DistributedRangeSplitter(total_samples, world_size)
    
    # Check ranges are valid
    if not splitter.validate_ranges():
        raise ValueError("Invalid range configuration")
    
    # Check all samples are covered
    all_ranges = splitter.get_all_ranges()
    total_assigned = sum(end - start for start, end in all_ranges)
    
    if total_assigned != total_samples:
        raise ValueError(f"Sample count mismatch: {total_assigned} != {total_samples}")
    
    print(f"✓ Distributed setup validated: {total_samples} samples, {world_size} ranks")

# Use before starting training
validate_distributed_setup(total_samples=10000, world_size=8)
```

## Troubleshooting

### Common Issues

#### 1. Rank >= World Size Error

```python
# Error: rank (4) must be < world_size (4)
params = gen_distributed_augmentation_params(42, rank=4, world_size=4)

# Fix: Use 0-based ranking
params = gen_distributed_augmentation_params(42, rank=3, world_size=4)  # ✓
```

#### 2. Inconsistent Results

```python
# Problem: Different base_seed values
params1 = gen_distributed_augmentation_params(42, 0, config, base_seed=123)
params2 = gen_distributed_augmentation_params(42, 0, config, base_seed=456)
# params1 != params2

# Solution: Use consistent base_seed
base_seed = 12345
params1 = gen_distributed_augmentation_params(42, 0, config, base_seed=base_seed)
params2 = gen_distributed_augmentation_params(42, 0, config, base_seed=base_seed)
# params1 == params2 ✓
```

#### 3. Empty Ranges

```python
# Problem: More ranks than samples
splitter = DistributedRangeSplitter(total_samples=2, world_size=5)
start, end = splitter.get_rank_range(rank=4)
# start == end (empty range)

# Solution: Check for empty ranges
if end > start:
    # Process samples
    param_stream = stream_distributed_augmentation_chain(...)
else:
    print(f"Rank {rank} has no samples to process")
```

### Performance Optimization

#### 1. Chunk Size Tuning

```python
# For small datasets
param_stream = stream_distributed_augmentation_chain(
    num_samples=1000,
    chunk_size=100,  # Smaller chunks
    ...
)

# For large datasets
param_stream = stream_distributed_augmentation_chain(
    num_samples=1000000,
    chunk_size=10000,  # Larger chunks for efficiency
    ...
)
```

#### 2. Memory Monitoring

```python
import psutil

def monitor_memory_usage():
    """Monitor memory usage during distributed processing."""
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent:.1f}% "
          f"({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")

# Use during processing
for i, params in enumerate(param_stream):
    if i % 1000 == 0:  # Check every 1000 samples
        monitor_memory_usage()
    # Process params...
```

## Integration Examples

### PyTorch Distributed Training

```python
import torch
import torch.distributed as dist
from src.distributed import stream_distributed_augmentation_chain

def setup_distributed():
    """Initialize PyTorch distributed training."""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def get_distributed_dataloader(dataset_size, batch_size, config):
    """Create distributed dataloader with DPA augmentations."""
    rank, world_size = setup_distributed()
    
    # Get augmentation parameters for this rank
    param_stream = stream_distributed_augmentation_chain(
        num_samples=dataset_size,
        rank=rank,
        world_size=world_size,
        config=config
    )
    
    # Convert to list for indexing (if dataset_size is manageable)
    augmentation_params = list(param_stream)
    
    # Create custom dataset with augmentations
    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, aug_params):
            self.base_dataset = base_dataset
            self.aug_params = aug_params
        
        def __len__(self):
            return len(self.aug_params)
        
        def __getitem__(self, idx):
            data = self.base_dataset[idx]
            params = self.aug_params[idx]
            # Apply augmentations using params
            return augmented_data
    
    # Create distributed sampler
    dataset = AugmentedDataset(base_dataset, augmentation_params)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler
    )
```

### TensorFlow/Keras Integration

```python
import tensorflow as tf
from src.distributed import gen_distributed_augmentation_params

class DistributedAugmentationLayer(tf.keras.layers.Layer):
    """Custom Keras layer for distributed augmentation."""
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Get distributed training info
        strategy = tf.distribute.get_strategy()
        self.rank = strategy.extended.worker_devices.index(
            tf.config.experimental.get_device_name()
        )
        self.world_size = strategy.num_replicas_in_sync
    
    def call(self, inputs, sample_ids):
        """Apply distributed augmentations."""
        augmented_batch = []
        
        for i, sample_id in enumerate(sample_ids):
            params = gen_distributed_augmentation_params(
                sample_id=sample_id.numpy(),
                rank=self.rank,
                config=self.config,
                world_size=self.world_size
            )
            
            # Apply augmentations to inputs[i] using params
            augmented = self.apply_augmentations(inputs[i], params)
            augmented_batch.append(augmented)
        
        return tf.stack(augmented_batch)
```

## Advanced Topics

### Custom Seed Generation

```python
from src.distributed import RankAwareSeedGenerator

# Create custom seed generator
generator = RankAwareSeedGenerator(
    base_seed=42,
    world_size=8
)

# Generate seeds for specific rank/sample combinations
seed = generator.generate_seed(sample_id=100, rank=3)
print(f"Seed for sample 100, rank 3: {seed}")

# Use with custom random number generator
import random
random.seed(int(seed, 16))  # Convert hex to int
custom_value = random.random()
```

### Dynamic World Size

```python
def handle_dynamic_scaling(current_samples, old_world_size, new_world_size):
    """Handle dynamic scaling of distributed training."""
    
    # Redistribute samples with new world size
    old_splitter = DistributedRangeSplitter(current_samples, old_world_size)
    new_splitter = DistributedRangeSplitter(current_samples, new_world_size)
    
    print("Redistribution plan:")
    for rank in range(new_world_size):
        start, end = new_splitter.get_rank_range(rank)
        if end > start:
            print(f"New rank {rank}: samples {start}-{end-1}")
        else:
            print(f"New rank {rank}: no samples")
    
    return new_splitter

# Example: Scale from 4 to 6 ranks
new_splitter = handle_dynamic_scaling(
    current_samples=1000,
    old_world_size=4,
    new_world_size=6
)
```

This comprehensive guide covers all aspects of DPA's distributed training capabilities. For more examples, see the `examples/distributed_*.py` files in the repository.