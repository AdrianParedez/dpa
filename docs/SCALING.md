# Scaling DPA for Large Models

This document covers considerations and optimizations for using DPA with large-scale machine learning projects.

## Current Capabilities

DPA works well for:
- Generating millions of deterministic augmentation parameters
- Reproducible training across multiple runs
- Small to medium-scale datasets and models
- Single-GPU and simple multi-GPU setups

## Limitations at Scale

### 1. Recursive Fibonacci

The current implementation uses recursion with caching:

```python
@lru_cache(maxsize=None)
def fib(n: int) -> int:
    if n == 0: return 0
    if n == 1: return 1
    return fib(n - 1) + fib(n - 2)
```

**Issue:** While memoization helps, recursive calls can be inefficient for very deep augmentation chains.

**Workaround:** Use smaller `augmentation_depth` values (default 10 is fine for most cases).

**Future:** Switch to iterative Fibonacci in v0.2.0.

### 2. Memory Usage

Generating all parameters upfront stores everything in memory:

```python
# This loads all parameters into memory
results = generate_augmentation_chain(1_000_000, config)
```

**Issue:** Large datasets can consume significant memory.

**Workaround:** Generate and save to JSON, then load on-demand:

```python
# Generate once and save
generate_augmentation_chain(1_000_000, config, save_path="augmentations.json")

# Load in batches during training
loaded = load_augmentation_chain("augmentations.json")
batch_params = loaded[start_idx:end_idx]
```

**Future:** Add streaming/generator API in v0.2.0 for on-demand generation.

### 3. Distributed Training

Multi-GPU and multi-node training need careful coordination to ensure different processes get different augmentations.

**Current Workaround:**

```python
def get_augmentation_params_distributed(
    sample_id: int,
    rank: int,
    world_size: int,
    config: AugmentationConfig
):
    # Create unique ID for each process
    unique_id = sample_id * world_size + rank
    return gen_augmentation_params(unique_id, config)
```

**Future:** Built-in distributed training support in v0.2.0.

## Best Practices for Large-Scale Use

### 1. Batch Generation and Caching

```python
from src.dpa import generate_augmentation_chain, save_augmentation_chain

# Generate once during data preparation
config = get_preset("moderate")
generate_augmentation_chain(
    num_samples=10_000_000,
    config=config,
    save_path="augmentations/train.json"
)

# Load during training as needed
from src.dpa import load_augmentation_chain
augmentations = load_augmentation_chain("augmentations/train.json")
```

### 2. On-Demand Generation (Recommended for Extreme Scale)

```python
from src.dpa import gen_augmentation_params

class OnDemandAugmentationLoader:
    def __init__(self, config):
        self.config = config
    
    def get_params(self, sample_id: int):
        return gen_augmentation_params(sample_id, self.config)

# Use in dataloader
loader = OnDemandAugmentationLoader(get_preset("moderate"))

for sample_id in range(dataset_size):
    params = loader.get_params(sample_id)  # Generated fresh each time
    # Apply augmentations...
```

### 3. Framework Integration

**PyTorch:**
```python
from torch.utils.data import Dataset
from src.dpa import gen_augmentation_params, get_preset

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, config):
        self.dataset = base_dataset
        self.config = config
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        params = gen_augmentation_params(idx, self.config)
        
        # Apply using params (implementation depends on your needs)
        augmented = apply_augmentations(image, params)
        return augmented, label
```

**TensorFlow:**
```python
import tensorflow as tf
from src.dpa import gen_augmentation_params, get_preset

def augmentation_fn(sample_id, image, label):
    params = gen_augmentation_params(sample_id, get_preset("moderate"))
    # Apply TensorFlow operations using params
    image = tf.image.rotate(image, params['rotation'] * 3.14159 / 180)
    return image, label

dataset = dataset.map(augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE)
```

## Performance Considerations

- **Parameter generation:** SHA256 hashing is very fast, typical bottleneck is I/O not computation
- **Memory:** Pre-generated augmentations file is typically 1-10MB per million samples (depends on precision)
- **Caching:** Pre-generate augmentations during data preparation, not during training

## Roadmap (v0.2.0)

Planned optimizations for large-scale use:

- Iterative Fibonacci implementation
- Streaming/generator API for memory-efficient generation
- Built-in distributed training support
- Batch processing utilities
- Performance benchmarking tools

## Questions or Issues?

Open an issue on GitHub if you encounter scaling problems or have specific use cases in mind.