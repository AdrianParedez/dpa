# Usage Guide

This guide covers the essential usage patterns for DPA, from basic parameter generation to advanced distributed training and performance optimization.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Streaming API](#streaming-api)
- [Distributed Training](#distributed-training)
- [Batch Processing](#batch-processing)
- [Performance Benchmarking](#performance-benchmarking)
- [Advanced Integration](#advanced-integration)

## Basic Usage

```python
from dpa import generate_augmentation_chain

results = generate_augmentation_chain(10, verbose=True)
```

## Using Presets

Three presets are available: mild, moderate, aggressive.

```python
from dpa import generate_augmentation_chain, get_preset

config = get_preset("moderate")
results = generate_augmentation_chain(20, config)
```

The presets differ in their parameter ranges:

- **Mild**: Conservative ranges for subtle augmentation
- **Moderate**: Balanced ranges for standard augmentation
- **Aggressive**: Wide ranges for heavy augmentation

## Custom Configuration

Create your own augmentation configuration:

```python
from dpa import AugmentationConfig, generate_augmentation_chain

custom_config = AugmentationConfig(
    rotation_range=(-60, 60),
    brightness_range=(0.6, 1.4),
    noise_range=(0, 0.2),
    scale_range=(0.6, 1.4),
    contrast_range=(0.5, 1.5),
    augmentation_depth=15,
)

results = generate_augmentation_chain(50, custom_config)
```

## Save and Load

Generate augmentations and persist them to JSON:

```python
from dpa import generate_augmentation_chain, load_augmentation_chain

config = get_preset("moderate")
results = generate_augmentation_chain(
    100,
    config,
    save_path="augmentations.json"
)

loaded = load_augmentation_chain("augmentations.json")
```

The JSON file includes metadata, all augmentation parameters, and computed statistics.

## Statistics

Compute statistics for your augmentation chain:

```python
from dpa import load_augmentation_chain, compute_statistics

loaded = load_augmentation_chain("augmentations.json")
stats = compute_statistics(loaded)

for param, stat_dict in stats.items():
    print(f"{param}: mean={stat_dict['mean']:.3f}, stdev={stat_dict['stdev']:.3f}")
```

## Single Sample Augmentation

Generate augmentation parameters for a single sample:

```python
from dpa import gen_augmentation_params

params = gen_augmentation_params(seed_id=42)
print(params)
```

This generates a dictionary with rotation, brightness, noise, scale, contrast, and the underlying hash.

## Use Cases

**ML Training**: Generate consistent augmentations for reproducible experiments

```python
config = get_preset("moderate")
for epoch in range(num_epochs):
    augmentations = generate_augmentation_chain(num_samples, config)
    # Augmentations are identical across epochs
```

**Data Versioning**: Save augmentation chains with model checkpoints

```python
generate_augmentation_chain(
    num_samples,
    config,
    save_path=f"checkpoints/aug_v{version}.json"
)
```

**Cross-Team Collaboration**: Share augmentation strategies

```python
loaded = load_augmentation_chain("team_augmentations.json")
# Apply loaded augmentations to your data
```

## How It Works

DPA generates deterministic augmentations using this process:

1. Hash the sample ID with SHA256
2. Iteratively combine the hash with Fibonacci numbers and re-hash
3. Convert the final hash to a random seed
4. Generate random parameters within specified ranges

This ensures identical results for the same seed_id while maintaining statistical diversity across different samples.

## Streaming API

For large datasets, use the streaming API to avoid loading all parameters into memory:

### Basic Streaming

```python
from src.dpa import stream_augmentation_chain

config = get_preset("moderate")

# Stream parameters for 1 million samples
param_stream = stream_augmentation_chain(1000000, config, chunk_size=1000)

for params in param_stream:
    # Process each parameter set individually
    apply_augmentation(image, params)
```

### Range-Based Streaming

```python
from src.dpa import stream_augmentation_range

# Stream parameters for a specific range
param_stream = stream_augmentation_range(
    start_id=1000, 
    end_id=2000, 
    config=config
)

for params in param_stream:
    # Process parameters for samples 1000-1999
    process_sample(params)
```

### Streaming Statistics

```python
from src.dpa import compute_streaming_statistics

# Compute statistics without loading all data into memory
param_stream = stream_augmentation_chain(100000, config)
stats = compute_streaming_statistics(param_stream)

print(f"Rotation mean: {stats['rotation']['mean']:.2f}")
```

## Distributed Training

DPA provides rank-aware parameter generation for distributed training setups:

### Basic Distributed Usage

```python
from src.distributed import gen_distributed_augmentation_params

config = get_preset("moderate")

# Each rank gets unique parameters for the same sample_id
params_rank0 = gen_distributed_augmentation_params(
    sample_id=42, 
    rank=0, 
    config=config, 
    world_size=4
)

params_rank1 = gen_distributed_augmentation_params(
    sample_id=42, 
    rank=1, 
    config=config, 
    world_size=4
)

# params_rank0 != params_rank1 (different augmentations)
```

### Range Splitting

```python
from src.distributed import DistributedRangeSplitter, stream_distributed_augmentation_chain

total_samples = 10000
world_size = 4
rank = 0  # Current process rank

# Automatically split samples across ranks
splitter = DistributedRangeSplitter(total_samples, world_size)
start_id, end_id = splitter.get_rank_range(rank)

print(f"Rank {rank} processes samples {start_id} to {end_id-1}")

# Stream parameters for this rank's samples
param_stream = stream_distributed_augmentation_chain(
    num_samples=total_samples,
    rank=rank,
    world_size=world_size,
    config=config
)

for params in param_stream:
    # Process only this rank's assigned samples
    process_sample(params)
```

### PyTorch Integration

```python
import torch.distributed as dist
from src.distributed import stream_distributed_augmentation_chain

# Initialize PyTorch distributed training
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()

# Get augmentation parameters for this rank
config = get_preset("moderate")
param_stream = stream_distributed_augmentation_chain(
    num_samples=len(dataset),
    rank=rank,
    world_size=world_size,
    config=config
)

# Use in training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        # Get augmentation parameters for this batch
        batch_params = [next(param_stream) for _ in range(len(data))]
        
        # Apply augmentations
        augmented_data = apply_augmentations(data, batch_params)
        
        # Train with augmented data
        train_step(augmented_data, target)
```

## Batch Processing

Process parameters in intelligent batches for optimal performance:

### Sequential Batching

```python
from src.batch import BatchProcessor, BatchStrategy, BatchConfig
from src.dpa import stream_augmentation_chain

# Create batch configuration
batch_config = BatchConfig(
    strategy=BatchStrategy.SEQUENTIAL,
    batch_size=32
)

processor = BatchProcessor(BatchStrategy.SEQUENTIAL, batch_config)

# Process parameters in batches
config = get_preset("moderate")
param_stream = stream_augmentation_chain(1000, config)

for batch in processor.process_stream(param_stream):
    # Process 32 samples at once
    process_batch(batch)
```

### Memory-Optimized Batching

```python
# Automatically adjust batch size based on available memory
batch_config = BatchConfig(
    strategy=BatchStrategy.MEMORY_OPTIMIZED,
    max_memory_mb=2000,  # 2GB limit
    min_batch_size=8     # Minimum 8 samples per batch
)

processor = BatchProcessor(BatchStrategy.MEMORY_OPTIMIZED, batch_config)

param_stream = stream_augmentation_chain(50000, config)

for batch in processor.process_with_memory_monitoring(param_stream):
    # Batch size automatically adjusts based on memory usage
    memory_info = processor._memory_batcher.monitor_memory_usage()
    print(f"Processing {len(batch)} samples, memory: {memory_info['percent_used']:.1f}%")
    
    process_batch(batch)
```

### Adaptive Batching

```python
# Automatically optimize batch size for best performance
batch_config = BatchConfig(
    strategy=BatchStrategy.ADAPTIVE,
    batch_size=16,       # Starting batch size
    adaptive_sizing=True # Enable automatic optimization
)

processor = BatchProcessor(BatchStrategy.ADAPTIVE, batch_config)

param_stream = stream_augmentation_chain(10000, config)

for batch in processor.process_stream(param_stream):
    # Batch size adapts based on processing performance
    process_batch(batch)
```

## Performance Benchmarking

Measure and optimize performance with comprehensive benchmarking tools:

### Basic Performance Measurement

```python
from src.benchmark import measure_time, measure_memory

config = get_preset("moderate")

# Measure execution time
with measure_time() as timer:
    results = generate_augmentation_chain(10000, config)

print(f"Generated 10k samples in {timer['elapsed_seconds']:.2f}s")
print(f"Throughput: {10000 / timer['elapsed_seconds']:.1f} samples/sec")

# Measure memory usage
with measure_memory() as memory:
    results = generate_augmentation_chain(50000, config)

print(f"Memory usage: {memory['delta_mb']:.1f}MB for 50k samples")
```

### Performance Profiling

```python
from src.benchmark import PerformanceProfiler

profiler = PerformanceProfiler(enable_memory_tracking=True)

# Profile function execution
@profiler.profile_function
def generate_batch(size):
    return generate_augmentation_chain(size, config)

# Use profiled function
results = generate_batch(1000)

# Get profiling results
summary = profiler.get_profile_summary()
for operation, result in summary.items():
    print(f"{operation}: {result.avg_time_per_call_ms:.2f}ms avg")
```

### Comprehensive Benchmarking

```python
from src.benchmark import BenchmarkRunner, BenchmarkConfig

# Configure benchmarking
benchmark_config = BenchmarkConfig(
    iterations=50,
    warmup_iterations=5,
    measure_memory=True,
    measure_cpu=True
)

runner = BenchmarkRunner(benchmark_config)

# Benchmark different configurations
configs = {
    "mild": get_preset("mild"),
    "moderate": get_preset("moderate"),
    "aggressive": get_preset("aggressive")
}

for config_name, config in configs.items():
    metrics = runner.benchmark_generation(5000, config)
    
    print(f"{config_name}:")
    print(f"  Throughput: {metrics.throughput_samples_per_second:.1f} samples/sec")
    print(f"  Memory: {metrics.memory_usage_mb:.1f}MB")
    print(f"  CPU: {metrics.cpu_usage_percent:.1f}%")
```

## Advanced Integration

### Combined Distributed + Batch Processing

```python
from src.distributed import stream_distributed_augmentation_chain
from src.batch import BatchProcessor, BatchStrategy, BatchConfig

def distributed_batch_training(rank, world_size, total_samples):
    """Complete distributed training with batch processing."""
    
    config = get_preset("moderate")
    
    # Configure memory-optimized batching
    batch_config = BatchConfig(
        strategy=BatchStrategy.MEMORY_OPTIMIZED,
        max_memory_mb=1000,
        min_batch_size=16
    )
    processor = BatchProcessor(BatchStrategy.MEMORY_OPTIMIZED, batch_config)
    
    # Get distributed parameter stream
    param_stream = stream_distributed_augmentation_chain(
        num_samples=total_samples,
        rank=rank,
        world_size=world_size,
        config=config
    )
    
    # Process in memory-optimized batches
    for batch in processor.process_stream(param_stream):
        # Each rank processes its assigned samples in optimal batches
        train_on_batch(batch)

# Use in distributed training
for rank in range(world_size):
    distributed_batch_training(rank, world_size=4, total_samples=100000)
```

### Performance Optimization Workflow

```python
from src.benchmark import BenchmarkRunner, BenchmarkConfig
from src.batch import BatchStrategy, BatchConfig

def optimize_performance(dataset_size):
    """Systematic performance optimization."""
    
    benchmark_config = BenchmarkConfig(iterations=30, measure_memory=True)
    runner = BenchmarkRunner(benchmark_config)
    
    # Step 1: Find best augmentation configuration
    configs = {
        "lightweight": get_preset("mild"),
        "balanced": get_preset("moderate"),
        "intensive": get_preset("aggressive")
    }
    
    best_config = None
    best_throughput = 0
    
    for name, config in configs.items():
        metrics = runner.benchmark_generation(1000, config)
        if metrics.throughput_samples_per_second > best_throughput:
            best_throughput = metrics.throughput_samples_per_second
            best_config = config
    
    # Step 2: Optimize batch processing
    batch_strategies = [
        BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=32),
        BatchConfig(strategy=BatchStrategy.MEMORY_OPTIMIZED, max_memory_mb=2000),
        BatchConfig(strategy=BatchStrategy.ADAPTIVE, batch_size=16, adaptive_sizing=True)
    ]
    
    best_batch_config = None
    best_batch_throughput = 0
    
    for batch_config in batch_strategies:
        metrics = runner.benchmark_batch_processing(dataset_size, batch_config)
        if metrics.throughput_samples_per_second > best_batch_throughput:
            best_batch_throughput = metrics.throughput_samples_per_second
            best_batch_config = batch_config
    
    print(f"Optimized configuration:")
    print(f"  Best augmentation config: {best_throughput:.1f} samples/sec")
    print(f"  Best batch config: {best_batch_throughput:.1f} samples/sec")
    
    return best_config, best_batch_config

# Optimize for your dataset
optimal_config, optimal_batch = optimize_performance(dataset_size=50000)
```

For more detailed examples and advanced usage patterns, see the comprehensive guides:
- [Distributed Training Guide](DISTRIBUTED.md)
- [Batch Processing Guide](BATCH.md)
- [Benchmarking Guide](BENCHMARK.md)