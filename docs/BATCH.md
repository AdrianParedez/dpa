# Batch Processing Guide

This guide covers DPA's intelligent batch processing capabilities, including different batching strategies, memory management, and performance optimization techniques.

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Quick Start](#quick-start)
- [Batching Strategies](#batching-strategies)
- [Memory Management](#memory-management)
- [Performance Optimization](#performance-optimization)
- [Integration with Streaming](#integration-with-streaming)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

DPA's batch processing system provides intelligent strategies for processing augmentation parameters in batches, with automatic memory management and performance optimization. This is essential for handling large datasets efficiently while maintaining optimal resource utilization.

### Key Benefits

- **Multiple Strategies**: Sequential, round-robin, memory-optimized, and adaptive batching
- **Memory Management**: Dynamic batch sizing based on available memory
- **Performance Optimization**: Adaptive batch sizes for optimal throughput
- **Resource Monitoring**: Real-time memory and CPU usage tracking
- **Seamless Integration**: Works with streaming API and distributed training

## Core Concepts

### Batch Strategies

1. **Sequential**: Simple consecutive grouping of samples
2. **Round-Robin**: Distributes samples evenly across batches
3. **Memory-Optimized**: Adjusts batch size based on memory constraints
4. **Adaptive**: Learns optimal batch size based on performance feedback

### Memory Awareness

- **Dynamic Sizing**: Batch sizes adjust based on available memory
- **Safety Margins**: Prevents out-of-memory errors
- **Resource Monitoring**: Tracks memory usage in real-time
- **Limit Enforcement**: Respects user-defined memory limits

## Quick Start

### Basic Batch Processing

```python
from src.batch import BatchProcessor, BatchStrategy, BatchConfig
from src.dpa import stream_augmentation_chain, get_preset

# Create batch configuration
batch_config = BatchConfig(
    strategy=BatchStrategy.SEQUENTIAL,
    batch_size=10
)

# Create batch processor
processor = BatchProcessor(BatchStrategy.SEQUENTIAL, batch_config)

# Create parameter stream
config = get_preset("moderate")
param_stream = stream_augmentation_chain(100, config)

# Process in batches
for batch in processor.process_stream(param_stream):
    print(f"Processing batch of {len(batch)} samples")
    # Process each batch...
```

### Memory-Optimized Processing

```python
from src.batch import BatchStrategy, BatchConfig

# Memory-optimized configuration
batch_config = BatchConfig(
    strategy=BatchStrategy.MEMORY_OPTIMIZED,
    max_memory_mb=500,  # 500MB limit
    min_batch_size=5    # Minimum 5 samples per batch
)

processor = BatchProcessor(BatchStrategy.MEMORY_OPTIMIZED, batch_config)

# Process with memory monitoring
param_stream = stream_augmentation_chain(1000, config)

for batch in processor.process_with_memory_monitoring(param_stream):
    # Get memory info
    memory_info = processor._memory_batcher.monitor_memory_usage()
    print(f"Batch size: {len(batch)}, Memory: {memory_info['percent_used']:.1f}%")
    
    # Process batch...
```

## Batching Strategies

### 1. Sequential Strategy

Groups samples in consecutive order (0-4, 5-9, 10-14, etc.).

```python
from src.batch import BatchStrategy, BatchConfig, BatchProcessor

batch_config = BatchConfig(
    strategy=BatchStrategy.SEQUENTIAL,
    batch_size=8
)

processor = BatchProcessor(BatchStrategy.SEQUENTIAL, batch_config)
```

**Use Cases:**
- Simple processing workflows
- When order matters
- Debugging and testing

**Example:**
```python
# Input: samples 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
# Output batches:
# Batch 1: [0, 1, 2, 3, 4, 5, 6, 7]
# Batch 2: [8, 9]
```

### 2. Round-Robin Strategy

Distributes samples evenly across multiple batches.

```python
batch_config = BatchConfig(
    strategy=BatchStrategy.ROUND_ROBIN,
    batch_size=4  # Creates 4 batches
)

processor = BatchProcessor(BatchStrategy.ROUND_ROBIN, batch_config)
```

**Use Cases:**
- Load balancing across workers
- Parallel processing
- Even distribution requirements

**Example:**
```python
# Input: samples 0, 1, 2, 3, 4, 5, 6, 7
# Output batches (batch_size=4):
# Batch 1: [0, 4]
# Batch 2: [1, 5]
# Batch 3: [2, 6]
# Batch 4: [3, 7]
```

### 3. Memory-Optimized Strategy

Automatically adjusts batch size based on available memory.

```python
batch_config = BatchConfig(
    strategy=BatchStrategy.MEMORY_OPTIMIZED,
    max_memory_mb=1000,     # Maximum memory usage
    min_batch_size=1,       # Minimum batch size
    adaptive_sizing=True    # Enable dynamic adjustment
)

processor = BatchProcessor(BatchStrategy.MEMORY_OPTIMIZED, batch_config)
```

**Features:**
- **Dynamic Sizing**: Batch size changes based on memory pressure
- **Safety Margins**: Prevents out-of-memory errors
- **Monitoring**: Real-time memory usage tracking
- **Limits**: Respects minimum batch size constraints

**Example Usage:**
```python
param_stream = stream_augmentation_chain(5000, config)

for batch in processor.process_with_memory_monitoring(param_stream):
    memory_info = processor._memory_batcher.monitor_memory_usage()
    
    if memory_info['percent_used'] > 80:
        print("⚠️ High memory usage detected")
    
    # Process batch with current memory-optimized size
    process_batch(batch)
```

### 4. Adaptive Strategy

Learns optimal batch size based on performance feedback.

```python
batch_config = BatchConfig(
    strategy=BatchStrategy.ADAPTIVE,
    batch_size=16,          # Starting batch size
    adaptive_sizing=True    # Enable learning
)

processor = BatchProcessor(BatchStrategy.ADAPTIVE, batch_config)
```

**Features:**
- **Performance Learning**: Adjusts based on throughput
- **Dynamic Optimization**: Finds optimal batch size automatically
- **Feedback Loop**: Uses processing time to improve performance

**Example:**
```python
import time

param_stream = stream_augmentation_chain(1000, config)

for batch_num, batch in enumerate(processor.process_stream(param_stream)):
    start_time = time.time()
    
    # Process batch
    process_batch(batch)
    
    processing_time = time.time() - start_time
    throughput = len(batch) / processing_time
    
    print(f"Batch {batch_num}: {len(batch)} samples, "
          f"throughput: {throughput:.1f} samples/sec")
```

## Memory Management

### Memory-Aware Batcher

The `MemoryAwareBatcher` class provides intelligent memory management:

```python
from src.batch import MemoryAwareBatcher

# Create memory-aware batcher
batcher = MemoryAwareBatcher(
    max_memory_mb=2000,  # 2GB limit
    min_batch_size=5     # Minimum 5 samples
)

# Monitor current memory usage
memory_info = batcher.monitor_memory_usage()
print(f"Available memory: {memory_info['available_mb']} MB")
print(f"Memory usage: {memory_info['percent_used']:.1f}%")

# Calculate optimal batch size
sample_size_bytes = 1024  # 1KB per sample
optimal_size = batcher.calculate_optimal_batch_size(sample_size_bytes)
print(f"Optimal batch size: {optimal_size} samples")
```

### Dynamic Batch Size Adjustment

```python
def demonstrate_memory_adjustment():
    """Show how batch size adjusts to memory pressure."""
    batcher = MemoryAwareBatcher(max_memory_mb=500, min_batch_size=2)
    current_batch_size = 20
    
    # Simulate different memory conditions
    memory_scenarios = [
        {"percent_used": 30, "available_mb": 350},  # Low usage
        {"percent_used": 60, "available_mb": 200},  # Medium usage
        {"percent_used": 85, "available_mb": 75},   # High usage
        {"percent_used": 95, "available_mb": 25},   # Critical usage
    ]
    
    for scenario in memory_scenarios:
        new_size = batcher.adjust_batch_size(current_batch_size, scenario)
        
        print(f"Memory: {scenario['percent_used']}% used, "
              f"Available: {scenario['available_mb']}MB")
        print(f"Batch size: {current_batch_size} → {new_size}")
        
        current_batch_size = new_size
        print()

demonstrate_memory_adjustment()
```

### Memory Monitoring Context Manager

```python
from src.batch import memory_monitor

# Monitor memory usage during processing
with memory_monitor(max_memory_mb=1000) as monitor:
    param_stream = stream_augmentation_chain(10000, config)
    
    for batch in processor.process_stream(param_stream):
        # Check if approaching memory limit
        if monitor.get_usage_percent() > 90:
            print("⚠️ Approaching memory limit!")
            break
        
        process_batch(batch)

print(f"Peak memory usage: {monitor.get_peak_usage_mb():.1f} MB")
```

## Performance Optimization

### Batch Size Optimization

Find the optimal batch size for your workload:

```python
def find_optimal_batch_size(param_stream, batch_sizes_to_test):
    """Find the optimal batch size for maximum throughput."""
    results = {}
    
    for batch_size in batch_sizes_to_test:
        batch_config = BatchConfig(
            strategy=BatchStrategy.SEQUENTIAL,
            batch_size=batch_size
        )
        processor = BatchProcessor(BatchStrategy.SEQUENTIAL, batch_config)
        
        # Measure performance
        start_time = time.time()
        processed_samples = 0
        
        for batch in processor.process_stream(param_stream):
            # Simulate processing
            time.sleep(0.001 * len(batch))  # 1ms per sample
            processed_samples += len(batch)
        
        total_time = time.time() - start_time
        throughput = processed_samples / total_time
        
        results[batch_size] = throughput
        print(f"Batch size {batch_size}: {throughput:.1f} samples/sec")
    
    # Find best batch size
    best_size = max(results.keys(), key=lambda x: results[x])
    print(f"\nOptimal batch size: {best_size}")
    return best_size

# Test different batch sizes
config = get_preset("moderate")
param_stream = stream_augmentation_chain(1000, config)
optimal_size = find_optimal_batch_size(param_stream, [1, 5, 10, 20, 50])
```

### Strategy Performance Comparison

```python
def compare_strategies(num_samples=1000):
    """Compare performance of different batching strategies."""
    config = get_preset("moderate")
    
    strategies = [
        (BatchStrategy.SEQUENTIAL, {"batch_size": 20}),
        (BatchStrategy.ROUND_ROBIN, {"batch_size": 10}),
        (BatchStrategy.MEMORY_OPTIMIZED, {"max_memory_mb": 500, "min_batch_size": 5}),
        (BatchStrategy.ADAPTIVE, {"batch_size": 15, "adaptive_sizing": True}),
    ]
    
    results = {}
    
    for strategy, config_params in strategies:
        batch_config = BatchConfig(strategy=strategy, **config_params)
        processor = BatchProcessor(strategy, batch_config)
        
        # Measure performance
        start_time = time.time()
        param_stream = stream_augmentation_chain(num_samples, config)
        
        batch_count = 0
        for batch in processor.process_stream(param_stream):
            batch_count += 1
            time.sleep(0.001 * len(batch))  # Simulate processing
        
        total_time = time.time() - start_time
        throughput = num_samples / total_time
        
        results[strategy.value] = {
            'throughput': throughput,
            'batch_count': batch_count,
            'avg_batch_size': num_samples / batch_count
        }
    
    # Display results
    print("Strategy Performance Comparison:")
    print("-" * 50)
    for strategy, metrics in results.items():
        print(f"{strategy}:")
        print(f"  Throughput: {metrics['throughput']:.1f} samples/sec")
        print(f"  Batches: {metrics['batch_count']}")
        print(f"  Avg batch size: {metrics['avg_batch_size']:.1f}")
        print()

compare_strategies()
```

### Adaptive Performance Tuning

```python
class AdaptivePerformanceTuner:
    """Automatically tune batch processing performance."""
    
    def __init__(self, initial_batch_size=10):
        self.batch_size = initial_batch_size
        self.performance_history = []
        self.adjustment_factor = 1.2
    
    def process_with_tuning(self, param_stream, config):
        """Process stream with automatic performance tuning."""
        
        while True:
            batch_config = BatchConfig(
                strategy=BatchStrategy.ADAPTIVE,
                batch_size=self.batch_size,
                adaptive_sizing=True
            )
            processor = BatchProcessor(BatchStrategy.ADAPTIVE, batch_config)
            
            # Measure performance for current batch size
            start_time = time.time()
            processed_samples = 0
            
            batch_count = 0
            for batch in processor.process_stream(param_stream):
                # Process batch
                time.sleep(0.001 * len(batch))
                processed_samples += len(batch)
                batch_count += 1
                
                # Stop after processing some batches for measurement
                if batch_count >= 10:
                    break
            
            if processed_samples == 0:
                break
            
            elapsed_time = time.time() - start_time
            throughput = processed_samples / elapsed_time
            
            self.performance_history.append({
                'batch_size': self.batch_size,
                'throughput': throughput
            })
            
            print(f"Batch size {self.batch_size}: {throughput:.1f} samples/sec")
            
            # Adjust batch size based on performance
            if len(self.performance_history) > 1:
                prev_throughput = self.performance_history[-2]['throughput']
                
                if throughput > prev_throughput:
                    # Performance improved, try larger batch size
                    self.batch_size = int(self.batch_size * self.adjustment_factor)
                else:
                    # Performance degraded, try smaller batch size
                    self.batch_size = max(1, int(self.batch_size / self.adjustment_factor))
            else:
                # First measurement, try larger batch size
                self.batch_size = int(self.batch_size * self.adjustment_factor)
            
            # Stop if batch size becomes too large or performance plateaus
            if self.batch_size > 1000 or len(self.performance_history) > 10:
                break
        
        # Return best configuration
        best_config = max(self.performance_history, key=lambda x: x['throughput'])
        return best_config

# Use adaptive tuner
tuner = AdaptivePerformanceTuner(initial_batch_size=5)
config = get_preset("moderate")
param_stream = stream_augmentation_chain(5000, config)

best_config = tuner.process_with_tuning(param_stream, config)
print(f"Best configuration: batch_size={best_config['batch_size']}, "
      f"throughput={best_config['throughput']:.1f} samples/sec")
```

## Integration with Streaming

### Streaming + Batch Processing

Combine streaming API with batch processing for optimal memory usage:

```python
from src.dpa import stream_augmentation_chain
from src.batch import BatchProcessor, BatchStrategy, BatchConfig

def process_large_dataset_efficiently():
    """Process large dataset with streaming + batching."""
    
    # Configuration
    total_samples = 100000  # Large dataset
    config = get_preset("aggressive")
    
    # Create streaming + batch processing pipeline
    batch_config = BatchConfig(
        strategy=BatchStrategy.MEMORY_OPTIMIZED,
        max_memory_mb=1000,
        min_batch_size=10
    )
    processor = BatchProcessor(BatchStrategy.MEMORY_OPTIMIZED, batch_config)
    
    # Stream in chunks to manage memory
    chunk_size = 5000
    total_processed = 0
    
    for chunk_start in range(0, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)
        chunk_size_actual = chunk_end - chunk_start
        
        print(f"Processing chunk {chunk_start}-{chunk_end-1} ({chunk_size_actual} samples)")
        
        # Stream this chunk
        param_stream = stream_augmentation_chain(
            num_samples=chunk_size_actual,
            config=config,
            start_id=chunk_start
        )
        
        # Process chunk in batches
        chunk_processed = 0
        for batch in processor.process_stream(param_stream):
            # Process batch
            process_batch(batch)
            chunk_processed += len(batch)
        
        total_processed += chunk_processed
        print(f"  Processed {chunk_processed} samples in chunk")
    
    print(f"Total processed: {total_processed}/{total_samples}")

def process_batch(batch):
    """Simulate batch processing."""
    # Your actual processing logic here
    time.sleep(0.001 * len(batch))  # Simulate work

process_large_dataset_efficiently()
```

### Distributed + Batch Processing

Combine distributed training with batch processing:

```python
from src.distributed import stream_distributed_augmentation_chain
from src.batch import BatchProcessor, BatchStrategy, BatchConfig

def distributed_batch_processing(rank, world_size, total_samples):
    """Process distributed data with batching."""
    
    config = get_preset("moderate")
    
    # Create batch processor
    batch_config = BatchConfig(
        strategy=BatchStrategy.MEMORY_OPTIMIZED,
        max_memory_mb=500,
        min_batch_size=5
    )
    processor = BatchProcessor(BatchStrategy.MEMORY_OPTIMIZED, batch_config)
    
    # Get distributed parameter stream
    param_stream = stream_distributed_augmentation_chain(
        num_samples=total_samples,
        rank=rank,
        world_size=world_size,
        config=config
    )
    
    # Process in batches
    processed_batches = 0
    processed_samples = 0
    
    for batch in processor.process_stream(param_stream):
        # Process batch for this rank
        process_batch(batch)
        
        processed_batches += 1
        processed_samples += len(batch)
        
        if processed_batches % 10 == 0:
            memory_info = processor._memory_batcher.monitor_memory_usage()
            print(f"Rank {rank}: {processed_batches} batches, "
                  f"{processed_samples} samples, "
                  f"memory: {memory_info['percent_used']:.1f}%")
    
    print(f"Rank {rank} completed: {processed_samples} samples in {processed_batches} batches")

# Simulate distributed processing
for rank in range(4):  # 4 ranks
    distributed_batch_processing(rank=rank, world_size=4, total_samples=10000)
```

## Best Practices

### 1. Choose the Right Strategy

```python
# For simple, ordered processing
strategy = BatchStrategy.SEQUENTIAL

# For parallel processing with load balancing
strategy = BatchStrategy.ROUND_ROBIN

# For memory-constrained environments
strategy = BatchStrategy.MEMORY_OPTIMIZED

# For performance optimization
strategy = BatchStrategy.ADAPTIVE
```

### 2. Set Appropriate Memory Limits

```python
import psutil

# Get system memory
total_memory_gb = psutil.virtual_memory().total / (1024**3)

# Use 50-70% of available memory for batch processing
max_memory_mb = int(total_memory_gb * 0.6 * 1024)

batch_config = BatchConfig(
    strategy=BatchStrategy.MEMORY_OPTIMIZED,
    max_memory_mb=max_memory_mb,
    min_batch_size=5
)
```

### 3. Monitor Performance

```python
def monitor_batch_performance(processor, param_stream):
    """Monitor batch processing performance."""
    
    batch_times = []
    batch_sizes = []
    memory_usages = []
    
    for batch in processor.process_stream(param_stream):
        start_time = time.time()
        
        # Process batch
        process_batch(batch)
        
        # Record metrics
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        batch_sizes.append(len(batch))
        
        # Monitor memory if available
        if hasattr(processor, '_memory_batcher'):
            memory_info = processor._memory_batcher.monitor_memory_usage()
            memory_usages.append(memory_info['percent_used'])
    
    # Analyze performance
    avg_batch_size = sum(batch_sizes) / len(batch_sizes)
    avg_batch_time = sum(batch_times) / len(batch_times)
    throughput = avg_batch_size / avg_batch_time
    
    print(f"Performance Summary:")
    print(f"  Average batch size: {avg_batch_size:.1f}")
    print(f"  Average batch time: {avg_batch_time:.3f}s")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    
    if memory_usages:
        avg_memory = sum(memory_usages) / len(memory_usages)
        print(f"  Average memory usage: {avg_memory:.1f}%")
```

### 4. Error Handling

```python
from src.batch import BatchProcessingError, MemoryLimitExceededError

def robust_batch_processing(processor, param_stream):
    """Robust batch processing with error handling."""
    
    try:
        for batch in processor.process_stream(param_stream):
            try:
                process_batch(batch)
            except MemoryLimitExceededError:
                print("Memory limit exceeded, reducing batch size")
                # Reduce batch size and continue
                processor.config.batch_size = max(1, processor.config.batch_size // 2)
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Log error and continue with next batch
                continue
                
    except BatchProcessingError as e:
        print(f"Batch processing error: {e}")
        # Handle batch processing specific errors
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Handle other errors
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors

```python
# Problem: Batch size too large for available memory
batch_config = BatchConfig(
    strategy=BatchStrategy.SEQUENTIAL,
    batch_size=10000  # Too large!
)

# Solution: Use memory-optimized strategy
batch_config = BatchConfig(
    strategy=BatchStrategy.MEMORY_OPTIMIZED,
    max_memory_mb=1000,  # Set appropriate limit
    min_batch_size=1
)
```

#### 2. Poor Performance

```python
# Problem: Batch size too small
batch_config = BatchConfig(
    strategy=BatchStrategy.SEQUENTIAL,
    batch_size=1  # Too small, high overhead
)

# Solution: Use adaptive strategy or optimize batch size
batch_config = BatchConfig(
    strategy=BatchStrategy.ADAPTIVE,
    batch_size=50,  # Better starting point
    adaptive_sizing=True
)
```

#### 3. Uneven Batch Sizes

```python
# Problem: Last batch is much smaller
# 100 samples with batch_size=30 → batches: [30, 30, 30, 10]

# Solution: Use round-robin for more even distribution
batch_config = BatchConfig(
    strategy=BatchStrategy.ROUND_ROBIN,
    batch_size=4  # Creates 4 more evenly sized batches
)
```

### Performance Debugging

```python
def debug_batch_performance():
    """Debug batch processing performance issues."""
    
    config = get_preset("moderate")
    param_stream = stream_augmentation_chain(1000, config)
    
    # Test different configurations
    configs_to_test = [
        ("Small Sequential", BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=5)),
        ("Large Sequential", BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=50)),
        ("Memory Optimized", BatchConfig(strategy=BatchStrategy.MEMORY_OPTIMIZED, max_memory_mb=500)),
        ("Adaptive", BatchConfig(strategy=BatchStrategy.ADAPTIVE, batch_size=20, adaptive_sizing=True)),
    ]
    
    for name, batch_config in configs_to_test:
        processor = BatchProcessor(batch_config.strategy, batch_config)
        
        start_time = time.time()
        batch_count = 0
        total_samples = 0
        
        for batch in processor.process_stream(param_stream):
            batch_count += 1
            total_samples += len(batch)
            time.sleep(0.001 * len(batch))  # Simulate processing
        
        total_time = time.time() - start_time
        throughput = total_samples / total_time
        
        print(f"{name}:")
        print(f"  Batches: {batch_count}")
        print(f"  Avg batch size: {total_samples / batch_count:.1f}")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print()

debug_batch_performance()
```

### Memory Debugging

```python
def debug_memory_usage():
    """Debug memory usage during batch processing."""
    
    import psutil
    import gc
    
    config = get_preset("aggressive")
    
    # Monitor memory before processing
    initial_memory = psutil.virtual_memory().used / (1024**2)  # MB
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    batch_config = BatchConfig(
        strategy=BatchStrategy.MEMORY_OPTIMIZED,
        max_memory_mb=500,
        min_batch_size=1
    )
    processor = BatchProcessor(BatchStrategy.MEMORY_OPTIMIZED, batch_config)
    
    param_stream = stream_augmentation_chain(5000, config)
    
    batch_count = 0
    for batch in processor.process_stream(param_stream):
        batch_count += 1
        
        # Process batch
        process_batch(batch)
        
        # Monitor memory every 10 batches
        if batch_count % 10 == 0:
            current_memory = psutil.virtual_memory().used / (1024**2)
            memory_delta = current_memory - initial_memory
            
            print(f"Batch {batch_count}: {len(batch)} samples, "
                  f"memory delta: +{memory_delta:.1f} MB")
            
            # Force garbage collection if memory usage is high
            if memory_delta > 200:  # 200MB increase
                gc.collect()
                print("  Performed garbage collection")
    
    # Final memory check
    final_memory = psutil.virtual_memory().used / (1024**2)
    total_delta = final_memory - initial_memory
    print(f"Final memory delta: +{total_delta:.1f} MB")

debug_memory_usage()
```

This comprehensive guide covers all aspects of DPA's batch processing capabilities. For more examples, see the `examples/batch_*.py` files in the repository.