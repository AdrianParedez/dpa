# API Reference

This document provides comprehensive API documentation for all DPA modules.

## Table of Contents

- [Core Module (`src.dpa`)](#core-module-srcdpa)
- [Distributed Training (`src.distributed`)](#distributed-training-srcdistributed)
- [Batch Processing (`src.batch`)](#batch-processing-srcbatch)
- [Benchmarking (`src.benchmark`)](#benchmarking-srcbenchmark)

## Core Module (`src.dpa`)

### AugmentationConfig

Dataclass for configuring augmentation parameter ranges.

```python
@dataclass
class AugmentationConfig:
    rotation_range: Tuple[float, float] = (-30, 30)
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    noise_range: Tuple[float, float] = (0, 0.1)
    scale_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    augmentation_depth: int = 10
```

## Functions

### generate_augmentation_chain

```python
generate_augmentation_chain(
    num_samples: int,
    config: Optional[AugmentationConfig] = None,
    verbose: bool = False,
    save_path: Optional[str] = None
) -> List[Dict[str, Any]]
```

Generate augmentation parameters for multiple samples.

**Parameters:**
- `num_samples`: Number of samples to generate
- `config`: Custom AugmentationConfig (uses defaults if None)
- `verbose`: Print results to stdout
- `save_path`: Path to save JSON file

**Returns:** List of augmentation parameter dictionaries

### gen_augmentation_params

```python
gen_augmentation_params(
    seed_id: int,
    config: Optional[AugmentationConfig] = None
) -> Dict[str, Any]
```

Generate augmentation parameters for a single sample.

**Parameters:**
- `seed_id`: Unique sample identifier
- `config`: Custom AugmentationConfig

**Returns:** Dictionary with augmentation parameters

### gen_augmentation_seed

```python
gen_augmentation_seed(seed_id: int, augmentation_depth: int = 10) -> str
```

Generate deterministic hash seed using Fibonacci chain.

**Parameters:**
- `seed_id`: Unique sample identifier
- `augmentation_depth`: Number of hash iterations

**Returns:** SHA256 hash as hexadecimal string

### save_augmentation_chain

```python
save_augmentation_chain(
    params_list: List[Dict[str, Any]],
    filepath: str,
    config: Optional[AugmentationConfig] = None,
    include_stats: bool = True
) -> None
```

Save augmentation chain to JSON file with optional statistics.

**Parameters:**
- `params_list`: Augmentation parameters to save
- `filepath`: Output file path
- `config`: Configuration to include in metadata
- `include_stats`: Include statistics in output

### load_augmentation_chain

```python
load_augmentation_chain(filepath: str) -> List[Dict[str, Any]]
```

Load augmentation chain from JSON file.

**Parameters:**
- `filepath`: Path to JSON file

**Returns:** List of augmentation parameter dictionaries

### compute_statistics

```python
compute_statistics(params_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]
```

Compute statistics for augmentation parameters.

**Parameters:**
- `params_list`: Augmentation parameters

**Returns:** Dictionary with mean, stdev, min, max for each parameter

### get_preset

```python
get_preset(preset_name: str) -> AugmentationConfig
```

Get a preset configuration.

**Parameters:**
- `preset_name`: "mild", "moderate", or "aggressive"

**Returns:** AugmentationConfig instance

### stream_augmentation_chain

```python
stream_augmentation_chain(
    num_samples: int,
    config: Optional[AugmentationConfig] = None,
    chunk_size: int = 1000,
    start_id: int = 0
) -> Generator[Dict[str, Any], None, None]
```

Generate augmentation parameters as a memory-efficient stream.

**Parameters:**
- `num_samples`: Number of samples to generate
- `config`: Custom AugmentationConfig (uses defaults if None)
- `chunk_size`: Number of samples to process in each chunk
- `start_id`: Starting sample ID (default: 0)

**Returns:** Generator yielding augmentation parameter dictionaries

### stream_augmentation_range

```python
stream_augmentation_range(
    start_id: int,
    end_id: int,
    config: Optional[AugmentationConfig] = None,
    chunk_size: int = 1000
) -> Generator[Dict[str, Any], None, None]
```

Generate augmentation parameters for a specific ID range.

**Parameters:**
- `start_id`: Starting sample ID (inclusive)
- `end_id`: Ending sample ID (exclusive)
- `config`: Custom AugmentationConfig (uses defaults if None)
- `chunk_size`: Number of samples to process in each chunk

**Returns:** Generator yielding augmentation parameter dictionaries

### compute_streaming_statistics

```python
compute_streaming_statistics(
    param_generator: Generator[Dict[str, Any], None, None]
) -> Dict[str, Dict[str, float]]
```

Compute statistics from a parameter generator without loading all data into memory.

**Parameters:**
- `param_generator`: Generator yielding augmentation parameters

**Returns:** Dictionary with mean, stdev, min, max for each parameter

## Distributed Training (`src.distributed`)

### gen_distributed_augmentation_params

```python
gen_distributed_augmentation_params(
    sample_id: int,
    rank: int,
    config: Optional[AugmentationConfig] = None,
    base_seed: int = 0,
    world_size: int = 1
) -> Dict[str, Any]
```

Generate rank-aware augmentation parameters for distributed training.

**Parameters:**
- `sample_id`: Unique identifier for the sample
- `rank`: Process rank (0-based)
- `config`: Augmentation configuration (uses default if None)
- `base_seed`: Base seed for reproducibility
- `world_size`: Total number of processes

**Returns:** Dictionary with augmentation parameters

### RankAwareSeedGenerator

```python
class RankAwareSeedGenerator:
    def __init__(self, base_seed: int = 0, world_size: int = 1)
    def generate_seed(self, sample_id: int, rank: int, augmentation_depth: int = 10) -> str
```

Generates rank-aware seeds for distributed training.

**Methods:**
- `generate_seed()`: Generate unique seed for rank/sample combination

### DistributedRangeSplitter

```python
class DistributedRangeSplitter:
    def __init__(self, total_samples: int, world_size: int)
    def get_rank_range(self, rank: int) -> Tuple[int, int]
    def get_all_ranges(self) -> List[Tuple[int, int]]
    def validate_ranges(self) -> bool
```

Splits sample ranges across distributed training ranks.

**Methods:**
- `get_rank_range()`: Get start/end range for specific rank
- `get_all_ranges()`: Get all rank ranges
- `validate_ranges()`: Validate range configuration

### stream_distributed_augmentation_chain

```python
stream_distributed_augmentation_chain(
    num_samples: int,
    rank: int,
    world_size: int,
    config: Optional[AugmentationConfig] = None,
    chunk_size: int = 1000,
    base_seed: int = 0
) -> Generator[Dict[str, Any], None, None]
```

Stream augmentation parameters for a specific rank in distributed training.

**Parameters:**
- `num_samples`: Total number of samples across all ranks
- `rank`: Current process rank
- `world_size`: Total number of processes
- `config`: Augmentation configuration
- `chunk_size`: Chunk size for streaming
- `base_seed`: Base seed for reproducibility

**Returns:** Generator yielding parameters for this rank's samples

## Batch Processing (`src.batch`)

### BatchStrategy

```python
class BatchStrategy(Enum):
    SEQUENTIAL = "sequential"
    ROUND_ROBIN = "round_robin"
    MEMORY_OPTIMIZED = "memory_optimized"
    ADAPTIVE = "adaptive"
```

Enumeration of available batch processing strategies.

### BatchConfig

```python
@dataclass
class BatchConfig:
    strategy: BatchStrategy
    batch_size: Optional[int] = None
    max_memory_mb: int = 1000
    min_batch_size: int = 1
    adaptive_sizing: bool = False
```

Configuration for batch processing operations.

### BatchProcessor

```python
class BatchProcessor:
    def __init__(self, strategy: BatchStrategy, config: BatchConfig)
    def process_stream(self, param_generator: Generator) -> Generator[List[Dict], None, None]
    def process_with_memory_monitoring(self, param_generator: Generator) -> Generator[List[Dict], None, None]
```

Main batch processor for handling different batching strategies.

**Methods:**
- `process_stream()`: Process parameter stream in batches
- `process_with_memory_monitoring()`: Process with memory monitoring

### MemoryAwareBatcher

```python
class MemoryAwareBatcher:
    def __init__(self, max_memory_mb: int = 1000, min_batch_size: int = 1)
    def monitor_memory_usage(self) -> Dict[str, int]
    def calculate_optimal_batch_size(self, sample_size_bytes: int) -> int
    def adjust_batch_size(self, current_size: int, memory_usage: Dict[str, int]) -> int
```

Memory-aware batch size calculator and monitor.

**Methods:**
- `monitor_memory_usage()`: Get current memory usage information
- `calculate_optimal_batch_size()`: Calculate optimal batch size for memory constraints
- `adjust_batch_size()`: Adjust batch size based on memory usage

## Benchmarking (`src.benchmark`)

### PerformanceProfiler

```python
class PerformanceProfiler:
    def __init__(self, enable_memory_tracking: bool = False)
    def start_profiling(self, operation_name: str) -> None
    def end_profiling(self, operation_name: str) -> ProfileResult
    def profile_function(self, func: Callable) -> Callable
    def profile_context(self, operation_name: str) -> ContextManager
    def get_profile_summary(self) -> Dict[str, ProfileResult]
```

Comprehensive performance profiling with timing and memory tracking.

**Methods:**
- `start_profiling()` / `end_profiling()`: Manual profiling control
- `profile_function()`: Decorator for automatic function profiling
- `profile_context()`: Context manager for block profiling
- `get_profile_summary()`: Get summary of all profiling results

### BenchmarkRunner

```python
class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig)
    def benchmark_generation(self, num_samples: int, config: AugmentationConfig) -> PerformanceMetrics
    def benchmark_streaming(self, num_samples: int, config: AugmentationConfig) -> PerformanceMetrics
    def benchmark_batch_processing(self, num_samples: int, batch_config: BatchConfig) -> PerformanceMetrics
```

Comprehensive benchmark runner for performance analysis.

**Methods:**
- `benchmark_generation()`: Benchmark parameter generation
- `benchmark_streaming()`: Benchmark streaming operations
- `benchmark_batch_processing()`: Benchmark batch processing

### BenchmarkConfig

```python
@dataclass
class BenchmarkConfig:
    iterations: int = 10
    warmup_iterations: int = 2
    measure_memory: bool = False
    measure_cpu: bool = False
    output_format: str = "summary"
```

Configuration for benchmark operations.

### PerformanceMetrics

```python
@dataclass
class PerformanceMetrics:
    throughput_samples_per_second: float
    avg_latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    total_time_seconds: float
```

Performance metrics returned by benchmark operations.

### Measurement Utilities

#### measure_time

```python
@contextmanager
def measure_time() -> Generator[Dict[str, float], None, None]
```

Context manager for measuring execution time.

**Returns:** Dictionary with timing information:
- `elapsed_seconds`: Wall clock time
- `cpu_seconds`: CPU time
- `start_time`: Start timestamp
- `end_time`: End timestamp

#### measure_memory

```python
@contextmanager
def measure_memory() -> Generator[Dict[str, int], None, None]
```

Context manager for measuring memory usage.

**Returns:** Dictionary with memory information:
- `delta_mb`: Memory delta in MB
- `delta_rss_mb`: RSS memory delta in MB
- `initial_rss_mb`: Initial RSS memory
- `final_rss_mb`: Final RSS memory

#### benchmark_function

```python
def benchmark_function(func: Callable, iterations: int = 100) -> BenchmarkResult
```

Benchmark a function with statistical analysis.

**Parameters:**
- `func`: Function to benchmark (should take no arguments)
- `iterations`: Number of iterations to run

**Returns:** BenchmarkResult with statistical analysis