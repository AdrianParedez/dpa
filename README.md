# DPA - Deterministic Procedural Augmentation

A Python library for generating deterministic, reproducible data augmentation parameters using Fibonacci-based seeding. Features advanced distributed training support, intelligent batch processing, and comprehensive performance benchmarking.

## Installation

```bash
git clone https://github.com/AdrianParedez/dpa.git
cd dpa
```

**Requirements:** Python 3.12+

**Runtime:** No external dependencies

**Development:** Install dev dependencies (optional)

```bash
pip install -r requirements-dev.txt
```

## Module Structure

```
src/
├── dpa.py          # Core augmentation generation and streaming API
├── distributed.py  # Distributed training support and rank-aware generation
├── batch.py        # Batch processing strategies and memory management
└── benchmark.py    # Performance profiling and benchmarking tools

examples/
├── example_usage.py                          # Basic usage examples
├── distributed_*.py                          # Distributed training examples
├── batch_*.py                                # Batch processing examples
├── benchmark_*.py                            # Benchmarking examples
└── integration_compatibility_demo.py         # Integration examples

tests/
├── test_*.py                                 # Unit tests for each module
└── test_integration_*.py                     # Integration tests
```

## Quick Start

### Basic Usage
```python
from src.dpa import generate_augmentation_chain, get_preset

config = get_preset("moderate")
results = generate_augmentation_chain(100, config, save_path="augmentations.json")
```

### Distributed Training
```python
from src.distributed import gen_distributed_augmentation_params

# Generate unique parameters for each rank
params = gen_distributed_augmentation_params(
    sample_id=42, 
    rank=0, 
    config=config, 
    world_size=4
)
```

### Batch Processing
```python
from src.batch import BatchProcessor, BatchStrategy, BatchConfig

batch_config = BatchConfig(
    strategy=BatchStrategy.MEMORY_OPTIMIZED,
    max_memory_mb=500
)
processor = BatchProcessor(BatchStrategy.MEMORY_OPTIMIZED, batch_config)
```

### Performance Benchmarking
```python
from src.benchmark import BenchmarkRunner, BenchmarkConfig

benchmark_config = BenchmarkConfig(iterations=50, measure_memory=True)
runner = BenchmarkRunner(benchmark_config)
metrics = runner.benchmark_generation(1000, config)
```

## Key Features

### Core Functionality
- **Deterministic Generation**: Reproducible augmentation parameters using Fibonacci-based seeding
- **Streaming API**: Memory-efficient processing of large datasets
- **Preset Configurations**: Three built-in presets (mild, moderate, aggressive)
- **Persistence**: Save and load augmentation chains as JSON with statistics

### Distributed Training Support
- **Rank-Aware Generation**: Unique parameters per rank while maintaining determinism
- **Range Splitting**: Automatic data distribution across multiple processes
- **Multi-Process Coordination**: Seamless integration with distributed training frameworks
- **Reproducibility**: Consistent results across distributed runs

### Intelligent Batch Processing
- **Multiple Strategies**: Sequential, round-robin, memory-optimized, and adaptive batching
- **Memory Management**: Dynamic batch sizing based on available memory
- **Performance Optimization**: Adaptive batch sizes for optimal throughput
- **Resource Monitoring**: Real-time memory and CPU usage tracking

### Performance Benchmarking
- **Comprehensive Profiling**: Detailed performance analysis with timing and memory tracking
- **Comparative Analysis**: Compare different configurations and strategies
- **Optimization Workflows**: Systematic performance improvement processes
- **Regression Detection**: Identify performance changes over time

### Quality Assurance
- **327 Comprehensive Tests**: Extensive test coverage across all modules
- **Integration Testing**: End-to-end validation of complex workflows
- **Backward Compatibility**: Seamless integration with existing code

## Examples

### Basic Examples
- `examples/example_usage.py` - Core functionality and basic usage patterns

### Distributed Training Examples
- `examples/distributed_rank_aware_generation.py` - Rank-aware parameter generation
- `examples/distributed_range_splitting.py` - Data range splitting across ranks
- `examples/distributed_multiprocess_coordination.py` - Multi-process coordination

### Batch Processing Examples
- `examples/batch_processing_strategies.py` - Different batching strategies
- `examples/batch_memory_aware_processing.py` - Memory-aware batch processing
- `examples/batch_performance_optimization.py` - Performance optimization techniques

### Benchmarking Examples
- `examples/benchmark_performance_profiling.py` - Performance profiling and analysis
- `examples/benchmark_comparison_analysis.py` - Comparative benchmarking
- `examples/benchmark_optimization_workflow.py` - Complete optimization workflow

### Integration Examples
- `examples/integration_compatibility_demo.py` - Backward compatibility demonstration

For detailed documentation, see the [docs/](docs/) directory.

## API Overview

### Core Functions (`src.dpa`)
```python
# Basic generation
generate_augmentation_chain(num_samples, config, save_path=None)
gen_augmentation_params(sample_id, config)

# Streaming API
stream_augmentation_chain(num_samples, config, chunk_size=1000)
stream_augmentation_range(start_id, end_id, config)

# Presets and configuration
get_preset("mild" | "moderate" | "aggressive")
AugmentationConfig(rotation_range, brightness_range, ...)
```

### Distributed Training (`src.distributed`)
```python
# Rank-aware generation
gen_distributed_augmentation_params(sample_id, rank, config, world_size)

# Range splitting
DistributedRangeSplitter(total_samples, world_size)
stream_distributed_augmentation_chain(num_samples, rank, world_size, config)
```

### Batch Processing (`src.batch`)
```python
# Batch processing
BatchProcessor(strategy, config)
BatchConfig(strategy, batch_size, max_memory_mb)
BatchStrategy.SEQUENTIAL | ROUND_ROBIN | MEMORY_OPTIMIZED | ADAPTIVE
```

### Benchmarking (`src.benchmark`)
```python
# Performance analysis
BenchmarkRunner(config)
PerformanceProfiler(enable_memory_tracking=True)
measure_time() | measure_memory()  # Context managers
```

## Testing

Run all tests (327 comprehensive tests):

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
# Core functionality
pytest tests/test_dpa.py -v

# Distributed training
pytest tests/test_distributed.py -v

# Batch processing
pytest tests/test_batch.py -v

# Benchmarking
pytest tests/test_benchmark.py -v

# Integration tests
pytest tests/test_integration_*.py -v
```

Lint with Ruff:

```bash
ruff check .
```

## Documentation

- [API Reference](docs/API.md) - Complete function documentation
- [Usage Guide](docs/GUIDE.md) - Examples and best practices
- [Architecture](docs/ARCHITECTURE.md) - Deep dive into how DPA works
- [Scaling Guide](docs/SCALING.md) - Enterprise-scale deployment and optimization
- [Distributed Training Guide](docs/DISTRIBUTED.md) - Distributed training setup and usage
- [Batch Processing Guide](docs/BATCH.md) - Batch processing strategies and optimization
- [Benchmarking Guide](docs/BENCHMARK.md) - Performance analysis and optimization

## Performance

DPA is designed for high-performance scenarios:

- **Throughput**: 30,000+ samples/second for parameter generation
- **Memory Efficiency**: Streaming API with constant memory usage
- **Scalability**: Linear scaling across distributed training ranks
- **Optimization**: Adaptive batch sizing for optimal resource utilization

## Use Cases

- **Machine Learning**: Deterministic data augmentation for reproducible experiments
- **Distributed Training**: Multi-GPU/multi-node training with unique augmentations per rank
- **Large-Scale Processing**: Memory-efficient processing of massive datasets
- **Performance Analysis**: Benchmarking and optimization of augmentation pipelines
- **Research**: Reproducible experiments with consistent augmentation parameters

## Releases

- **[v0.2.1](docs/releases/RELEASE_NOTES_v0.2.1.md)** - Critical Bug Fixes & Enhanced Validation
- **[v0.2.0](docs/releases/RELEASE_NOTES_v0.2.0.md)** - Distributed Training, Batch Processing & Benchmarking
- **v0.1.0** - Initial Release

For detailed release notes, see [docs/releases/](docs/releases/).

For the complete changelog, see [CHANGELOG.md](CHANGELOG.md).

## License

MIT License - see LICENSE file for details