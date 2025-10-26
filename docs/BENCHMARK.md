# Benchmarking Guide

This guide covers DPA's comprehensive benchmarking and performance analysis capabilities, including profiling, comparative analysis, and optimization workflows.

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Quick Start](#quick-start)
- [Performance Profiling](#performance-profiling)
- [Comparative Analysis](#comparative-analysis)
- [Optimization Workflows](#optimization-workflows)
- [Measurement Utilities](#measurement-utilities)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

DPA's benchmarking system provides comprehensive tools for measuring, analyzing, and optimizing the performance of augmentation operations. It includes detailed profiling, comparative analysis, and systematic optimization workflows.

### Key Benefits

- **Comprehensive Profiling**: Detailed timing and memory analysis
- **Comparative Analysis**: Compare configurations, strategies, and approaches
- **Optimization Workflows**: Systematic performance improvement processes
- **Regression Detection**: Identify performance changes over time
- **Resource Monitoring**: Track CPU, memory, and throughput metrics

## Core Concepts

### Performance Metrics

- **Throughput**: Samples processed per second
- **Latency**: Time per individual operation
- **Memory Usage**: RAM consumption during operations
- **CPU Usage**: Processor utilization
- **Scalability**: Performance across different data sizes

### Profiling Types

1. **Function Profiling**: Measure individual function performance
2. **Operation Profiling**: Track specific operations over time
3. **Comparative Profiling**: Compare different approaches
4. **Regression Profiling**: Detect performance changes

## Quick Start

### Basic Performance Measurement

```python
from src.benchmark import measure_time, measure_memory
from src.dpa import generate_augmentation_chain, get_preset

config = get_preset("moderate")

# Measure execution time
with measure_time() as timer:
    results = generate_augmentation_chain(1000, config)

print(f"Generation took {timer['elapsed_seconds']:.3f} seconds")
print(f"Throughput: {1000 / timer['elapsed_seconds']:.1f} samples/sec")

# Measure memory usage
with measure_memory() as memory:
    results = generate_augmentation_chain(5000, config)

print(f"Memory delta: {memory['delta_mb']:.2f} MB")
```

### Performance Profiling

```python
from src.benchmark import PerformanceProfiler

# Create profiler
profiler = PerformanceProfiler(enable_memory_tracking=True)

# Profile operations
profiler.start_profiling("augmentation_generation")

for i in range(100):
    params = gen_augmentation_params(i, config)

result = profiler.end_profiling("augmentation_generation")

print(f"Average time per call: {result.avg_time_per_call_ms:.3f}ms")
print(f"Total time: {result.total_time_seconds:.3f}s")
print(f"Memory delta: {result.memory_delta_mb:.2f}MB")
```

### Benchmark Runner

```python
from src.benchmark import BenchmarkRunner, BenchmarkConfig

# Create benchmark configuration
benchmark_config = BenchmarkConfig(
    iterations=50,
    warmup_iterations=5,
    measure_memory=True,
    measure_cpu=True
)

# Create benchmark runner
runner = BenchmarkRunner(benchmark_config)

# Benchmark parameter generation
metrics = runner.benchmark_generation(1000, config)

print(f"Throughput: {metrics.throughput_samples_per_second:.1f} samples/sec")
print(f"Average latency: {metrics.avg_latency_ms:.2f}ms")
print(f"Memory usage: {metrics.memory_usage_mb:.1f}MB")
print(f"CPU usage: {metrics.cpu_usage_percent:.1f}%")
```

## Performance Profiling

### Function-Level Profiling

Profile individual functions to identify bottlenecks:

```python
from src.benchmark import PerformanceProfiler
from src.dpa import gen_augmentation_params, stream_augmentation_chain

profiler = PerformanceProfiler(enable_memory_tracking=True)

# Profile different operations
operations = [
    ("single_generation", lambda: gen_augmentation_params(42, config)),
    ("batch_generation", lambda: [gen_augmentation_params(i, config) for i in range(100)]),
    ("streaming_generation", lambda: list(stream_augmentation_chain(100, config))),
]

for operation_name, operation_func in operations:
    profiler.start_profiling(operation_name)
    
    # Run operation multiple times for better measurement
    for _ in range(10):
        result = operation_func()
    
    profile_result = profiler.end_profiling(operation_name)
    
    print(f"{operation_name}:")
    print(f"  Avg time per call: {profile_result.avg_time_per_call_ms:.3f}ms")
    print(f"  Total time: {profile_result.total_time_seconds:.4f}s")
    print(f"  Memory delta: {profile_result.memory_delta_mb:.3f}MB")
    print()
```

### Decorator-Based Profiling

Use decorators for automatic profiling:

```python
profiler = PerformanceProfiler()

@profiler.profile_function
def generate_parameters_batch(num_samples, config):
    """Generate a batch of parameters."""
    return [gen_augmentation_params(i, config) for i in range(num_samples)]

@profiler.profile_function
def generate_parameters_streaming(num_samples, config):
    """Generate parameters using streaming."""
    return list(stream_augmentation_chain(num_samples, config))

# Use profiled functions
batch_result = generate_parameters_batch(100, config)
streaming_result = generate_parameters_streaming(100, config)

# Get profiling summary
summary = profiler.get_profile_summary()
for operation, result in summary.items():
    print(f"{operation}: {result.avg_time_per_call_ms:.3f}ms avg")
```

### Context Manager Profiling

Use context managers for block-level profiling:

```python
profiler = PerformanceProfiler()

# Profile a code block
with profiler.profile_context("complex_operation"):
    # Setup
    config = get_preset("aggressive")
    
    # Generation
    results = []
    for i in range(1000):
        params = gen_augmentation_params(i, config)
        results.append(params)
    
    # Statistics
    from src.dpa import compute_statistics
    stats = compute_statistics(results)

# Get results
result = profiler.get_profile_summary()["complex_operation"]
print(f"Complex operation took {result.total_time_seconds:.3f}s")
```

### Detailed Operation Breakdown

Profile individual components of operations:

```python
def profile_operation_breakdown():
    """Profile the breakdown of augmentation generation."""
    
    profiler = PerformanceProfiler(enable_memory_tracking=True)
    config = get_preset("moderate")
    
    # Profile individual components
    components = [
        ("seed_generation", lambda: gen_augmentation_seed(42, 10)),
        ("parameter_extraction", lambda: gen_augmentation_params(42, config)),
        ("config_validation", lambda: config.rotation_range),
        ("random_operations", lambda: [gen_augmentation_params(i, config) for i in range(10)]),
    ]
    
    for component_name, component_func in components:
        profiler.start_profiling(component_name)
        
        # Run component multiple times
        for _ in range(100):
            result = component_func()
        
        profile_result = profiler.end_profiling(component_name)
        
        print(f"{component_name}:")
        print(f"  Time per call: {profile_result.avg_time_per_call_ms * 1000:.1f}μs")
        print(f"  Total time: {profile_result.total_time_seconds:.4f}s")
    
    # Show relative performance
    summary = profiler.get_profile_summary()
    total_time = sum(r.total_time_seconds for r in summary.values())
    
    print("\nRelative Performance:")
    for operation, result in sorted(summary.items(), 
                                  key=lambda x: x[1].total_time_seconds, 
                                  reverse=True):
        percentage = (result.total_time_seconds / total_time) * 100
        print(f"  {operation}: {percentage:.1f}% of total time")

profile_operation_breakdown()
```

## Comparative Analysis

### Configuration Comparison

Compare different augmentation configurations:

```python
from src.benchmark import BenchmarkRunner, BenchmarkConfig

def compare_configurations():
    """Compare performance of different augmentation configurations."""
    
    # Define configurations to compare
    configs = {
        "mild": get_preset("mild"),
        "moderate": get_preset("moderate"),
        "aggressive": get_preset("aggressive"),
        "custom_light": AugmentationConfig(
            rotation_range=(-15, 15),
            brightness_range=(0.9, 1.1),
            noise_range=(0, 0.05),
            scale_range=(0.95, 1.05),
            contrast_range=(0.95, 1.05),
            augmentation_depth=5
        ),
    }
    
    # Benchmark configuration
    benchmark_config = BenchmarkConfig(
        iterations=30,
        warmup_iterations=5,
        measure_memory=True,
        measure_cpu=True
    )
    
    runner = BenchmarkRunner(benchmark_config)
    results = {}
    
    print("Configuration Performance Comparison:")
    print("-" * 50)
    
    for config_name, config in configs.items():
        metrics = runner.benchmark_generation(1000, config)
        results[config_name] = metrics
        
        print(f"{config_name}:")
        print(f"  Throughput: {metrics.throughput_samples_per_second:.1f} samples/sec")
        print(f"  Latency: {metrics.avg_latency_ms:.3f}ms")
        print(f"  Memory: {metrics.memory_usage_mb:.1f}MB")
        print(f"  CPU: {metrics.cpu_usage_percent:.1f}%")
        print()
    
    # Find best performers
    best_throughput = max(results.items(), key=lambda x: x[1].throughput_samples_per_second)
    best_memory = min(results.items(), key=lambda x: x[1].memory_usage_mb)
    
    print("Best Performers:")
    print(f"  Highest throughput: {best_throughput[0]} ({best_throughput[1].throughput_samples_per_second:.1f} samples/sec)")
    print(f"  Lowest memory: {best_memory[0]} ({best_memory[1].memory_usage_mb:.1f}MB)")

compare_configurations()
```

### Strategy Comparison

Compare different processing strategies:

```python
from src.batch import BatchProcessor, BatchStrategy, BatchConfig

def compare_batch_strategies():
    """Compare performance of different batch processing strategies."""
    
    num_samples = 5000
    config = get_preset("moderate")
    
    strategies = {
        "sequential": BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=50),
        "round_robin": BatchConfig(strategy=BatchStrategy.ROUND_ROBIN, batch_size=10),
        "memory_optimized": BatchConfig(strategy=BatchStrategy.MEMORY_OPTIMIZED, 
                                      max_memory_mb=500, min_batch_size=10),
        "adaptive": BatchConfig(strategy=BatchStrategy.ADAPTIVE, 
                               batch_size=30, adaptive_sizing=True),
    }
    
    benchmark_config = BenchmarkConfig(iterations=20, measure_memory=True)
    runner = BenchmarkRunner(benchmark_config)
    
    results = {}
    
    for strategy_name, batch_config in strategies.items():
        metrics = runner.benchmark_batch_processing(num_samples, batch_config)
        results[strategy_name] = metrics
        
        print(f"{strategy_name}:")
        print(f"  Throughput: {metrics.throughput_samples_per_second:.1f} samples/sec")
        print(f"  Total time: {metrics.total_time_seconds:.3f}s")
        print(f"  Memory: {metrics.memory_usage_mb:.1f}MB")
        print()
    
    # Rank by performance
    ranked = sorted(results.items(), 
                   key=lambda x: x[1].throughput_samples_per_second, 
                   reverse=True)
    
    print("Performance Ranking:")
    for i, (strategy, metrics) in enumerate(ranked, 1):
        print(f"  {i}. {strategy}: {metrics.throughput_samples_per_second:.1f} samples/sec")

compare_batch_strategies()
```

### Scaling Analysis

Analyze how performance scales with data size:

```python
def analyze_scaling_performance():
    """Analyze how performance scales with different data sizes."""
    
    config = get_preset("moderate")
    sample_sizes = [100, 500, 1000, 5000, 10000]
    
    benchmark_config = BenchmarkConfig(
        iterations=20,
        warmup_iterations=3,
        measure_memory=True
    )
    runner = BenchmarkRunner(benchmark_config)
    
    results = {}
    
    print("Scaling Performance Analysis:")
    print("-" * 40)
    
    for size in sample_sizes:
        metrics = runner.benchmark_generation(size, config)
        results[size] = metrics
        
        print(f"{size:5d} samples: {metrics.throughput_samples_per_second:6.1f} samples/sec, "
              f"{metrics.memory_usage_mb:5.1f}MB")
    
    # Analyze scaling characteristics
    print("\nScaling Analysis:")
    
    # Check if throughput is consistent (good scaling)
    throughputs = [metrics.throughput_samples_per_second for metrics in results.values()]
    throughput_variation = (max(throughputs) - min(throughputs)) / min(throughputs) * 100
    
    print(f"  Throughput variation: {throughput_variation:.1f}%")
    
    if throughput_variation < 20:
        print("  ✓ Good scaling - consistent throughput")
    else:
        print("  ⚠ Poor scaling - throughput varies significantly")
    
    # Check memory scaling
    memory_usages = [metrics.memory_usage_mb for metrics in results.values()]
    memory_growth = memory_usages[-1] / memory_usages[0]
    sample_growth = sample_sizes[-1] / sample_sizes[0]
    
    print(f"  Memory growth: {memory_growth:.1f}x for {sample_growth:.1f}x more samples")
    
    if memory_growth < sample_growth * 0.1:  # Less than 10% linear growth
        print("  ✓ Excellent memory efficiency")
    elif memory_growth < sample_growth * 0.5:  # Less than 50% linear growth
        print("  ✓ Good memory efficiency")
    else:
        print("  ⚠ High memory growth")

analyze_scaling_performance()
```

## Optimization Workflows

### Systematic Performance Optimization

Complete workflow for optimizing DPA performance:

```python
class PerformanceOptimizer:
    """Systematic performance optimization workflow."""
    
    def __init__(self):
        self.benchmark_config = BenchmarkConfig(
            iterations=30,
            warmup_iterations=5,
            measure_memory=True,
            measure_cpu=True
        )
        self.runner = BenchmarkRunner(self.benchmark_config)
        self.optimization_history = []
    
    def step1_baseline_measurement(self, num_samples=1000):
        """Establish baseline performance."""
        print("=== Step 1: Baseline Measurement ===")
        
        config = get_preset("moderate")
        baseline_metrics = self.runner.benchmark_generation(num_samples, config)
        
        self.optimization_history.append({
            'step': 'baseline',
            'config': 'moderate',
            'metrics': baseline_metrics
        })
        
        print(f"Baseline throughput: {baseline_metrics.throughput_samples_per_second:.1f} samples/sec")
        print(f"Baseline latency: {baseline_metrics.avg_latency_ms:.2f}ms")
        print(f"Baseline memory: {baseline_metrics.memory_usage_mb:.1f}MB")
        
        return baseline_metrics
    
    def step2_configuration_optimization(self, num_samples=1000):
        """Optimize augmentation configuration."""
        print("\n=== Step 2: Configuration Optimization ===")
        
        configs = {
            "lightweight": AugmentationConfig(
                rotation_range=(-30, 30),
                brightness_range=(0.8, 1.2),
                noise_range=(0, 0.1),
                scale_range=(0.9, 1.1),
                contrast_range=(0.9, 1.1),
                augmentation_depth=5
            ),
            "balanced": AugmentationConfig(
                rotation_range=(-45, 45),
                brightness_range=(0.7, 1.3),
                noise_range=(0, 0.15),
                scale_range=(0.8, 1.2),
                contrast_range=(0.8, 1.2),
                augmentation_depth=8
            ),
        }
        
        best_config = None
        best_metrics = None
        
        for config_name, config in configs.items():
            metrics = self.runner.benchmark_generation(num_samples, config)
            
            self.optimization_history.append({
                'step': 'config_optimization',
                'config': config_name,
                'metrics': metrics
            })
            
            print(f"{config_name}: {metrics.throughput_samples_per_second:.1f} samples/sec")
            
            if best_metrics is None or metrics.throughput_samples_per_second > best_metrics.throughput_samples_per_second:
                best_config = config
                best_metrics = metrics
        
        print(f"Best configuration: {best_metrics.throughput_samples_per_second:.1f} samples/sec")
        return best_config, best_metrics
    
    def step3_batch_optimization(self, num_samples=2000, optimal_config=None):
        """Optimize batch processing."""
        print("\n=== Step 3: Batch Processing Optimization ===")
        
        if optimal_config is None:
            optimal_config = get_preset("moderate")
        
        batch_configs = {
            "small_batches": BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=10),
            "medium_batches": BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=50),
            "large_batches": BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=200),
            "memory_optimized": BatchConfig(strategy=BatchStrategy.MEMORY_OPTIMIZED, 
                                          max_memory_mb=500, min_batch_size=10),
        }
        
        best_batch_config = None
        best_metrics = None
        
        for batch_name, batch_config in batch_configs.items():
            metrics = self.runner.benchmark_batch_processing(num_samples, batch_config)
            
            self.optimization_history.append({
                'step': 'batch_optimization',
                'config': batch_name,
                'metrics': metrics
            })
            
            print(f"{batch_name}: {metrics.throughput_samples_per_second:.1f} samples/sec")
            
            if best_metrics is None or metrics.throughput_samples_per_second > best_metrics.throughput_samples_per_second:
                best_batch_config = batch_config
                best_metrics = metrics
        
        print(f"Best batch config: {best_metrics.throughput_samples_per_second:.1f} samples/sec")
        return best_batch_config, best_metrics
    
    def step4_validation(self, num_samples=5000, final_config=None):
        """Validate optimized configuration."""
        print("\n=== Step 4: Validation ===")
        
        if final_config is None:
            final_config = get_preset("moderate")
        
        final_metrics = self.runner.benchmark_generation(num_samples, final_config)
        
        self.optimization_history.append({
            'step': 'validation',
            'config': 'optimized_final',
            'metrics': final_metrics
        })
        
        print(f"Final throughput: {final_metrics.throughput_samples_per_second:.1f} samples/sec")
        print(f"Final latency: {final_metrics.avg_latency_ms:.2f}ms")
        print(f"Final memory: {final_metrics.memory_usage_mb:.1f}MB")
        
        return final_metrics
    
    def generate_report(self):
        """Generate optimization report."""
        print("\n=== Optimization Report ===")
        
        if len(self.optimization_history) < 2:
            print("Insufficient data for report")
            return
        
        baseline = self.optimization_history[0]
        final = self.optimization_history[-1]
        
        # Calculate improvements
        throughput_improvement = (
            (final['metrics'].throughput_samples_per_second - 
             baseline['metrics'].throughput_samples_per_second) /
            baseline['metrics'].throughput_samples_per_second
        ) * 100
        
        latency_improvement = (
            (baseline['metrics'].avg_latency_ms - final['metrics'].avg_latency_ms) /
            baseline['metrics'].avg_latency_ms
        ) * 100
        
        print(f"Throughput improvement: {throughput_improvement:+.1f}%")
        print(f"Latency improvement: {latency_improvement:+.1f}%")
        
        # Show optimization steps
        print("\nOptimization Steps:")
        for i, step in enumerate(self.optimization_history, 1):
            print(f"  {i}. {step['config']}: {step['metrics'].throughput_samples_per_second:.1f} samples/sec")
        
        if throughput_improvement > 10:
            print("\n✅ Significant performance improvement achieved!")
        elif throughput_improvement > 0:
            print("\n✅ Performance improvement achieved.")
        else:
            print("\n⚠️ No significant improvement. Consider different approaches.")

# Run optimization workflow
optimizer = PerformanceOptimizer()
baseline = optimizer.step1_baseline_measurement()
best_config, config_metrics = optimizer.step2_configuration_optimization()
best_batch, batch_metrics = optimizer.step3_batch_optimization(optimal_config=best_config)
final_metrics = optimizer.step4_validation(final_config=best_config)
optimizer.generate_report()
```

### Performance Regression Detection

Detect performance regressions over time:

```python
def detect_performance_regression():
    """Detect performance regressions by comparing against baseline."""
    
    config = get_preset("moderate")
    num_samples = 1000
    
    # Establish baseline (simulate historical performance)
    print("Establishing baseline performance...")
    baseline_times = []
    
    for run in range(5):
        with measure_time() as timer:
            results = generate_augmentation_chain(num_samples, config)
        baseline_times.append(timer['elapsed_seconds'])
    
    baseline_avg = sum(baseline_times) / len(baseline_times)
    baseline_throughput = num_samples / baseline_avg
    
    print(f"Baseline throughput: {baseline_throughput:.1f} samples/sec")
    
    # Test current performance (simulate with slight regression)
    print("\nTesting current performance...")
    current_times = []
    
    for run in range(5):
        with measure_time() as timer:
            results = generate_augmentation_chain(num_samples, config)
            # Simulate slight performance regression
            time.sleep(0.001)  # 1ms additional delay
        current_times.append(timer['elapsed_seconds'])
    
    current_avg = sum(current_times) / len(current_times)
    current_throughput = num_samples / current_avg
    
    print(f"Current throughput: {current_throughput:.1f} samples/sec")
    
    # Analyze regression
    performance_change = (current_throughput - baseline_throughput) / baseline_throughput * 100
    
    print(f"\nPerformance change: {performance_change:+.1f}%")
    
    # Determine if regression is significant
    regression_threshold = 5.0  # 5% threshold
    
    if abs(performance_change) > regression_threshold:
        if performance_change < 0:
            print(f"🚨 Performance regression detected! ({abs(performance_change):.1f}% slower)")
        else:
            print(f"📈 Performance improvement detected! ({performance_change:.1f}% faster)")
    else:
        print(f"✅ Performance within acceptable range (±{regression_threshold}%)")

detect_performance_regression()
```

## Measurement Utilities

### Context Managers

Use context managers for easy measurement:

```python
from src.benchmark import measure_time, measure_memory

# Time measurement
with measure_time() as timer:
    # Your code here
    results = generate_augmentation_chain(1000, config)

print(f"Elapsed time: {timer['elapsed_seconds']:.3f}s")
print(f"CPU time: {timer['cpu_seconds']:.3f}s")

# Memory measurement
with measure_memory() as memory:
    # Your code here
    large_results = generate_augmentation_chain(10000, config)

print(f"Memory delta: {memory['delta_mb']:.2f}MB")
print(f"RSS delta: {memory['delta_rss_mb']:.2f}MB")

# Combined measurement
with measure_time() as timer, measure_memory() as memory:
    # Your code here
    results = generate_augmentation_chain(5000, config)

print(f"Time: {timer['elapsed_seconds']:.3f}s, Memory: {memory['delta_mb']:.2f}MB")
```

### Function Benchmarking

Benchmark functions with statistical analysis:

```python
from src.benchmark import benchmark_function

def benchmark_generation_function():
    """Benchmark parameter generation function."""
    config = get_preset("moderate")
    
    # Define function to benchmark
    def generation_task():
        return generate_augmentation_chain(100, config)
    
    # Benchmark with statistical analysis
    result = benchmark_function(generation_task, iterations=50)
    
    print(f"Function: {result.function_name}")
    print(f"Iterations: {result.iterations}")
    print(f"Mean time: {result.mean_time_seconds:.4f}s")
    print(f"Std deviation: {result.std_time_seconds:.4f}s")
    print(f"Min time: {result.min_time_seconds:.4f}s")
    print(f"Max time: {result.max_time_seconds:.4f}s")
    print(f"Median time: {result.median_time_seconds:.4f}s")
    
    # Calculate confidence interval
    confidence_95 = 1.96 * result.std_time_seconds / (result.iterations ** 0.5)
    print(f"95% confidence interval: ±{confidence_95:.4f}s")

benchmark_generation_function()
```

### Custom Performance Metrics

Create custom performance metrics:

```python
class CustomPerformanceMetrics:
    """Custom performance metrics for specific use cases."""
    
    def __init__(self):
        self.metrics = {}
    
    def measure_throughput_vs_quality(self, sample_sizes, configs):
        """Measure throughput vs augmentation quality tradeoff."""
        
        results = {}
        
        for config_name, config in configs.items():
            config_results = {}
            
            for size in sample_sizes:
                with measure_time() as timer:
                    params = generate_augmentation_chain(size, config)
                
                throughput = size / timer['elapsed_seconds']
                
                # Calculate "quality" metric (augmentation diversity)
                quality_score = self._calculate_quality_score(params, config)
                
                config_results[size] = {
                    'throughput': throughput,
                    'quality': quality_score,
                    'efficiency': throughput * quality_score  # Combined metric
                }
            
            results[config_name] = config_results
        
        return results
    
    def _calculate_quality_score(self, params, config):
        """Calculate augmentation quality score."""
        if not params:
            return 0.0
        
        # Simple quality metric based on parameter diversity
        rotations = [p['rotation'] for p in params]
        brightness = [p['brightness'] for p in params]
        
        rotation_diversity = max(rotations) - min(rotations)
        brightness_diversity = max(brightness) - min(brightness)
        
        # Normalize by config ranges
        rotation_range = config.rotation_range[1] - config.rotation_range[0]
        brightness_range = config.brightness_range[1] - config.brightness_range[0]
        
        quality = (rotation_diversity / rotation_range + 
                  brightness_diversity / brightness_range) / 2
        
        return min(quality, 1.0)  # Cap at 1.0
    
    def measure_memory_efficiency(self, sample_sizes):
        """Measure memory efficiency across different data sizes."""
        
        config = get_preset("moderate")
        results = {}
        
        for size in sample_sizes:
            with measure_memory() as memory:
                params = generate_augmentation_chain(size, config)
            
            memory_per_sample = memory['delta_mb'] / size if size > 0 else 0
            
            results[size] = {
                'total_memory_mb': memory['delta_mb'],
                'memory_per_sample_kb': memory_per_sample * 1024,
                'samples': size
            }
        
        return results

# Use custom metrics
metrics = CustomPerformanceMetrics()

# Measure throughput vs quality
configs = {
    "mild": get_preset("mild"),
    "aggressive": get_preset("aggressive")
}
quality_results = metrics.measure_throughput_vs_quality([100, 500, 1000], configs)

for config_name, config_results in quality_results.items():
    print(f"{config_name} configuration:")
    for size, result in config_results.items():
        print(f"  {size} samples: {result['throughput']:.1f} samples/sec, "
              f"quality: {result['quality']:.3f}, "
              f"efficiency: {result['efficiency']:.1f}")
    print()

# Measure memory efficiency
memory_results = metrics.measure_memory_efficiency([100, 500, 1000, 5000])

print("Memory Efficiency:")
for size, result in memory_results.items():
    print(f"  {size} samples: {result['memory_per_sample_kb']:.2f} KB/sample")
```

## Best Practices

### 1. Proper Benchmarking Setup

```python
# ✓ Good: Proper warmup and multiple iterations
benchmark_config = BenchmarkConfig(
    iterations=50,          # Sufficient iterations for statistical significance
    warmup_iterations=5,    # Warmup to stabilize performance
    measure_memory=True,    # Include memory measurements
    measure_cpu=True        # Include CPU measurements
)

# ✗ Bad: Insufficient measurements
benchmark_config = BenchmarkConfig(
    iterations=1,           # Too few iterations
    warmup_iterations=0,    # No warmup
    measure_memory=False    # Missing important metrics
)
```

### 2. Statistical Significance

```python
def ensure_statistical_significance():
    """Ensure benchmark results are statistically significant."""
    
    config = get_preset("moderate")
    measurements = []
    
    # Collect multiple measurements
    for _ in range(30):  # At least 30 samples for normal distribution
        with measure_time() as timer:
            results = generate_augmentation_chain(1000, config)
        measurements.append(timer['elapsed_seconds'])
    
    # Calculate statistics
    mean_time = sum(measurements) / len(measurements)
    variance = sum((x - mean_time) ** 2 for x in measurements) / (len(measurements) - 1)
    std_dev = variance ** 0.5
    
    # Calculate confidence interval (95%)
    confidence_interval = 1.96 * std_dev / (len(measurements) ** 0.5)
    
    print(f"Mean time: {mean_time:.4f}s ± {confidence_interval:.4f}s (95% CI)")
    print(f"Coefficient of variation: {(std_dev / mean_time) * 100:.1f}%")
    
    # Check if results are stable (CV < 10%)
    if (std_dev / mean_time) * 100 < 10:
        print("✅ Results are statistically stable")
    else:
        print("⚠️ Results show high variation - consider more iterations")

ensure_statistical_significance()
```

### 3. Environment Control

```python
import gc
import psutil

def controlled_benchmark_environment():
    """Set up controlled environment for benchmarking."""
    
    # Force garbage collection
    gc.collect()
    
    # Check system load
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    print(f"System state before benchmark:")
    print(f"  CPU usage: {cpu_percent:.1f}%")
    print(f"  Memory usage: {memory_percent:.1f}%")
    
    # Warn if system is under load
    if cpu_percent > 50:
        print("⚠️ High CPU usage detected - results may be unreliable")
    
    if memory_percent > 80:
        print("⚠️ High memory usage detected - results may be unreliable")
    
    # Set process priority (if possible)
    try:
        import os
        if hasattr(os, 'nice'):
            os.nice(-5)  # Higher priority on Unix systems
            print("✅ Process priority increased")
    except:
        pass
    
    return cpu_percent < 50 and memory_percent < 80

# Use before important benchmarks
if controlled_benchmark_environment():
    # Run benchmarks
    pass
else:
    print("System not suitable for benchmarking")
```

### 4. Result Validation

```python
def validate_benchmark_results(results):
    """Validate benchmark results for sanity."""
    
    # Check for reasonable values
    if results.throughput_samples_per_second <= 0:
        raise ValueError("Invalid throughput: must be positive")
    
    if results.avg_latency_ms <= 0:
        raise ValueError("Invalid latency: must be positive")
    
    # Check for unrealistic values
    if results.throughput_samples_per_second > 1000000:  # 1M samples/sec
        print("⚠️ Unusually high throughput - verify results")
    
    if results.avg_latency_ms > 1000:  # 1 second per sample
        print("⚠️ Unusually high latency - verify results")
    
    # Check consistency
    expected_latency = 1000 / results.throughput_samples_per_second  # ms
    actual_latency = results.avg_latency_ms
    
    if abs(expected_latency - actual_latency) / expected_latency > 0.1:  # 10% tolerance
        print("⚠️ Inconsistent throughput/latency relationship")
    
    print("✅ Benchmark results validated")

# Use after benchmarking
benchmark_config = BenchmarkConfig(iterations=30, measure_memory=True)
runner = BenchmarkRunner(benchmark_config)
results = runner.benchmark_generation(1000, get_preset("moderate"))
validate_benchmark_results(results)
```

## Troubleshooting

### Common Issues

#### 1. Inconsistent Results

```python
# Problem: Results vary significantly between runs
# Cause: System load, insufficient warmup, too few iterations

# Solution: Proper benchmarking setup
benchmark_config = BenchmarkConfig(
    iterations=50,          # More iterations
    warmup_iterations=10,   # More warmup
    measure_memory=True
)

# Also check system state
def check_system_stability():
    cpu_usage = []
    for _ in range(10):
        cpu_usage.append(psutil.cpu_percent(interval=0.1))
    
    cpu_variation = (max(cpu_usage) - min(cpu_usage)) / min(cpu_usage) * 100
    
    if cpu_variation > 20:
        print("⚠️ System CPU usage is unstable")
        return False
    
    return True

if check_system_stability():
    # Run benchmarks
    pass
```

#### 2. Memory Measurement Issues

```python
# Problem: Memory measurements show negative values
# Cause: Garbage collection, memory fragmentation

# Solution: Force GC and use multiple measurements
def reliable_memory_measurement():
    import gc
    
    # Force garbage collection before measurement
    gc.collect()
    
    measurements = []
    
    for _ in range(5):  # Multiple measurements
        gc.collect()  # GC before each measurement
        
        with measure_memory() as memory:
            results = generate_augmentation_chain(1000, get_preset("moderate"))
        
        measurements.append(memory['delta_mb'])
    
    # Use median to avoid outliers
    measurements.sort()
    median_memory = measurements[len(measurements) // 2]
    
    print(f"Reliable memory measurement: {median_memory:.2f}MB")
    return median_memory

reliable_memory_measurement()
```

#### 3. Performance Profiling Overhead

```python
# Problem: Profiling adds significant overhead
# Cause: Too frequent measurements, memory tracking enabled

# Solution: Optimize profiling configuration
profiler = PerformanceProfiler(
    enable_memory_tracking=False  # Disable if not needed
)

# Profile larger operations to reduce overhead
@profiler.profile_function
def batch_operation():
    """Profile batch operations instead of individual calls."""
    config = get_preset("moderate")
    return [gen_augmentation_params(i, config) for i in range(100)]

# Instead of profiling each individual call
# for i in range(100):
#     with profiler.profile_context(f"call_{i}"):  # High overhead
#         gen_augmentation_params(i, config)
```

### Performance Debugging

```python
def debug_performance_issues():
    """Debug common performance issues."""
    
    config = get_preset("moderate")
    
    # Test 1: Check if problem is in generation or processing
    print("Testing parameter generation performance...")
    
    with measure_time() as timer:
        params = [gen_augmentation_params(i, config) for i in range(1000)]
    
    generation_time = timer['elapsed_seconds']
    generation_throughput = 1000 / generation_time
    
    print(f"Generation: {generation_throughput:.1f} samples/sec")
    
    # Test 2: Check memory usage pattern
    print("\nTesting memory usage pattern...")
    
    memory_measurements = []
    
    for batch_size in [100, 500, 1000, 2000]:
        with measure_memory() as memory:
            params = generate_augmentation_chain(batch_size, config)
        
        memory_per_sample = memory['delta_mb'] / batch_size
        memory_measurements.append((batch_size, memory_per_sample))
        
        print(f"  {batch_size} samples: {memory_per_sample * 1024:.2f} KB/sample")
    
    # Check if memory usage is linear
    memory_ratios = []
    for i in range(1, len(memory_measurements)):
        prev_size, prev_memory = memory_measurements[i-1]
        curr_size, curr_memory = memory_measurements[i]
        
        size_ratio = curr_size / prev_size
        memory_ratio = curr_memory / prev_memory
        
        memory_ratios.append(memory_ratio / size_ratio)
    
    avg_memory_ratio = sum(memory_ratios) / len(memory_ratios)
    
    if 0.9 <= avg_memory_ratio <= 1.1:
        print("  ✅ Linear memory scaling")
    else:
        print(f"  ⚠️ Non-linear memory scaling (ratio: {avg_memory_ratio:.2f})")
    
    # Test 3: Check for bottlenecks in different operations
    print("\nTesting operation bottlenecks...")
    
    operations = [
        ("seed_generation", lambda: [gen_augmentation_seed(i, 10) for i in range(100)]),
        ("config_creation", lambda: [get_preset("moderate") for _ in range(100)]),
        ("param_generation", lambda: [gen_augmentation_params(i, config) for i in range(100)]),
    ]
    
    for op_name, op_func in operations:
        with measure_time() as timer:
            result = op_func()
        
        op_throughput = 100 / timer['elapsed_seconds']
        print(f"  {op_name}: {op_throughput:.1f} ops/sec")

debug_performance_issues()
```

This comprehensive guide covers all aspects of DPA's benchmarking and performance analysis capabilities. For more examples, see the `examples/benchmark_*.py` files in the repository.