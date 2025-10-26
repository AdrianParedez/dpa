#!/usr/bin/env python3
"""
Batch Processing Example: Performance Optimization

This example demonstrates how to optimize batch processing performance
through different strategies, adaptive sizing, and performance monitoring.

Requirements addressed: 2.1, 2.4, 5.1 (performance optimization)
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.batch import BatchConfig, BatchProcessor, BatchStrategy
from src.benchmark import PerformanceProfiler, measure_time
from src.dpa import get_preset, stream_augmentation_chain


def demo_batch_size_optimization():
    """Demonstrate finding optimal batch size for performance."""
    print("=== Batch Size Optimization Demo ===\n")

    num_samples = 100
    config = get_preset("moderate")

    # Test different batch sizes
    batch_sizes = [1, 5, 10, 20, 50]
    results = {}

    print(f"Testing different batch sizes with {num_samples} samples:")
    print("Measuring throughput (samples/second) for each batch size\n")

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")

        batch_config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=batch_size)
        processor = BatchProcessor(BatchStrategy.SEQUENTIAL, batch_config)

        # Measure processing time
        start_time = time.time()

        param_stream = stream_augmentation_chain(num_samples, config)
        processed_samples = 0

        for batch in processor.process_stream(param_stream):
            # Simulate processing work
            time.sleep(0.001 * len(batch))  # 1ms per sample
            processed_samples += len(batch)

        total_time = time.time() - start_time
        throughput = processed_samples / total_time

        results[batch_size] = {
            "throughput": throughput,
            "total_time": total_time,
            "processed": processed_samples,
        }

        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  Total time: {total_time:.3f}s")
        print()

    # Find optimal batch size
    best_batch_size = max(results.keys(), key=lambda x: results[x]["throughput"])
    best_throughput = results[best_batch_size]["throughput"]

    print("Optimization Results:")
    print(f"  Best batch size: {best_batch_size}")
    print(f"  Best throughput: {best_throughput:.1f} samples/sec")
    print(f"  Performance gain: {best_throughput / results[1]['throughput']:.1f}x vs batch_size=1")


def demo_strategy_performance_comparison():
    """Compare performance of different batching strategies."""
    print("\n=== Strategy Performance Comparison ===\n")

    num_samples = 80
    batch_size = 8
    config = get_preset("mild")

    strategies = [
        BatchStrategy.SEQUENTIAL,
        BatchStrategy.ROUND_ROBIN,
        BatchStrategy.MEMORY_OPTIMIZED,
        BatchStrategy.ADAPTIVE,
    ]

    results = {}

    print(f"Comparing strategies with {num_samples} samples, batch size {batch_size}:")
    print("Measuring processing time and throughput\n")

    for strategy in strategies:
        print(f"Testing {strategy.value} strategy...")

        batch_config = BatchConfig(strategy=strategy, batch_size=batch_size, max_memory_mb=200)
        processor = BatchProcessor(strategy, batch_config)

        # Measure performance
        with measure_time() as timer:
            param_stream = stream_augmentation_chain(num_samples, config)

            batch_count = 0
            processed_samples = 0

            for batch in processor.process_stream(param_stream):
                batch_count += 1
                processed_samples += len(batch)

                # Simulate variable processing time based on strategy
                if strategy == BatchStrategy.MEMORY_OPTIMIZED:
                    time.sleep(0.002 * len(batch))  # Slightly slower due to memory checks
                elif strategy == BatchStrategy.ADAPTIVE:
                    time.sleep(0.0015 * len(batch))  # Adaptive overhead
                else:
                    time.sleep(0.001 * len(batch))  # Base processing time

        total_time = timer["elapsed_seconds"]
        throughput = processed_samples / total_time

        results[strategy.value] = {
            "throughput": throughput,
            "total_time": total_time,
            "batch_count": batch_count,
            "avg_batch_size": processed_samples / batch_count,
        }

        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Batches: {batch_count}")
        print(f"  Avg batch size: {processed_samples / batch_count:.1f}")
        print()

    # Find best performing strategy
    best_strategy = max(results.keys(), key=lambda x: results[x]["throughput"])
    print(f"Best performing strategy: {best_strategy}")
    print(f"Best throughput: {results[best_strategy]['throughput']:.1f} samples/sec")


def demo_adaptive_performance_tuning():
    """Demonstrate adaptive performance tuning during processing."""
    print("\n=== Adaptive Performance Tuning Demo ===\n")

    num_samples = 60
    config = get_preset("aggressive")

    # Create adaptive processor
    batch_config = BatchConfig(
        strategy=BatchStrategy.ADAPTIVE,
        batch_size=5,  # Starting size
        adaptive_sizing=True,
    )
    processor = BatchProcessor(BatchStrategy.ADAPTIVE, batch_config)

    print(f"Processing {num_samples} samples with adaptive tuning")
    print("Monitoring performance and batch size adjustments:\n")

    param_stream = stream_augmentation_chain(num_samples, config)

    batch_times = []
    batch_sizes = []
    throughputs = []

    batch_num = 0
    for batch in processor.process_stream(param_stream):
        batch_num += 1
        batch_size = len(batch)
        batch_sizes.append(batch_size)

        # Measure batch processing time
        start_time = time.time()

        # Simulate processing with variable complexity
        complexity_factor = 1.0 + (batch_size - 5) * 0.1  # Complexity increases with size
        time.sleep(0.002 * batch_size * complexity_factor)

        batch_time = time.time() - start_time
        batch_times.append(batch_time)

        throughput = batch_size / batch_time
        throughputs.append(throughput)

        print(f"Batch {batch_num}: {batch_size} samples in {batch_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} samples/sec")

        # Show adaptation logic
        if batch_num > 1:
            prev_throughput = throughputs[-2]
            if throughput > prev_throughput * 1.1:
                print("  📈 Performance improved - may increase batch size")
            elif throughput < prev_throughput * 0.9:
                print("  📉 Performance degraded - may decrease batch size")
            else:
                print("  ➡️  Performance stable")
        print()

    # Performance summary
    if throughputs:
        avg_throughput = statistics.mean(throughputs)
        max_throughput = max(throughputs)
        final_batch_size = batch_sizes[-1]
        initial_batch_size = batch_sizes[0]

        print("Adaptive Tuning Results:")
        print(f"  Initial batch size: {initial_batch_size}")
        print(f"  Final batch size: {final_batch_size}")
        print(f"  Average throughput: {avg_throughput:.1f} samples/sec")
        print(f"  Peak throughput: {max_throughput:.1f} samples/sec")
        print(f"  Adaptation factor: {final_batch_size / initial_batch_size:.1f}x")


def demo_memory_performance_tradeoff():
    """Demonstrate the tradeoff between memory usage and performance."""
    print("\n=== Memory vs Performance Tradeoff Demo ===\n")

    num_samples = 50
    config = get_preset("moderate")

    # Test different memory limits
    memory_limits = [50, 100, 200, 500]  # MB

    print(f"Testing memory vs performance tradeoff with {num_samples} samples:")
    print("Lower memory limits may reduce batch sizes and performance\n")

    for memory_limit in memory_limits:
        print(f"Memory limit: {memory_limit}MB")

        batch_config = BatchConfig(
            strategy=BatchStrategy.MEMORY_OPTIMIZED, max_memory_mb=memory_limit, min_batch_size=1
        )
        processor = BatchProcessor(BatchStrategy.MEMORY_OPTIMIZED, batch_config)

        # Measure performance
        start_time = time.time()
        param_stream = stream_augmentation_chain(num_samples, config)

        batch_sizes = []
        for batch in processor.process_stream(param_stream):
            batch_sizes.append(len(batch))
            time.sleep(0.001 * len(batch))  # Simulate processing

        total_time = time.time() - start_time
        throughput = num_samples / total_time
        avg_batch_size = statistics.mean(batch_sizes) if batch_sizes else 0

        print(f"  Average batch size: {avg_batch_size:.1f}")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  Total batches: {len(batch_sizes)}")
        print()


def demo_profiling_batch_operations():
    """Demonstrate profiling of batch operations for optimization."""
    print("=== Profiling Batch Operations Demo ===\n")

    num_samples = 40
    batch_size = 8
    config = get_preset("moderate")

    # Create profiler
    profiler = PerformanceProfiler(enable_memory_tracking=True)

    print(f"Profiling batch processing of {num_samples} samples:")
    print("Measuring detailed performance metrics\n")

    # Profile batch processing
    profiler.start_profiling("batch_processing")

    batch_config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=batch_size)
    processor = BatchProcessor(BatchStrategy.SEQUENTIAL, batch_config)

    param_stream = stream_augmentation_chain(num_samples, config)

    batch_num = 0
    for batch in processor.process_stream(param_stream):
        batch_num += 1

        # Profile individual batch processing
        profiler.start_profiling(f"batch_{batch_num}")

        # Simulate batch processing work
        time.sleep(0.005 * len(batch))

        profiler.end_profiling(f"batch_{batch_num}")

    profiler.end_profiling("batch_processing")

    # Get profiling results
    profile_summary = profiler.get_profile_summary()

    print("Profiling Results:")
    print("-" * 50)

    for operation, result in profile_summary.items():
        print(f"{operation}:")
        print(f"  Total time: {result.total_time_seconds:.3f}s")
        print(f"  Call count: {result.call_count}")
        print(f"  Avg time per call: {result.avg_time_per_call_ms:.2f}ms")
        if result.memory_delta_mb != 0:
            print(f"  Memory delta: {result.memory_delta_mb:.1f}MB")
        print()


def demo_performance_optimization_workflow():
    """Demonstrate a complete performance optimization workflow."""
    print("=== Performance Optimization Workflow ===\n")

    num_samples = 100
    config = get_preset("moderate")

    print("Step 1: Baseline measurement")
    print("-" * 30)

    # Baseline with default settings
    baseline_config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=10)
    baseline_processor = BatchProcessor(BatchStrategy.SEQUENTIAL, baseline_config)

    start_time = time.time()
    param_stream = stream_augmentation_chain(num_samples, config)

    baseline_batches = 0
    for batch in baseline_processor.process_stream(param_stream):
        time.sleep(0.001 * len(batch))
        baseline_batches += 1

    baseline_time = time.time() - start_time
    baseline_throughput = num_samples / baseline_time

    print(f"Baseline throughput: {baseline_throughput:.1f} samples/sec")
    print(f"Baseline time: {baseline_time:.3f}s")
    print()

    print("Step 2: Strategy optimization")
    print("-" * 30)

    # Test memory-optimized strategy
    optimized_config = BatchConfig(
        strategy=BatchStrategy.MEMORY_OPTIMIZED, max_memory_mb=300, min_batch_size=5
    )
    optimized_processor = BatchProcessor(BatchStrategy.MEMORY_OPTIMIZED, optimized_config)

    start_time = time.time()
    param_stream = stream_augmentation_chain(num_samples, config)

    optimized_batches = 0
    for batch in optimized_processor.process_stream(param_stream):
        time.sleep(0.001 * len(batch))
        optimized_batches += 1

    optimized_time = time.time() - start_time
    optimized_throughput = num_samples / optimized_time

    print(f"Optimized throughput: {optimized_throughput:.1f} samples/sec")
    print(f"Optimized time: {optimized_time:.3f}s")
    print(f"Performance improvement: {optimized_throughput / baseline_throughput:.1f}x")
    print()

    print("Step 3: Adaptive tuning")
    print("-" * 30)

    # Test adaptive strategy
    adaptive_config = BatchConfig(
        strategy=BatchStrategy.ADAPTIVE, batch_size=8, adaptive_sizing=True
    )
    adaptive_processor = BatchProcessor(BatchStrategy.ADAPTIVE, adaptive_config)

    start_time = time.time()
    param_stream = stream_augmentation_chain(num_samples, config)

    adaptive_batches = 0
    for batch in adaptive_processor.process_stream(param_stream):
        time.sleep(0.001 * len(batch))
        adaptive_batches += 1

    adaptive_time = time.time() - start_time
    adaptive_throughput = num_samples / adaptive_time

    print(f"Adaptive throughput: {adaptive_throughput:.1f} samples/sec")
    print(f"Adaptive time: {adaptive_time:.3f}s")
    print(f"Performance improvement: {adaptive_throughput / baseline_throughput:.1f}x")
    print()

    print("Optimization Summary:")
    print("-" * 30)
    best_throughput = max(baseline_throughput, optimized_throughput, adaptive_throughput)

    if best_throughput == adaptive_throughput:
        best_strategy = "Adaptive"
    elif best_throughput == optimized_throughput:
        best_strategy = "Memory-Optimized"
    else:
        best_strategy = "Sequential (Baseline)"

    print(f"Best strategy: {best_strategy}")
    print(f"Best throughput: {best_throughput:.1f} samples/sec")
    print(f"Total improvement: {best_throughput / baseline_throughput:.1f}x")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Batch Processing: Performance Optimization")
    print("=" * 70)

    demo_batch_size_optimization()
    demo_strategy_performance_comparison()
    demo_adaptive_performance_tuning()
    demo_memory_performance_tradeoff()
    demo_profiling_batch_operations()
    demo_performance_optimization_workflow()

    print("\n" + "=" * 70)
    print("Demo completed! Optimize batch processing for your specific use case.")
    print("=" * 70 + "\n")
