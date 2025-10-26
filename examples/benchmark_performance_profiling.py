#!/usr/bin/env python3
"""
Benchmarking Example: Performance Profiling

This example demonstrates how to use the performance profiling capabilities
to measure and analyze the performance of augmentation operations.

Requirements addressed: 3.1 (performance profiling)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.batch import BatchConfig, BatchProcessor, BatchStrategy
from src.benchmark import PerformanceProfiler, measure_memory, measure_time
from src.distributed import gen_distributed_augmentation_params
from src.dpa import gen_augmentation_params, get_preset, stream_augmentation_chain


def demo_basic_profiling():
    """Demonstrate basic performance profiling capabilities."""
    print("=== Basic Performance Profiling Demo ===\n")

    # Create profiler
    profiler = PerformanceProfiler(enable_memory_tracking=True)

    config = get_preset("moderate")
    num_samples = 100

    print(f"Profiling generation of {num_samples} augmentation parameters")
    print("Measuring timing and memory usage:\n")

    # Profile parameter generation
    profiler.start_profiling("parameter_generation")

    for sample_id in range(num_samples):
        params = gen_augmentation_params(sample_id, config)

        # Simulate some processing
        time.sleep(0.0001)  # 0.1ms per sample

    result = profiler.end_profiling("parameter_generation")

    print("Profiling Results:")
    print(f"  Operation: {result.operation_name}")
    print(f"  Total time: {result.total_time_seconds:.3f}s")
    print(f"  Call count: {result.call_count}")
    print(f"  Average time per call: {result.avg_time_per_call_ms:.2f}ms")
    print(f"  Memory delta: {result.memory_delta_mb:.2f}MB")
    print()

    # Calculate throughput
    throughput = num_samples / result.total_time_seconds
    print(f"Throughput: {throughput:.1f} samples/second")


def demo_function_profiling_decorator():
    """Demonstrate profiling using decorators."""
    print("\n=== Function Profiling Decorator Demo ===\n")

    profiler = PerformanceProfiler()

    # Create profiled versions of functions
    @profiler.profile_function
    def generate_single_param(sample_id, config):
        """Generate a single augmentation parameter set."""
        return gen_augmentation_params(sample_id, config)

    @profiler.profile_function
    def generate_distributed_param(sample_id, rank, config):
        """Generate a distributed augmentation parameter set."""
        return gen_distributed_augmentation_params(sample_id, rank, config)

    config = get_preset("mild")

    print("Profiling functions using decorators:")
    print("Comparing regular vs distributed parameter generation\n")

    # Test regular generation
    print("Regular parameter generation:")
    for i in range(10):
        params = generate_single_param(i, config)

    # Test distributed generation
    print("Distributed parameter generation:")
    for i in range(10):
        params = generate_distributed_param(i, rank=0, config=config)

    # Get profiling summary
    profile_summary = profiler.get_profile_summary()

    print("\nProfiling Summary:")
    print("-" * 50)

    for operation, result in profile_summary.items():
        print(f"{operation}:")
        print(f"  Total time: {result.total_time_seconds:.4f}s")
        print(f"  Call count: {result.call_count}")
        print(f"  Avg time per call: {result.avg_time_per_call_ms:.3f}ms")
        print()


def demo_streaming_profiling():
    """Demonstrate profiling of streaming operations."""
    print("=== Streaming Operations Profiling Demo ===\n")

    profiler = PerformanceProfiler(enable_memory_tracking=True)
    config = get_preset("aggressive")
    num_samples = 50

    print(f"Profiling streaming generation of {num_samples} samples")
    print("Measuring streaming performance vs batch generation:\n")

    # Profile streaming generation
    profiler.start_profiling("streaming_generation")

    param_stream = stream_augmentation_chain(num_samples, config)
    streaming_params = list(param_stream)

    streaming_result = profiler.end_profiling("streaming_generation")

    # Profile batch generation for comparison
    profiler.start_profiling("batch_generation")

    batch_params = []
    for sample_id in range(num_samples):
        params = gen_augmentation_params(sample_id, config)
        batch_params.append(params)

    batch_result = profiler.end_profiling("batch_generation")

    print("Performance Comparison:")
    print("-" * 40)
    print("Streaming approach:")
    print(f"  Time: {streaming_result.total_time_seconds:.4f}s")
    print(f"  Memory delta: {streaming_result.memory_delta_mb:.2f}MB")
    print(f"  Throughput: {num_samples / streaming_result.total_time_seconds:.1f} samples/sec")
    print()

    print("Batch approach:")
    print(f"  Time: {batch_result.total_time_seconds:.4f}s")
    print(f"  Memory delta: {batch_result.memory_delta_mb:.2f}MB")
    print(f"  Throughput: {num_samples / batch_result.total_time_seconds:.1f} samples/sec")
    print()

    # Determine which is better
    if streaming_result.total_time_seconds < batch_result.total_time_seconds:
        print("✓ Streaming approach is faster")
    else:
        print("✓ Batch approach is faster")

    if streaming_result.memory_delta_mb < batch_result.memory_delta_mb:
        print("✓ Streaming approach uses less memory")
    else:
        print("✓ Batch approach uses less memory")


def demo_context_manager_profiling():
    """Demonstrate profiling using context managers."""
    print("\n=== Context Manager Profiling Demo ===\n")

    config = get_preset("moderate")
    num_samples = 30

    print("Using context managers to profile different operations:")
    print(f"Processing {num_samples} samples with timing and memory measurement\n")

    # Time measurement
    print("1. Timing measurement:")
    with measure_time() as timer:
        params_list = []
        for sample_id in range(num_samples):
            params = gen_augmentation_params(sample_id, config)
            params_list.append(params)
            time.sleep(0.001)  # Simulate processing

    print(f"   Total time: {timer['elapsed_seconds']:.3f}s")
    print(f"   Throughput: {num_samples / timer['elapsed_seconds']:.1f} samples/sec")
    print()

    # Memory measurement
    print("2. Memory measurement:")
    with measure_memory() as memory:
        large_data = []
        for i in range(1000):
            # Create some data to show memory usage
            data = [gen_augmentation_params(i % 10, config) for _ in range(10)]
            large_data.extend(data)

    print(f"   Memory delta: {memory['delta_mb']:.2f}MB")
    print(f"   Final RSS: {memory['final_rss_mb']:.2f}MB")
    print()

    # Combined measurement
    print("3. Combined timing and memory:")
    with measure_time() as timer, measure_memory() as memory:
        # Process with batch processing
        batch_config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=10)
        processor = BatchProcessor(BatchStrategy.SEQUENTIAL, batch_config)

        param_stream = stream_augmentation_chain(num_samples, config)

        processed_batches = []
        for batch in processor.process_stream(param_stream):
            processed_batches.append(batch)
            time.sleep(0.005)  # Simulate batch processing

    print(f"   Processing time: {timer['elapsed_seconds']:.3f}s")
    print(f"   Memory usage: {memory['delta_mb']:.2f}MB")
    print(f"   Batches processed: {len(processed_batches)}")
    print(
        f"   Batch throughput: {len(processed_batches) / timer['elapsed_seconds']:.1f} batches/sec"
    )


def demo_comparative_profiling():
    """Demonstrate comparative profiling of different approaches."""
    print("\n=== Comparative Profiling Demo ===\n")

    profiler = PerformanceProfiler(enable_memory_tracking=True)
    config = get_preset("moderate")
    num_samples = 40

    approaches = [
        ("sequential", "Sequential generation"),
        ("streaming", "Streaming generation"),
        ("distributed_rank0", "Distributed generation (rank 0)"),
        ("distributed_rank1", "Distributed generation (rank 1)"),
    ]

    print(f"Comparing different approaches for {num_samples} samples:")
    print("Measuring performance characteristics of each approach\n")

    results = {}

    for approach_name, description in approaches:
        print(f"Testing: {description}")

        profiler.start_profiling(approach_name)

        if approach_name == "sequential":
            params_list = []
            for sample_id in range(num_samples):
                params = gen_augmentation_params(sample_id, config)
                params_list.append(params)

        elif approach_name == "streaming":
            param_stream = stream_augmentation_chain(num_samples, config)
            params_list = list(param_stream)

        elif approach_name == "distributed_rank0":
            params_list = []
            for sample_id in range(num_samples):
                params = gen_distributed_augmentation_params(sample_id, 0, config, world_size=2)
                params_list.append(params)

        elif approach_name == "distributed_rank1":
            params_list = []
            for sample_id in range(num_samples):
                params = gen_distributed_augmentation_params(sample_id, 1, config, world_size=2)
                params_list.append(params)

        result = profiler.end_profiling(approach_name)
        results[approach_name] = result

        throughput = num_samples / result.total_time_seconds
        print(f"  Time: {result.total_time_seconds:.4f}s")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  Memory: {result.memory_delta_mb:.2f}MB")
        print()

    # Find best performing approach
    best_time = min(results.values(), key=lambda x: x.total_time_seconds)
    best_memory = min(results.values(), key=lambda x: abs(x.memory_delta_mb))

    best_time_approach = [k for k, v in results.items() if v == best_time][0]
    best_memory_approach = [k for k, v in results.items() if v == best_memory][0]

    print("Performance Summary:")
    print("-" * 30)
    print(f"Fastest approach: {best_time_approach}")
    print(f"Most memory efficient: {best_memory_approach}")

    # Calculate relative performance
    baseline = results["sequential"]
    for approach_name, result in results.items():
        if approach_name != "sequential":
            speedup = baseline.total_time_seconds / result.total_time_seconds
            memory_ratio = (
                result.memory_delta_mb / baseline.memory_delta_mb
                if baseline.memory_delta_mb != 0
                else 1
            )
            print(f"{approach_name}: {speedup:.2f}x speed, {memory_ratio:.2f}x memory")


def demo_detailed_operation_profiling():
    """Demonstrate detailed profiling of individual operations."""
    print("\n=== Detailed Operation Profiling Demo ===\n")

    profiler = PerformanceProfiler(enable_memory_tracking=True)
    config = get_preset("aggressive")

    print("Profiling individual components of augmentation generation:")
    print("Breaking down the time spent in different operations\n")

    # Profile different aspects of parameter generation
    operations = [
        ("seed_generation", lambda: "sample_42_depth_10"),
        ("config_access", lambda: config.rotation_range),
        ("random_generation", lambda: gen_augmentation_params(42, config)),
        ("parameter_extraction", lambda: gen_augmentation_params(42, config)["rotation"]),
    ]

    for operation_name, operation_func in operations:
        print(f"Profiling: {operation_name}")

        profiler.start_profiling(operation_name)

        # Run operation multiple times for better measurement
        for _ in range(100):
            result = operation_func()

        profile_result = profiler.end_profiling(operation_name)

        avg_time_microseconds = profile_result.avg_time_per_call_ms * 1000
        print(f"  Average time per call: {avg_time_microseconds:.1f}μs")
        print(f"  Total time (100 calls): {profile_result.total_time_seconds:.4f}s")
        print(f"  Memory impact: {profile_result.memory_delta_mb:.3f}MB")
        print()

    # Get complete profiling summary
    profile_summary = profiler.get_profile_summary()

    print("Complete Profiling Summary:")
    print("-" * 40)

    # Sort by total time
    sorted_operations = sorted(
        profile_summary.items(), key=lambda x: x[1].total_time_seconds, reverse=True
    )

    for operation, result in sorted_operations:
        percentage = (
            result.total_time_seconds / sum(r.total_time_seconds for r in profile_summary.values())
        ) * 100
        print(f"{operation}: {percentage:.1f}% of total time")


def demo_performance_regression_detection():
    """Demonstrate performance regression detection."""
    print("\n=== Performance Regression Detection Demo ===\n")

    config = get_preset("moderate")
    num_samples = 50

    print("Simulating performance regression detection:")
    print("Comparing current performance against baseline\n")

    # Establish baseline performance
    print("1. Establishing baseline performance...")
    baseline_times = []

    for run in range(5):
        with measure_time() as timer:
            params_list = []
            for sample_id in range(num_samples):
                params = gen_augmentation_params(sample_id, config)
                params_list.append(params)
        baseline_times.append(timer["elapsed_seconds"])

    baseline_avg = sum(baseline_times) / len(baseline_times)
    baseline_throughput = num_samples / baseline_avg

    print(f"   Baseline average time: {baseline_avg:.4f}s")
    print(f"   Baseline throughput: {baseline_throughput:.1f} samples/sec")
    print()

    # Simulate current performance (with slight regression)
    print("2. Testing current performance...")
    current_times = []

    for run in range(5):
        with measure_time() as timer:
            params_list = []
            for sample_id in range(num_samples):
                params = gen_augmentation_params(sample_id, config)
                params_list.append(params)
                # Simulate slight performance regression
                time.sleep(0.00005)  # 0.05ms additional delay
        current_times.append(timer["elapsed_seconds"])

    current_avg = sum(current_times) / len(current_times)
    current_throughput = num_samples / current_avg

    print(f"   Current average time: {current_avg:.4f}s")
    print(f"   Current throughput: {current_throughput:.1f} samples/sec")
    print()

    # Analyze regression
    print("3. Regression analysis:")
    performance_change = (current_avg - baseline_avg) / baseline_avg * 100
    throughput_change = (current_throughput - baseline_throughput) / baseline_throughput * 100

    print(f"   Time change: {performance_change:+.1f}%")
    print(f"   Throughput change: {throughput_change:+.1f}%")

    # Determine if regression is significant
    regression_threshold = 5.0  # 5% threshold

    if abs(performance_change) > regression_threshold:
        if performance_change > 0:
            print(f"   🚨 Performance regression detected! ({performance_change:.1f}% slower)")
        else:
            print(
                f"   📈 Performance improvement detected! ({abs(performance_change):.1f}% faster)"
            )
    else:
        print(f"   ✅ Performance within acceptable range (±{regression_threshold}%)")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Benchmarking: Performance Profiling")
    print("=" * 70)

    demo_basic_profiling()
    demo_function_profiling_decorator()
    demo_streaming_profiling()
    demo_context_manager_profiling()
    demo_comparative_profiling()
    demo_detailed_operation_profiling()
    demo_performance_regression_detection()

    print("\n" + "=" * 70)
    print("Demo completed! Use profiling to identify performance bottlenecks.")
    print("=" * 70 + "\n")
