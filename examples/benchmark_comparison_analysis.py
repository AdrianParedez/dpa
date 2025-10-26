#!/usr/bin/env python3
"""
Benchmarking Example: Comparison Analysis

This example demonstrates how to perform comprehensive benchmark comparisons
between different configurations, strategies, and approaches.

Requirements addressed: 3.5, 6.4 (benchmark comparisons and analysis)
"""

import sys
import statistics
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark import BenchmarkRunner, BenchmarkConfig, PerformanceMetrics
from src.dpa import get_preset, AugmentationConfig
from src.batch import BatchStrategy, BatchConfig
from src.distributed import DistributedRangeSplitter


def demo_configuration_comparison():
    """Demonstrate comparison between different augmentation configurations."""
    print("=== Configuration Comparison Demo ===\n")
    
    # Define different configurations to compare
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
        "custom_heavy": AugmentationConfig(
            rotation_range=(-90, 90),
            brightness_range=(0.5, 1.5),
            noise_range=(0, 0.3),
            scale_range=(0.5, 1.5),
            contrast_range=(0.5, 1.5),
            augmentation_depth=20
        )
    }
    
    num_samples = 100
    benchmark_config = BenchmarkConfig(
        iterations=50,
        warmup_iterations=5,
        measure_memory=True,
        measure_cpu=True
    )
    
    print(f"Comparing {len(configs)} configurations with {num_samples} samples each:")
    print("Measuring generation performance for different augmentation intensities\n")
    
    runner = BenchmarkRunner(benchmark_config)
    results = {}
    
    for config_name, config in configs.items():
        print(f"Benchmarking {config_name} configuration...")
        
        metrics = runner.benchmark_generation(num_samples, config)
        results[config_name] = metrics
        
        print(f"  Throughput: {metrics.throughput_samples_per_second:.1f} samples/sec")
        print(f"  Avg latency: {metrics.avg_latency_ms:.2f}ms")
        print(f"  Memory usage: {metrics.memory_usage_mb:.1f}MB")
        print(f"  CPU usage: {metrics.cpu_usage_percent:.1f}%")
        print()
    
    # Generate comparison report
    comparison_report = runner.compare_configurations([
        {"name": name, "config": config, "metrics": metrics}
        for name, (config, metrics) in zip(configs.keys(), zip(configs.values(), results.values()))
    ])
    
    print("Configuration Comparison Summary:")
    print("-" * 50)
    
    # Find best and worst performers
    best_throughput = max(results.items(), key=lambda x: x[1].throughput_samples_per_second)
    worst_throughput = min(results.items(), key=lambda x: x[1].throughput_samples_per_second)
    lowest_memory = min(results.items(), key=lambda x: x[1].memory_usage_mb)
    highest_memory = max(results.items(), key=lambda x: x[1].memory_usage_mb)
    
    print(f"Fastest configuration: {best_throughput[0]} ({best_throughput[1].throughput_samples_per_second:.1f} samples/sec)")
    print(f"Slowest configuration: {worst_throughput[0]} ({worst_throughput[1].throughput_samples_per_second:.1f} samples/sec)")
    print(f"Most memory efficient: {lowest_memory[0]} ({lowest_memory[1].memory_usage_mb:.1f}MB)")
    print(f"Most memory intensive: {highest_memory[0]} ({highest_memory[1].memory_usage_mb:.1f}MB)")
    
    # Performance ratio analysis
    performance_ratio = best_throughput[1].throughput_samples_per_second / worst_throughput[1].throughput_samples_per_second
    
    if lowest_memory[1].memory_usage_mb > 0:
        memory_ratio = highest_memory[1].memory_usage_mb / lowest_memory[1].memory_usage_mb
        print(f"Memory usage range: {memory_ratio:.1f}x difference")
    else:
        print(f"Memory usage range: All configurations use minimal memory")
    
    print(f"Performance range: {performance_ratio:.1f}x difference")


def demo_batch_strategy_comparison():
    """Demonstrate comparison between different batch processing strategies."""
    print("\n=== Batch Strategy Comparison Demo ===\n")
    
    num_samples = 200
    config = get_preset("moderate")
    
    # Define batch configurations to compare
    batch_configs = {
        "sequential_small": BatchConfig(
            strategy=BatchStrategy.SEQUENTIAL,
            batch_size=5
        ),
        "sequential_large": BatchConfig(
            strategy=BatchStrategy.SEQUENTIAL,
            batch_size=20
        ),
        "round_robin": BatchConfig(
            strategy=BatchStrategy.ROUND_ROBIN,
            batch_size=10
        ),
        "memory_optimized": BatchConfig(
            strategy=BatchStrategy.MEMORY_OPTIMIZED,
            max_memory_mb=100,
            min_batch_size=3
        ),
        "adaptive": BatchConfig(
            strategy=BatchStrategy.ADAPTIVE,
            batch_size=8,
            adaptive_sizing=True
        )
    }
    
    benchmark_config = BenchmarkConfig(
        iterations=30,
        warmup_iterations=3,
        measure_memory=True
    )
    
    print(f"Comparing {len(batch_configs)} batch strategies with {num_samples} samples:")
    print("Measuring batch processing performance\n")
    
    runner = BenchmarkRunner(benchmark_config)
    results = {}
    
    for strategy_name, batch_config in batch_configs.items():
        print(f"Benchmarking {strategy_name} strategy...")
        
        metrics = runner.benchmark_batch_processing(num_samples, batch_config)
        results[strategy_name] = metrics
        
        print(f"  Throughput: {metrics.throughput_samples_per_second:.1f} samples/sec")
        print(f"  Total time: {metrics.total_time_seconds:.3f}s")
        print(f"  Memory usage: {metrics.memory_usage_mb:.1f}MB")
        print()
    
    # Analyze results
    print("Batch Strategy Analysis:")
    print("-" * 40)
    
    # Sort by throughput
    sorted_results = sorted(results.items(), key=lambda x: x[1].throughput_samples_per_second, reverse=True)
    
    print("Ranking by throughput:")
    for i, (strategy, metrics) in enumerate(sorted_results, 1):
        print(f"  {i}. {strategy}: {metrics.throughput_samples_per_second:.1f} samples/sec")
    
    print()
    
    # Memory efficiency ranking
    memory_sorted = sorted(results.items(), key=lambda x: x[1].memory_usage_mb)
    
    print("Ranking by memory efficiency:")
    for i, (strategy, metrics) in enumerate(memory_sorted, 1):
        print(f"  {i}. {strategy}: {metrics.memory_usage_mb:.1f}MB")
    
    # Recommendations
    print("\nRecommendations:")
    best_throughput = sorted_results[0]
    best_memory = memory_sorted[0]
    
    print(f"  For maximum throughput: {best_throughput[0]}")
    print(f"  For memory efficiency: {best_memory[0]}")
    
    if best_throughput[0] == best_memory[0]:
        print(f"  Overall best: {best_throughput[0]} (best in both categories)")
    else:
        # Find balanced option
        balanced_scores = {}
        max_throughput = max(m.throughput_samples_per_second for m in results.values())
        min_memory = min(m.memory_usage_mb for m in results.values())
        
        for strategy, metrics in results.items():
            throughput_score = metrics.throughput_samples_per_second / max_throughput
            if metrics.memory_usage_mb > 0:
                memory_score = min_memory / metrics.memory_usage_mb
            else:
                memory_score = 1.0  # All have same minimal memory usage
            balanced_scores[strategy] = (throughput_score + memory_score) / 2
        
        best_balanced = max(balanced_scores.items(), key=lambda x: x[1])
        print(f"  For balanced performance: {best_balanced[0]}")


def demo_distributed_scaling_analysis():
    """Demonstrate analysis of distributed training scaling performance."""
    print("\n=== Distributed Scaling Analysis Demo ===\n")
    
    total_samples = 1000
    config = get_preset("moderate")
    
    # Test different world sizes
    world_sizes = [1, 2, 4, 8, 16]
    
    benchmark_config = BenchmarkConfig(
        iterations=20,
        warmup_iterations=2,
        measure_memory=True
    )
    
    print(f"Analyzing distributed scaling with {total_samples} total samples:")
    print("Testing different world sizes for distributed training\n")
    
    runner = BenchmarkRunner(benchmark_config)
    scaling_results = {}
    
    for world_size in world_sizes:
        print(f"Testing world_size = {world_size}...")
        
        # Calculate samples per rank
        splitter = DistributedRangeSplitter(total_samples, world_size)
        samples_per_rank = []
        
        for rank in range(world_size):
            start, end = splitter.get_rank_range(rank)
            samples_per_rank.append(end - start)
        
        avg_samples_per_rank = sum(samples_per_rank) / len(samples_per_rank)
        
        # Benchmark generation for one rank (simulating distributed performance)
        metrics = runner.benchmark_generation(int(avg_samples_per_rank), config)
        
        scaling_results[world_size] = {
            'metrics': metrics,
            'samples_per_rank': avg_samples_per_rank,
            'total_samples': total_samples,
            'theoretical_speedup': world_size
        }
        
        print(f"  Samples per rank: {avg_samples_per_rank:.1f}")
        print(f"  Time per rank: {metrics.total_time_seconds:.3f}s")
        print(f"  Throughput per rank: {metrics.throughput_samples_per_second:.1f} samples/sec")
        print(f"  Theoretical total throughput: {metrics.throughput_samples_per_second * world_size:.1f} samples/sec")
        print()
    
    # Scaling analysis
    print("Scaling Analysis:")
    print("-" * 30)
    
    baseline = scaling_results[1]['metrics']
    baseline_total_time = baseline.total_time_seconds * (total_samples / scaling_results[1]['samples_per_rank'])
    
    print(f"Baseline (world_size=1): {baseline_total_time:.3f}s total time")
    print()
    
    print("Scaling efficiency:")
    for world_size in world_sizes[1:]:  # Skip world_size=1
        result = scaling_results[world_size]
        
        # Calculate actual speedup
        actual_total_time = result['metrics'].total_time_seconds  # Time for one rank
        actual_speedup = baseline_total_time / actual_total_time
        theoretical_speedup = world_size
        efficiency = (actual_speedup / theoretical_speedup) * 100
        
        print(f"  World size {world_size:2d}: {actual_speedup:.1f}x speedup ({efficiency:.1f}% efficiency)")
    
    # Optimal world size recommendation
    print("\nRecommendations:")
    
    # Find the world size with best efficiency above 80%
    efficient_sizes = []
    for world_size in world_sizes[1:]:
        result = scaling_results[world_size]
        actual_total_time = result['metrics'].total_time_seconds
        actual_speedup = baseline_total_time / actual_total_time
        efficiency = (actual_speedup / world_size) * 100
        
        if efficiency >= 80:
            efficient_sizes.append((world_size, efficiency))
    
    if efficient_sizes:
        best_efficient = max(efficient_sizes, key=lambda x: x[0])  # Largest efficient size
        print(f"  Recommended world size: {best_efficient[0]} ({best_efficient[1]:.1f}% efficiency)")
    else:
        print("  Recommended world size: 1 (distributed overhead too high)")


def demo_performance_trend_analysis():
    """Demonstrate performance trend analysis over time."""
    print("\n=== Performance Trend Analysis Demo ===\n")
    
    config = get_preset("moderate")
    num_samples = 50
    
    # Simulate performance measurements over multiple "versions"
    versions = ["v1.0", "v1.1", "v1.2", "v1.3", "v1.4"]
    
    print("Simulating performance trend analysis across versions:")
    print("Tracking performance changes over time\n")
    
    benchmark_config = BenchmarkConfig(
        iterations=25,
        warmup_iterations=3,
        measure_memory=True
    )
    
    runner = BenchmarkRunner(benchmark_config)
    trend_data = {}
    
    # Simulate different performance characteristics for each version
    performance_modifiers = {
        "v1.0": 1.0,      # Baseline
        "v1.1": 0.95,     # 5% improvement
        "v1.2": 0.92,     # Additional 3% improvement
        "v1.3": 1.05,     # 5% regression (bug introduced)
        "v1.4": 0.88      # 12% improvement (optimization)
    }
    
    for version in versions:
        print(f"Benchmarking {version}...")
        
        # Get baseline metrics
        metrics = runner.benchmark_generation(num_samples, config)
        
        # Apply simulated performance modifier
        modifier = performance_modifiers[version]
        adjusted_metrics = PerformanceMetrics(
            throughput_samples_per_second=metrics.throughput_samples_per_second / modifier,
            avg_latency_ms=metrics.avg_latency_ms * modifier,
            memory_usage_mb=metrics.memory_usage_mb,
            cpu_usage_percent=metrics.cpu_usage_percent,
            total_time_seconds=metrics.total_time_seconds * modifier
        )
        
        trend_data[version] = adjusted_metrics
        
        print(f"  Throughput: {adjusted_metrics.throughput_samples_per_second:.1f} samples/sec")
        print(f"  Latency: {adjusted_metrics.avg_latency_ms:.2f}ms")
        print()
    
    # Trend analysis
    print("Performance Trend Analysis:")
    print("-" * 40)
    
    baseline_throughput = trend_data["v1.0"].throughput_samples_per_second
    
    print("Throughput changes:")
    for version in versions:
        throughput = trend_data[version].throughput_samples_per_second
        change = ((throughput - baseline_throughput) / baseline_throughput) * 100
        
        if change > 0:
            trend_indicator = "📈"
        elif change < -2:  # Significant regression
            trend_indicator = "📉"
        else:
            trend_indicator = "➡️"
        
        print(f"  {version}: {throughput:.1f} samples/sec ({change:+.1f}%) {trend_indicator}")
    
    # Identify regressions and improvements
    print("\nKey Changes:")
    
    for i in range(1, len(versions)):
        current_version = versions[i]
        previous_version = versions[i-1]
        
        current_throughput = trend_data[current_version].throughput_samples_per_second
        previous_throughput = trend_data[previous_version].throughput_samples_per_second
        
        change = ((current_throughput - previous_throughput) / previous_throughput) * 100
        
        if abs(change) > 2:  # Significant change
            if change > 0:
                print(f"  {previous_version} → {current_version}: {change:+.1f}% improvement")
            else:
                print(f"  {previous_version} → {current_version}: {change:+.1f}% regression ⚠️")
    
    # Overall trend
    final_throughput = trend_data[versions[-1]].throughput_samples_per_second
    overall_change = ((final_throughput - baseline_throughput) / baseline_throughput) * 100
    
    print(f"\nOverall trend ({versions[0]} → {versions[-1]}): {overall_change:+.1f}%")


def demo_comprehensive_benchmark_report():
    """Demonstrate generation of a comprehensive benchmark report."""
    print("\n=== Comprehensive Benchmark Report Demo ===\n")
    
    print("Generating comprehensive benchmark report...")
    print("Including multiple configurations and analysis\n")
    
    # Test configurations
    test_configs = {
        "baseline": get_preset("mild"),
        "production": get_preset("moderate"),
        "stress_test": get_preset("aggressive")
    }
    
    # Test parameters
    sample_counts = [50, 100, 200]
    
    benchmark_config = BenchmarkConfig(
        iterations=20,
        warmup_iterations=2,
        measure_memory=True,
        measure_cpu=True,
        output_format="json"
    )
    
    runner = BenchmarkRunner(benchmark_config)
    
    # Collect comprehensive results
    comprehensive_results = {}
    
    for config_name, config in test_configs.items():
        comprehensive_results[config_name] = {}
        
        for sample_count in sample_counts:
            print(f"  Testing {config_name} with {sample_count} samples...")
            
            metrics = runner.benchmark_generation(sample_count, config)
            comprehensive_results[config_name][sample_count] = {
                'throughput': metrics.throughput_samples_per_second,
                'latency': metrics.avg_latency_ms,
                'memory': metrics.memory_usage_mb,
                'cpu': metrics.cpu_usage_percent,
                'total_time': metrics.total_time_seconds
            }
    
    # Generate report
    print("\n" + "=" * 60)
    print("COMPREHENSIVE BENCHMARK REPORT")
    print("=" * 60)
    
    # Executive summary
    print("\nEXECUTIVE SUMMARY")
    print("-" * 20)
    
    all_throughputs = []
    for config_results in comprehensive_results.values():
        for sample_results in config_results.values():
            all_throughputs.append(sample_results['throughput'])
    
    avg_throughput = statistics.mean(all_throughputs)
    max_throughput = max(all_throughputs)
    min_throughput = min(all_throughputs)
    
    print(f"Average throughput across all tests: {avg_throughput:.1f} samples/sec")
    print(f"Peak throughput: {max_throughput:.1f} samples/sec")
    print(f"Minimum throughput: {min_throughput:.1f} samples/sec")
    print(f"Performance range: {max_throughput/min_throughput:.1f}x")
    
    # Detailed results
    print("\nDETAILED RESULTS")
    print("-" * 20)
    
    for config_name, config_results in comprehensive_results.items():
        print(f"\n{config_name.upper()} Configuration:")
        print(f"{'Samples':<8} {'Throughput':<12} {'Latency':<10} {'Memory':<8} {'CPU':<6}")
        print("-" * 50)
        
        for sample_count, results in config_results.items():
            print(f"{sample_count:<8} {results['throughput']:<12.1f} "
                  f"{results['latency']:<10.2f} {results['memory']:<8.1f} "
                  f"{results['cpu']:<6.1f}")
    
    # Performance recommendations
    print("\nPERFORMANCE RECOMMENDATIONS")
    print("-" * 30)
    
    # Find best configuration for each metric
    best_throughput_config = None
    best_throughput_value = 0
    
    for config_name, config_results in comprehensive_results.items():
        avg_config_throughput = statistics.mean([r['throughput'] for r in config_results.values()])
        if avg_config_throughput > best_throughput_value:
            best_throughput_value = avg_config_throughput
            best_throughput_config = config_name
    
    print(f"1. For maximum throughput: Use '{best_throughput_config}' configuration")
    print(f"   Average throughput: {best_throughput_value:.1f} samples/sec")
    
    # Memory efficiency
    best_memory_config = None
    best_memory_value = float('inf')
    
    for config_name, config_results in comprehensive_results.items():
        avg_config_memory = statistics.mean([r['memory'] for r in config_results.values()])
        if avg_config_memory < best_memory_value:
            best_memory_value = avg_config_memory
            best_memory_config = config_name
    
    print(f"2. For memory efficiency: Use '{best_memory_config}' configuration")
    print(f"   Average memory usage: {best_memory_value:.1f}MB")
    
    # Scaling recommendations
    print(f"3. For large datasets: Performance scales well up to {max(sample_counts)} samples")
    
    # Save report to file
    report_file = Path(__file__).parent / "benchmark_report.json"
    with open(report_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {report_file}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Benchmarking: Comparison Analysis")
    print("=" * 70)
    
    demo_configuration_comparison()
    demo_batch_strategy_comparison()
    demo_distributed_scaling_analysis()
    demo_performance_trend_analysis()
    demo_comprehensive_benchmark_report()
    
    print("\n" + "=" * 70)
    print("Demo completed! Use comparison analysis to optimize your setup.")
    print("=" * 70 + "\n")