#!/usr/bin/env python3
"""
Benchmarking Example: Performance Optimization Workflow

This example demonstrates a complete performance optimization workflow,
from initial profiling through iterative improvements to final validation.

Requirements addressed: 3.1, 3.5, 6.4 (optimization workflow)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.batch import BatchConfig, BatchStrategy
from src.benchmark import BenchmarkConfig, BenchmarkRunner, PerformanceProfiler
from src.distributed import gen_distributed_augmentation_params
from src.dpa import (
    AugmentationConfig,
    gen_augmentation_params,
    get_preset,
    stream_augmentation_chain,
)


class OptimizationWorkflow:
    """Complete optimization workflow for DPA performance."""

    def __init__(self):
        self.benchmark_config = BenchmarkConfig(
            iterations=30, warmup_iterations=5, measure_memory=True, measure_cpu=True
        )
        self.runner = BenchmarkRunner(self.benchmark_config)
        self.profiler = PerformanceProfiler(enable_memory_tracking=True)
        self.optimization_history = []

    def step1_baseline_measurement(self, num_samples=100):
        """Step 1: Establish baseline performance."""
        print("=== Step 1: Baseline Measurement ===\n")

        config = get_preset("moderate")

        print(f"Establishing baseline with {num_samples} samples using 'moderate' preset")
        print("This will be our reference point for optimization\n")

        # Measure baseline performance
        baseline_metrics = self.runner.benchmark_generation(num_samples, config)

        baseline_result = {
            "step": "baseline",
            "description": "Default moderate preset",
            "config": "moderate",
            "metrics": baseline_metrics,
            "samples": num_samples,
        }

        self.optimization_history.append(baseline_result)

        print("Baseline Results:")
        print(f"  Throughput: {baseline_metrics.throughput_samples_per_second:.1f} samples/sec")
        print(f"  Average latency: {baseline_metrics.avg_latency_ms:.2f}ms")
        print(f"  Memory usage: {baseline_metrics.memory_usage_mb:.1f}MB")
        print(f"  CPU usage: {baseline_metrics.cpu_usage_percent:.1f}%")
        print(f"  Total time: {baseline_metrics.total_time_seconds:.3f}s")

        return baseline_result

    def step2_identify_bottlenecks(self, num_samples=100):
        """Step 2: Profile to identify performance bottlenecks."""
        print("\n=== Step 2: Bottleneck Identification ===\n")

        config = get_preset("moderate")

        print("Profiling individual operations to identify bottlenecks")
        print("Breaking down the augmentation generation process\n")

        # Profile different aspects of generation
        operations = [
            ("parameter_generation", lambda: gen_augmentation_params(42, config)),
            ("streaming_generation", lambda: list(stream_augmentation_chain(10, config))),
            ("distributed_generation", lambda: gen_distributed_augmentation_params(42, 0, config)),
            ("config_access", lambda: (config.rotation_range, config.brightness_range)),
        ]

        bottleneck_results = {}

        for operation_name, operation_func in operations:
            print(f"Profiling: {operation_name}")

            self.profiler.start_profiling(operation_name)

            # Run operation multiple times
            iterations = 50 if "streaming" not in operation_name else 5
            for _ in range(iterations):
                result = operation_func()

            profile_result = self.profiler.end_profiling(operation_name)
            bottleneck_results[operation_name] = profile_result

            print(f"  Average time: {profile_result.avg_time_per_call_ms:.3f}ms")
            print(f"  Total time: {profile_result.total_time_seconds:.4f}s")
            print(f"  Memory delta: {profile_result.memory_delta_mb:.3f}MB")

        # Identify the slowest operation
        slowest_operation = max(bottleneck_results.items(), key=lambda x: x[1].total_time_seconds)

        print(f"\nBottleneck identified: {slowest_operation[0]}")
        print(f"This operation takes {slowest_operation[1].total_time_seconds:.4f}s total")

        return bottleneck_results

    def step3_configuration_optimization(self, num_samples=100):
        """Step 3: Optimize augmentation configuration."""
        print("\n=== Step 3: Configuration Optimization ===\n")

        print("Testing different augmentation configurations for optimal performance")
        print("Balancing augmentation quality with generation speed\n")

        # Test different configurations
        test_configs = {
            "lightweight": AugmentationConfig(
                rotation_range=(-30, 30),
                brightness_range=(0.8, 1.2),
                noise_range=(0, 0.1),
                scale_range=(0.9, 1.1),
                contrast_range=(0.9, 1.1),
                augmentation_depth=5,  # Reduced depth for speed
            ),
            "balanced": AugmentationConfig(
                rotation_range=(-45, 45),
                brightness_range=(0.7, 1.3),
                noise_range=(0, 0.15),
                scale_range=(0.8, 1.2),
                contrast_range=(0.8, 1.2),
                augmentation_depth=8,  # Balanced depth
            ),
            "optimized": AugmentationConfig(
                rotation_range=(-60, 60),
                brightness_range=(0.6, 1.4),
                noise_range=(0, 0.2),
                scale_range=(0.7, 1.3),
                contrast_range=(0.7, 1.3),
                augmentation_depth=12,  # Higher quality
            ),
        }

        config_results = {}

        for config_name, config in test_configs.items():
            print(f"Testing {config_name} configuration...")

            metrics = self.runner.benchmark_generation(num_samples, config)
            config_results[config_name] = metrics

            result = {
                "step": "config_optimization",
                "description": f"{config_name} configuration",
                "config": config_name,
                "metrics": metrics,
                "samples": num_samples,
            }
            self.optimization_history.append(result)

            print(f"  Throughput: {metrics.throughput_samples_per_second:.1f} samples/sec")
            print(f"  Latency: {metrics.avg_latency_ms:.2f}ms")
            print(f"  Memory: {metrics.memory_usage_mb:.1f}MB")

        # Find best configuration
        best_config = max(config_results.items(), key=lambda x: x[1].throughput_samples_per_second)

        print(f"\nBest configuration: {best_config[0]}")
        print(f"Throughput: {best_config[1].throughput_samples_per_second:.1f} samples/sec")

        return best_config[0], test_configs[best_config[0]]

    def step4_batch_optimization(self, num_samples=200, optimal_config=None):
        """Step 4: Optimize batch processing strategy."""
        print("\n=== Step 4: Batch Processing Optimization ===\n")

        if optimal_config is None:
            optimal_config = get_preset("moderate")

        print("Optimizing batch processing strategy for maximum throughput")
        print("Testing different batch sizes and strategies\n")

        # Test different batch configurations
        batch_configs = {
            "small_sequential": BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=5),
            "medium_sequential": BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=15),
            "large_sequential": BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=30),
            "memory_optimized": BatchConfig(
                strategy=BatchStrategy.MEMORY_OPTIMIZED, max_memory_mb=150, min_batch_size=5
            ),
            "adaptive": BatchConfig(
                strategy=BatchStrategy.ADAPTIVE, batch_size=10, adaptive_sizing=True
            ),
        }

        batch_results = {}

        for batch_name, batch_config in batch_configs.items():
            print(f"Testing {batch_name} batch strategy...")

            metrics = self.runner.benchmark_batch_processing(num_samples, batch_config)
            batch_results[batch_name] = metrics

            result = {
                "step": "batch_optimization",
                "description": f"{batch_name} batch strategy",
                "config": batch_name,
                "metrics": metrics,
                "samples": num_samples,
            }
            self.optimization_history.append(result)

            print(f"  Throughput: {metrics.throughput_samples_per_second:.1f} samples/sec")
            print(f"  Memory: {metrics.memory_usage_mb:.1f}MB")

        # Find best batch strategy
        best_batch = max(batch_results.items(), key=lambda x: x[1].throughput_samples_per_second)

        print(f"\nBest batch strategy: {best_batch[0]}")
        print(f"Throughput: {best_batch[1].throughput_samples_per_second:.1f} samples/sec")

        return best_batch[0], batch_configs[best_batch[0]]

    def step5_fine_tuning(self, num_samples=100, optimal_config=None, optimal_batch_config=None):
        """Step 5: Fine-tune the optimal configuration."""
        print("\n=== Step 5: Fine-Tuning ===\n")

        if optimal_config is None:
            optimal_config = get_preset("moderate")

        print("Fine-tuning the optimal configuration for maximum performance")
        print("Making small adjustments to squeeze out extra performance\n")

        # Create variations of the optimal configuration
        base_depth = optimal_config.augmentation_depth

        fine_tune_configs = {
            "depth_reduced": AugmentationConfig(
                rotation_range=optimal_config.rotation_range,
                brightness_range=optimal_config.brightness_range,
                noise_range=optimal_config.noise_range,
                scale_range=optimal_config.scale_range,
                contrast_range=optimal_config.contrast_range,
                augmentation_depth=max(5, base_depth - 3),
            ),
            "depth_optimal": optimal_config,
            "depth_increased": AugmentationConfig(
                rotation_range=optimal_config.rotation_range,
                brightness_range=optimal_config.brightness_range,
                noise_range=optimal_config.noise_range,
                scale_range=optimal_config.scale_range,
                contrast_range=optimal_config.contrast_range,
                augmentation_depth=base_depth + 3,
            ),
        }

        fine_tune_results = {}

        for config_name, config in fine_tune_configs.items():
            print(f"Testing {config_name} (depth={config.augmentation_depth})...")

            metrics = self.runner.benchmark_generation(num_samples, config)
            fine_tune_results[config_name] = metrics

            result = {
                "step": "fine_tuning",
                "description": f"{config_name} depth tuning",
                "config": config_name,
                "metrics": metrics,
                "samples": num_samples,
            }
            self.optimization_history.append(result)

            print(f"  Throughput: {metrics.throughput_samples_per_second:.1f} samples/sec")
            print(f"  Latency: {metrics.avg_latency_ms:.2f}ms")

        # Find best fine-tuned configuration
        best_fine_tuned = max(
            fine_tune_results.items(), key=lambda x: x[1].throughput_samples_per_second
        )

        print(f"\nBest fine-tuned configuration: {best_fine_tuned[0]}")
        print(f"Throughput: {best_fine_tuned[1].throughput_samples_per_second:.1f} samples/sec")

        return best_fine_tuned[0], fine_tune_configs[best_fine_tuned[0]]

    def step6_validation(self, num_samples=500, final_config=None):
        """Step 6: Validate the optimized configuration."""
        print("\n=== Step 6: Validation ===\n")

        if final_config is None:
            final_config = get_preset("moderate")

        print("Validating the optimized configuration with larger dataset")
        print("Ensuring performance improvements are consistent at scale\n")

        # Test with larger sample size
        print(f"Testing optimized configuration with {num_samples} samples...")

        final_metrics = self.runner.benchmark_generation(num_samples, final_config)

        result = {
            "step": "validation",
            "description": "Final optimized configuration validation",
            "config": "optimized_final",
            "metrics": final_metrics,
            "samples": num_samples,
        }
        self.optimization_history.append(result)

        print("Final Results:")
        print(f"  Throughput: {final_metrics.throughput_samples_per_second:.1f} samples/sec")
        print(f"  Average latency: {final_metrics.avg_latency_ms:.2f}ms")
        print(f"  Memory usage: {final_metrics.memory_usage_mb:.1f}MB")
        print(f"  CPU usage: {final_metrics.cpu_usage_percent:.1f}%")
        print(f"  Total time: {final_metrics.total_time_seconds:.3f}s")

        return final_metrics

    def generate_optimization_report(self):
        """Generate a comprehensive optimization report."""
        print("\n=== Optimization Report ===\n")

        if not self.optimization_history:
            print("No optimization data available.")
            return

        baseline = self.optimization_history[0]
        final = self.optimization_history[-1]

        print("OPTIMIZATION SUMMARY")
        print("-" * 40)

        # Calculate improvements
        throughput_improvement = (
            (
                final["metrics"].throughput_samples_per_second
                - baseline["metrics"].throughput_samples_per_second
            )
            / baseline["metrics"].throughput_samples_per_second
        ) * 100

        latency_improvement = (
            (baseline["metrics"].avg_latency_ms - final["metrics"].avg_latency_ms)
            / baseline["metrics"].avg_latency_ms
        ) * 100

        memory_change = (
            (final["metrics"].memory_usage_mb - baseline["metrics"].memory_usage_mb)
            / baseline["metrics"].memory_usage_mb
        ) * 100

        print(
            f"Baseline throughput: {baseline['metrics'].throughput_samples_per_second:.1f} samples/sec"
        )
        print(f"Final throughput: {final['metrics'].throughput_samples_per_second:.1f} samples/sec")
        print(f"Throughput improvement: {throughput_improvement:+.1f}%")
        print()

        print(f"Baseline latency: {baseline['metrics'].avg_latency_ms:.2f}ms")
        print(f"Final latency: {final['metrics'].avg_latency_ms:.2f}ms")
        print(f"Latency improvement: {latency_improvement:+.1f}%")
        print()

        print(f"Memory usage change: {memory_change:+.1f}%")
        print()

        # Show optimization steps
        print("OPTIMIZATION STEPS")
        print("-" * 40)

        for i, step in enumerate(self.optimization_history, 1):
            if i == 1:
                improvement = "baseline"
            else:
                prev_throughput = self.optimization_history[i - 2][
                    "metrics"
                ].throughput_samples_per_second
                curr_throughput = step["metrics"].throughput_samples_per_second
                step_improvement = ((curr_throughput - prev_throughput) / prev_throughput) * 100
                improvement = f"{step_improvement:+.1f}%"

            print(
                f"{i}. {step['description']}: {step['metrics'].throughput_samples_per_second:.1f} samples/sec ({improvement})"
            )

        # Recommendations
        print("\nRECOMMENDations")
        print("-" * 40)

        if throughput_improvement > 10:
            print("✅ Significant performance improvement achieved!")
        elif throughput_improvement > 0:
            print("✅ Performance improvement achieved.")
        else:
            print("⚠️  No significant performance improvement. Consider different approaches.")

        if latency_improvement > 10:
            print("✅ Latency significantly reduced.")

        if abs(memory_change) < 10:
            print("✅ Memory usage remained stable during optimization.")
        elif memory_change > 10:
            print("⚠️  Memory usage increased. Monitor for memory constraints.")

        print(
            f"\nFinal recommendation: Use the optimized configuration for {throughput_improvement:+.1f}% better performance."
        )


def demo_complete_optimization_workflow():
    """Demonstrate the complete optimization workflow."""
    print("=== Complete Performance Optimization Workflow ===\n")

    print("This demo shows a systematic approach to optimizing DPA performance")
    print("Following industry best practices for performance optimization\n")

    # Create workflow instance
    workflow = OptimizationWorkflow()

    # Execute optimization steps
    num_samples = 100  # Use smaller number for demo

    # Step 1: Baseline
    baseline = workflow.step1_baseline_measurement(num_samples)

    # Step 2: Identify bottlenecks
    bottlenecks = workflow.step2_identify_bottlenecks(num_samples)

    # Step 3: Optimize configuration
    optimal_config_name, optimal_config = workflow.step3_configuration_optimization(num_samples)

    # Step 4: Optimize batch processing
    optimal_batch_name, optimal_batch_config = workflow.step4_batch_optimization(
        num_samples * 2, optimal_config
    )

    # Step 5: Fine-tune
    final_config_name, final_config = workflow.step5_fine_tuning(num_samples, optimal_config)

    # Step 6: Validate
    final_metrics = workflow.step6_validation(num_samples * 3, final_config)

    # Generate report
    workflow.generate_optimization_report()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Benchmarking: Performance Optimization Workflow")
    print("=" * 70)

    demo_complete_optimization_workflow()

    print("\n" + "=" * 70)
    print("Demo completed! Follow this workflow to optimize your DPA performance.")
    print("=" * 70 + "\n")
