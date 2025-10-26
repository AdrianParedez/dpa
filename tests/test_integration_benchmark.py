"""
Integration tests for benchmarking functionality.

Tests profiling of existing DPA functions, performance regression detection,
and cross-configuration comparison accuracy as specified in requirements
3.5, 6.4, and 6.5.
"""

import json
import time

import pytest

from src.batch import BatchConfig, BatchProcessor, BatchStrategy
from src.benchmark import (
    BenchmarkConfig,
    BenchmarkRunner,
    ComparisonReport,
    MeasurementError,
    PerformanceMetrics,
    PerformanceProfiler,
    ProfilingError,
    benchmark_function,
    get_system_info,
    measure_memory,
    measure_time,
)
from src.distributed import (
    gen_distributed_augmentation_params,
    stream_distributed_augmentation_chain,
)
from src.dpa import AugmentationConfig, gen_augmentation_params, stream_augmentation_chain


class TestBenchmarkingIntegration:
    """Integration tests for benchmarking functionality."""

    def test_profiling_existing_dpa_functions(self):
        """Test profiling of existing DPA functions (Requirement 6.5)."""
        profiler = PerformanceProfiler(enable_memory_tracking=True)

        # Test profiling gen_augmentation_params
        @profiler.profile_function
        def profile_gen_params():
            config = AugmentationConfig(rotation_range=(-30, 30))
            return gen_augmentation_params(42, config)

        # Call the profiled function multiple times
        results = []
        for _i in range(10):
            result = profile_gen_params()
            results.append(result)

            # Verify function still works correctly
            assert "rotation" in result
            assert "brightness" in result
            assert "hash" in result
            assert -30 <= result["rotation"] <= 30

        # Get profiling summary
        summary = profiler.get_profile_summary()

        # Verify profiling data was collected
        assert len(summary) > 0

        # Find the profiled function
        profile_result = None
        for name, result in summary.items():
            if "profile_gen_params" in name:
                profile_result = result
                break

        assert profile_result is not None
        assert profile_result.call_count >= 10
        assert profile_result.total_time_seconds > 0
        assert profile_result.avg_time_per_call_ms > 0

        # Memory tracking should be enabled
        assert isinstance(profile_result.memory_delta_mb, float)

    def test_profiling_streaming_functions(self):
        """Test profiling of streaming functions."""
        profiler = PerformanceProfiler()

        # Profile streaming function
        with profiler.profile_context("stream_augmentation_test"):
            config = AugmentationConfig()
            stream = stream_augmentation_chain(100, config, chunk_size=20)
            chunks = list(stream)

            # Verify streaming worked correctly
            total_items = sum(len(chunk) for chunk in chunks)
            assert total_items == 100

        # Check profiling results
        summary = profiler.get_profile_summary()
        assert "stream_augmentation_test" in summary

        result = summary["stream_augmentation_test"]
        assert result.total_time_seconds > 0
        assert result.call_count == 1

    def test_profiling_distributed_functions(self):
        """Test profiling of distributed functions."""
        profiler = PerformanceProfiler()

        # Profile distributed parameter generation
        @profiler.profile_function
        def profile_distributed_gen():
            return gen_distributed_augmentation_params(
                sample_id=10, rank=0, world_size=4, base_seed=42
            )

        # Call multiple times
        for _ in range(5):
            result = profile_distributed_gen()
            assert "rank" in result
            assert result["rank"] == 0

        # Profile distributed streaming
        with profiler.profile_context("distributed_streaming_test"):
            stream = stream_distributed_augmentation_chain(
                num_samples=50, rank=0, world_size=2, base_seed=123
            )
            params_list = list(stream)
            assert len(params_list) == 25  # Rank 0 gets first half

        # Verify profiling results
        summary = profiler.get_profile_summary()

        # Check distributed generation profiling
        dist_gen_result = None
        for name, result in summary.items():
            if "profile_distributed_gen" in name:
                dist_gen_result = result
                break

        assert dist_gen_result is not None
        assert dist_gen_result.call_count >= 5  # May be higher due to test isolation issues

        # Check distributed streaming profiling
        assert "distributed_streaming_test" in summary
        stream_result = summary["distributed_streaming_test"]
        assert stream_result.total_time_seconds > 0

    def test_profiling_batch_processing_functions(self):
        """Test profiling of batch processing functions."""
        profiler = PerformanceProfiler()

        # Create test data generator
        def test_data_generator(count: int):
            for i in range(count):
                yield {"sample_id": i, "value": i * 2}

        # Profile batch processing
        with profiler.profile_context("batch_processing_test"):
            batch_config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=10)
            processor = BatchProcessor(BatchStrategy.SEQUENTIAL, batch_config)

            test_gen = test_data_generator(50)
            batches = list(processor.process_stream(test_gen))

            # Verify batch processing worked
            total_items = sum(len(batch) for batch in batches)
            assert total_items == 50

        # Check profiling results
        summary = profiler.get_profile_summary()
        assert "batch_processing_test" in summary

        result = summary["batch_processing_test"]
        assert result.total_time_seconds > 0
        assert result.call_count == 1

    def test_benchmark_runner_with_dpa_functions(self):
        """Test BenchmarkRunner with actual DPA functions."""
        config = BenchmarkConfig(
            iterations=5, warmup_iterations=2, measure_memory=True, measure_cpu=True
        )
        runner = BenchmarkRunner(config)

        # Benchmark parameter generation
        metrics = runner.benchmark_generation(num_samples=100)

        # Verify metrics are reasonable
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.throughput_samples_per_second > 0
        assert metrics.avg_latency_ms >= 0
        assert metrics.total_time_seconds > 0
        assert metrics.memory_usage_mb >= 0  # May be 0 if measurement disabled
        assert 0 <= metrics.cpu_usage_percent <= 100

        # Benchmark streaming operations
        streaming_metrics = runner.benchmark_streaming(num_samples=200, chunk_size=50)

        assert isinstance(streaming_metrics, PerformanceMetrics)
        assert streaming_metrics.throughput_samples_per_second > 0
        assert streaming_metrics.total_time_seconds > 0

        # Benchmark batch processing
        batch_metrics = runner.benchmark_batch_processing(num_samples=150)

        assert isinstance(batch_metrics, PerformanceMetrics)
        assert batch_metrics.throughput_samples_per_second > 0
        assert batch_metrics.total_time_seconds > 0

    def test_performance_regression_detection(self):
        """Test performance regression detection (Requirement 6.4)."""
        config = BenchmarkConfig(iterations=3, warmup_iterations=1)
        runner = BenchmarkRunner(config)

        # Create baseline and comparison configurations
        configs = [
            {
                "name": "baseline",
                "type": "generation",
                "num_samples": 100,
                "aug_config": AugmentationConfig(augmentation_depth=5),
            },
            {
                "name": "deeper_augmentation",
                "type": "generation",
                "num_samples": 100,
                "aug_config": AugmentationConfig(augmentation_depth=15),  # More expensive
            },
            {
                "name": "streaming_chunked",
                "type": "streaming",
                "num_samples": 100,
                "chunk_size": 25,
            },
        ]

        # Run comparison
        report = runner.compare_configurations(configs)

        # Verify report structure
        assert isinstance(report, ComparisonReport)
        assert report.baseline_config == "baseline"
        assert len(report.comparison_configs) == 2
        assert "deeper_augmentation" in report.comparison_configs
        assert "streaming_chunked" in report.comparison_configs

        # Verify metrics comparison
        assert "deeper_augmentation" in report.metrics_comparison
        assert "streaming_chunked" in report.metrics_comparison

        # Check that deeper augmentation shows performance impact
        deeper_metrics = report.metrics_comparison["deeper_augmentation"]
        assert "throughput_change_percent" in deeper_metrics
        assert "latency_change_percent" in deeper_metrics
        assert "memory_change_percent" in deeper_metrics
        assert "cpu_change_percent" in deeper_metrics

        # Deeper augmentation should likely be slower (positive latency change)
        # This is a regression detection test
        latency_change = deeper_metrics["latency_change_percent"]
        if latency_change > 20:  # Significant regression
            assert any("latency" in rec.lower() for rec in report.recommendations), (
                "Should recommend addressing latency regression"
            )

        # Verify performance improvements calculation
        assert "deeper_augmentation" in report.performance_improvements
        assert "streaming_chunked" in report.performance_improvements

        # Verify recommendations are generated
        assert isinstance(report.recommendations, list)
        assert len(report.recommendations) > 0

    def test_cross_configuration_comparison_accuracy(self):
        """Test cross-configuration comparison accuracy (Requirement 3.5)."""
        config = BenchmarkConfig(iterations=5, warmup_iterations=2)
        runner = BenchmarkRunner(config)

        # Create configurations with known performance characteristics
        configs = [
            {"name": "small_dataset", "type": "generation", "num_samples": 50},
            {
                "name": "large_dataset",
                "type": "generation",
                "num_samples": 200,  # 4x larger
            },
            {
                "name": "streaming_small_chunks",
                "type": "streaming",
                "num_samples": 100,
                "chunk_size": 10,
            },
            {
                "name": "streaming_large_chunks",
                "type": "streaming",
                "num_samples": 100,
                "chunk_size": 50,
            },
        ]

        # Run comparison
        report = runner.compare_configurations(configs)

        # Verify comparison accuracy
        large_dataset_metrics = report.metrics_comparison["large_dataset"]

        # Large dataset may have different latency characteristics
        # (latency is per-sample, so it might be similar or different)
        latency_change = large_dataset_metrics["latency_change_percent"]
        assert -50 <= latency_change <= 100, (
            f"Large dataset latency change should be reasonable: {latency_change}%"
        )

        # Throughput might be similar or slightly different
        throughput_change = large_dataset_metrics["throughput_change_percent"]
        assert -50 <= throughput_change <= 50, (
            f"Throughput change seems unreasonable: {throughput_change}%"
        )

        # Compare streaming configurations
        small_chunks_metrics = report.metrics_comparison["streaming_small_chunks"]
        large_chunks_metrics = report.metrics_comparison["streaming_large_chunks"]

        # Both should have reasonable performance characteristics
        for metrics in [small_chunks_metrics, large_chunks_metrics]:
            assert "absolute_metrics" in metrics
            abs_metrics = metrics["absolute_metrics"]
            assert abs_metrics["throughput_samples_per_second"] > 0
            assert abs_metrics["total_time_seconds"] > 0

        # Verify statistical accuracy by checking absolute metrics
        baseline_throughput = None
        for _config_name, metrics in report.metrics_comparison.items():
            abs_metrics = metrics["absolute_metrics"]
            current_throughput = abs_metrics["throughput_samples_per_second"]

            if baseline_throughput is None:
                # First comparison config - get baseline from runner history
                history = runner.get_results_history()
                baseline_entry = next(
                    (entry for entry in history if entry["config_name"] == "small_dataset"), None
                )
                if baseline_entry:
                    baseline_throughput = baseline_entry["metrics"].throughput_samples_per_second

            # Verify throughput calculations are consistent
            if baseline_throughput and baseline_throughput > 0:
                expected_change = (
                    (current_throughput - baseline_throughput) / baseline_throughput
                ) * 100
                actual_change = metrics["throughput_change_percent"]

                # Allow for small floating point differences
                assert abs(expected_change - actual_change) < 1.0, (
                    f"Throughput change calculation inaccurate: expected {expected_change:.2f}%, got {actual_change:.2f}%"
                )

    def test_benchmark_report_generation_completeness(self):
        """Test benchmark report generation completeness (Requirement 6.4)."""
        config = BenchmarkConfig(iterations=3, warmup_iterations=1)
        runner = BenchmarkRunner(config)

        # Create test configurations
        configs = [
            {"name": "baseline", "type": "generation", "num_samples": 50},
            {"name": "comparison", "type": "streaming", "num_samples": 50, "chunk_size": 25},
        ]

        # Run comparison
        report = runner.compare_configurations(configs)

        # Test different report formats
        formats_to_test = ["text", "json", "markdown"]

        for format_type in formats_to_test:
            report_content = runner.generate_performance_report(report, format_type)

            # Verify report content is not empty
            assert len(report_content) > 0, f"Empty report for format {format_type}"

            # Verify format-specific content
            if format_type == "json":
                # Should be valid JSON
                report_data = json.loads(report_content)
                assert "baseline_config" in report_data
                assert "metrics_comparison" in report_data
                assert "recommendations" in report_data
                assert "system_info" in report_data
                assert "benchmark_config" in report_data

                # Verify system info is included
                assert isinstance(report_data["system_info"], dict)

            elif format_type == "markdown":
                # Should have markdown formatting
                assert "# Performance Benchmark Report" in report_content
                assert "**Baseline Configuration:**" in report_content
                assert "## Performance Comparison" in report_content

                if report.recommendations:
                    assert "## Recommendations" in report_content

            elif format_type == "text":
                # Should have text formatting
                assert "Performance Benchmark Report" in report_content
                assert "Baseline Configuration:" in report_content
                assert "Performance Comparison:" in report_content

                # Should include metrics
                assert "Throughput Change:" in report_content
                assert "Latency Change:" in report_content

        # Test system info collection
        system_info = get_system_info()
        assert isinstance(system_info, dict)
        assert "cpu" in system_info or "error" in system_info
        assert "memory" in system_info or "error" in system_info

    def test_benchmark_integration_with_distributed_functions(self):
        """Test benchmarking integration with distributed functions."""
        config = BenchmarkConfig(iterations=3, warmup_iterations=1)
        runner = BenchmarkRunner(config)

        # Create configurations comparing distributed vs non-distributed
        configs = [
            {"name": "standard_generation", "type": "generation", "num_samples": 100},
            {
                "name": "distributed_rank0",
                "type": "generation",
                "num_samples": 100,
                "use_distributed": True,
                "rank": 0,
                "world_size": 4,
            },
        ]

        # Extend benchmark runner to handle distributed configurations
        original_benchmark_generation = runner.benchmark_generation

        def extended_benchmark_generation(num_samples, config=None, **kwargs):
            if kwargs.get("use_distributed"):
                # Benchmark distributed generation
                rank = kwargs.get("rank", 0)
                world_size = kwargs.get("world_size", 1)

                times = []
                for _ in range(runner.config.iterations):
                    start_time = time.time()

                    # Generate distributed parameters for this rank's range
                    from src.distributed import DistributedRangeSplitter

                    splitter = DistributedRangeSplitter(num_samples, world_size)
                    start_id, end_id = splitter.get_rank_range(rank)

                    for sample_id in range(start_id, end_id):
                        gen_distributed_augmentation_params(
                            sample_id=sample_id, rank=rank, world_size=world_size
                        )

                    end_time = time.time()
                    times.append(end_time - start_time)

                # Calculate metrics
                total_time = sum(times)
                avg_time = total_time / len(times)
                rank_samples = end_id - start_id
                throughput = (
                    (rank_samples * runner.config.iterations) / total_time if total_time > 0 else 0
                )

                return PerformanceMetrics(
                    throughput_samples_per_second=throughput,
                    avg_latency_ms=(avg_time / rank_samples * 1000) if rank_samples > 0 else 0,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    total_time_seconds=total_time,
                )
            else:
                return original_benchmark_generation(num_samples, config)

        # Monkey patch for this test
        runner.benchmark_generation = extended_benchmark_generation

        # Run comparison
        report = runner.compare_configurations(configs)

        # Verify comparison worked
        assert "distributed_rank0" in report.metrics_comparison

        distributed_metrics = report.metrics_comparison["distributed_rank0"]
        assert "throughput_change_percent" in distributed_metrics
        assert "absolute_metrics" in distributed_metrics

        # Distributed version should have reasonable performance
        abs_metrics = distributed_metrics["absolute_metrics"]
        assert abs_metrics["throughput_samples_per_second"] > 0

    def test_benchmark_integration_with_batch_processing(self):
        """Test benchmarking integration with batch processing."""
        config = BenchmarkConfig(iterations=3, warmup_iterations=1)
        runner = BenchmarkRunner(config)

        # Test different batch configurations
        configs = [
            {
                "name": "sequential_batching",
                "type": "batch_processing",
                "num_samples": 200,
                "batch_config": BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=25),
            },
            {
                "name": "memory_optimized_batching",
                "type": "batch_processing",
                "num_samples": 200,
                "batch_config": BatchConfig(
                    strategy=BatchStrategy.MEMORY_OPTIMIZED, batch_size=30, max_memory_mb=100
                ),
            },
        ]

        # Run comparison
        report = runner.compare_configurations(configs)

        # Verify both configurations were benchmarked
        assert "memory_optimized_batching" in report.metrics_comparison

        memory_opt_metrics = report.metrics_comparison["memory_optimized_batching"]
        assert "absolute_metrics" in memory_opt_metrics

        # Both should have reasonable performance
        abs_metrics = memory_opt_metrics["absolute_metrics"]
        assert abs_metrics["throughput_samples_per_second"] > 0
        assert abs_metrics["total_time_seconds"] > 0

    def test_performance_measurement_utilities_integration(self):
        """Test integration of performance measurement utilities."""
        # Test measure_time with actual DPA function
        with measure_time() as timer:
            config = AugmentationConfig()
            for i in range(50):
                gen_augmentation_params(i, config)

        assert timer["elapsed_seconds"] > 0
        assert timer["cpu_seconds"] >= 0
        assert timer["end_time"] > timer["start_time"]

        # Test measure_memory with actual operations
        with measure_memory() as memory:
            # Perform memory-intensive operation
            config = AugmentationConfig()
            stream = stream_augmentation_chain(500, config, chunk_size=100)
            chunks = list(stream)  # Force evaluation

            # Verify streaming worked
            total_items = sum(len(chunk) for chunk in chunks)
            assert total_items == 500

        assert "delta_mb" in memory
        assert "initial_rss_mb" in memory
        assert "final_rss_mb" in memory
        assert isinstance(memory["delta_mb"], float)

        # Test benchmark_function with DPA function
        def test_dpa_function():
            config = AugmentationConfig(rotation_range=(-15, 15))
            return gen_augmentation_params(42, config)

        result = benchmark_function(test_dpa_function, iterations=10)

        assert result.function_name == "test_dpa_function"
        assert result.iterations == 10
        assert result.avg_time_seconds > 0
        assert result.throughput_ops_per_second > 0
        assert result.min_time_seconds <= result.avg_time_seconds <= result.max_time_seconds

    def test_benchmark_error_handling_integration(self):
        """Test error handling in benchmark integration scenarios."""
        config = BenchmarkConfig(iterations=2, warmup_iterations=1)
        runner = BenchmarkRunner(config)

        # Test with invalid configuration
        invalid_configs = [{"name": "invalid", "type": "unknown_type", "num_samples": 100}]

        with pytest.raises(MeasurementError, match="Unknown configuration type"):
            runner.compare_configurations(invalid_configs)

        # Test with empty configuration list
        with pytest.raises(ValueError, match="At least one configuration must be provided"):
            runner.compare_configurations([])

        # Test profiler error handling
        profiler = PerformanceProfiler()

        # Test duplicate operation error
        profiler.start_profiling("test_op")
        with pytest.raises(ProfilingError):
            profiler.start_profiling("test_op")

        # Clean up
        profiler.end_profiling("test_op")

        # Test ending non-existent operation
        with pytest.raises(ProfilingError):
            profiler.end_profiling("nonexistent_op")

    def test_benchmark_results_history_management(self):
        """Test benchmark results history management."""
        config = BenchmarkConfig(iterations=2, warmup_iterations=1)
        runner = BenchmarkRunner(config)

        # Initially empty
        assert len(runner.get_results_history()) == 0

        # Run some benchmarks
        configs = [
            {"name": "test1", "type": "generation", "num_samples": 50},
            {"name": "test2", "type": "streaming", "num_samples": 50, "chunk_size": 25},
        ]

        runner.compare_configurations(configs)

        # History should be populated
        history = runner.get_results_history()
        assert len(history) == 2

        # Verify history entries
        for entry in history:
            assert "config_name" in entry
            assert "config" in entry
            assert "metrics" in entry
            assert "timestamp" in entry
            assert isinstance(entry["metrics"], PerformanceMetrics)
            assert entry["timestamp"] > 0

        # Clear history
        runner.clear_results_history()
        assert len(runner.get_results_history()) == 0

    def test_comprehensive_benchmarking_workflow(self):
        """Test comprehensive benchmarking workflow with all components."""
        # Create profiler for function-level profiling
        profiler = PerformanceProfiler()

        # Profile individual DPA functions
        @profiler.profile_function
        def profiled_generation():
            config = AugmentationConfig(augmentation_depth=8)
            return gen_augmentation_params(123, config)

        # Call profiled function
        for _ in range(5):
            result = profiled_generation()
            assert "hash" in result

        # Create benchmark runner for system-level benchmarking
        benchmark_config = BenchmarkConfig(
            iterations=3, warmup_iterations=1, measure_memory=True, measure_cpu=True
        )
        runner = BenchmarkRunner(benchmark_config)

        # Define comprehensive test configurations
        configs = [
            {
                "name": "baseline_generation",
                "type": "generation",
                "num_samples": 100,
                "aug_config": AugmentationConfig(augmentation_depth=5),
            },
            {
                "name": "optimized_generation",
                "type": "generation",
                "num_samples": 100,
                "aug_config": AugmentationConfig(augmentation_depth=3),
            },
            {
                "name": "streaming_small_chunks",
                "type": "streaming",
                "num_samples": 100,
                "chunk_size": 20,
            },
            {
                "name": "streaming_large_chunks",
                "type": "streaming",
                "num_samples": 100,
                "chunk_size": 50,
            },
            {"name": "batch_processing", "type": "batch_processing", "num_samples": 100},
        ]

        # Run comprehensive comparison
        report = runner.compare_configurations(configs)

        # Verify comprehensive results
        assert len(report.comparison_configs) == 4
        assert len(report.metrics_comparison) == 4
        assert len(report.performance_improvements) == 4

        # Generate reports in all formats
        text_report = runner.generate_performance_report(report, "text")
        json_report = runner.generate_performance_report(report, "json")
        markdown_report = runner.generate_performance_report(report, "markdown")

        # Verify all reports contain key information
        assert "baseline_generation" in text_report
        assert "optimized_generation" in text_report

        json_data = json.loads(json_report)
        assert json_data["baseline_config"] == "baseline_generation"
        assert len(json_data["metrics_comparison"]) == 4

        assert "# Performance Benchmark Report" in markdown_report
        assert "**Baseline Configuration:**" in markdown_report

        # Get profiling summary
        profiling_summary = profiler.get_profile_summary()

        # Verify profiling data
        assert len(profiling_summary) > 0

        # Get benchmark history
        history = runner.get_results_history()
        assert len(history) == 5  # All configurations

        # Verify recommendations were generated
        assert isinstance(report.recommendations, list)
        assert len(report.recommendations) > 0

        # Verify performance improvements were calculated
        for _config_name, improvement in report.performance_improvements.items():
            assert isinstance(improvement, float)
            # Improvement can be positive or negative
            assert -1000 <= improvement <= 1000  # Reasonable range
