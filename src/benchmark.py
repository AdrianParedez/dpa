"""
Benchmarking module for DPA library.

This module provides comprehensive performance measurement, profiling, and reporting
tools for optimizing augmentation pipeline performance. It includes timing measurement,
memory usage tracking, and statistical analysis capabilities.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import psutil

# Type variables for generic function decoration
P = ParamSpec("P")
T = TypeVar("T")


# Exception Classes
class BenchmarkError(Exception):
    """Base exception for benchmarking operations."""

    pass


class ProfilingError(BenchmarkError):
    """Raised when profiling operations fail."""

    pass


class InvalidBenchmarkConfigError(BenchmarkError):
    """Raised when benchmark configuration is invalid."""

    pass


class MeasurementError(BenchmarkError):
    """Raised when performance measurement fails."""

    pass


# Data Models
@dataclass
class PerformanceMetrics:
    """Performance metrics collected during benchmarking operations."""

    throughput_samples_per_second: float
    avg_latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    total_time_seconds: float

    def __post_init__(self):
        """Validate performance metrics after initialization."""
        if self.throughput_samples_per_second < 0:
            raise ValueError("throughput_samples_per_second must be non-negative")
        if self.avg_latency_ms < 0:
            raise ValueError("avg_latency_ms must be non-negative")
        if self.memory_usage_mb < 0:
            raise ValueError("memory_usage_mb must be non-negative")
        if self.cpu_usage_percent < 0 or self.cpu_usage_percent > 100:
            raise ValueError("cpu_usage_percent must be between 0 and 100")
        if self.total_time_seconds < 0:
            raise ValueError("total_time_seconds must be non-negative")


@dataclass
class ProfileResult:
    """Result of profiling a specific operation."""

    operation_name: str
    total_time_seconds: float
    call_count: int
    avg_time_per_call_ms: float
    memory_delta_mb: float

    def __post_init__(self):
        """Validate profile result after initialization."""
        if self.total_time_seconds < 0:
            raise ValueError("total_time_seconds must be non-negative")
        if self.call_count < 0:
            raise ValueError("call_count must be non-negative")
        if self.avg_time_per_call_ms < 0:
            raise ValueError("avg_time_per_call_ms must be non-negative")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark operations."""

    iterations: int = 100
    warmup_iterations: int = 10
    measure_memory: bool = True
    measure_cpu: bool = True
    output_format: str = "json"

    def __post_init__(self):
        """Validate benchmark configuration after initialization."""
        if self.iterations <= 0:
            raise InvalidBenchmarkConfigError("iterations must be positive")
        if self.warmup_iterations < 0:
            raise InvalidBenchmarkConfigError("warmup_iterations must be non-negative")
        if self.output_format not in ["json", "csv", "text"]:
            raise InvalidBenchmarkConfigError("output_format must be 'json', 'csv', or 'text'")


@dataclass
class BenchmarkResult:
    """Result of benchmarking a function or operation."""

    function_name: str
    iterations: int
    total_time_seconds: float
    avg_time_seconds: float
    min_time_seconds: float
    max_time_seconds: float
    std_dev_seconds: float
    throughput_ops_per_second: float
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    @property
    def avg_time_ms(self) -> float:
        """Average time per operation in milliseconds."""
        return self.avg_time_seconds * 1000

    @property
    def min_time_ms(self) -> float:
        """Minimum time per operation in milliseconds."""
        return self.min_time_seconds * 1000

    @property
    def max_time_ms(self) -> float:
        """Maximum time per operation in milliseconds."""
        return self.max_time_seconds * 1000


@dataclass
class ComparisonReport:
    """Report comparing performance across different configurations."""

    baseline_config: str
    comparison_configs: list[str]
    metrics_comparison: dict[str, dict[str, float]]
    performance_improvements: dict[str, float]
    recommendations: list[str] = field(default_factory=list)


# Performance Measurement Utilities
@contextmanager
def measure_time() -> Generator[dict[str, float], None, None]:
    """Context manager for measuring execution time.

    Yields:
        Dictionary that will be populated with timing information

    Example:
        >>> with measure_time() as timer:
        ...     # Some operation
        ...     time.sleep(0.1)
        >>> print(f"Operation took {timer['elapsed_seconds']:.3f} seconds")
    """
    result = {}
    start_time = time.perf_counter()
    start_process_time = time.process_time()

    try:
        yield result
    finally:
        end_time = time.perf_counter()
        end_process_time = time.process_time()

        result["elapsed_seconds"] = end_time - start_time
        result["cpu_seconds"] = end_process_time - start_process_time
        result["start_time"] = start_time
        result["end_time"] = end_time


@contextmanager
def measure_memory() -> Generator[dict[str, int], None, None]:
    """Context manager for measuring memory usage during operations.

    Yields:
        Dictionary that will be populated with memory information

    Example:
        >>> with measure_memory() as memory:
        ...     # Some memory-intensive operation
        ...     data = [i for i in range(1000000)]
        >>> print(f"Memory delta: {memory['delta_mb']} MB")
    """
    result = {}

    try:
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info()
        system_memory = psutil.virtual_memory()

        result["initial_rss_mb"] = initial_memory.rss / (1024 * 1024)
        result["initial_vms_mb"] = initial_memory.vms / (1024 * 1024)
        result["initial_system_available_mb"] = system_memory.available / (1024 * 1024)
        result["initial_system_used_mb"] = system_memory.used / (1024 * 1024)

        yield result

    finally:
        # Get final memory usage
        final_memory = process.memory_info()
        final_system_memory = psutil.virtual_memory()

        result["final_rss_mb"] = final_memory.rss / (1024 * 1024)
        result["final_vms_mb"] = final_memory.vms / (1024 * 1024)
        result["final_system_available_mb"] = final_system_memory.available / (1024 * 1024)
        result["final_system_used_mb"] = final_system_memory.used / (1024 * 1024)

        # Calculate deltas
        result["delta_rss_mb"] = result["final_rss_mb"] - result["initial_rss_mb"]
        result["delta_vms_mb"] = result["final_vms_mb"] - result["initial_vms_mb"]
        result["delta_system_mb"] = (
            result["final_system_used_mb"] - result["initial_system_used_mb"]
        )
        result["delta_mb"] = result["delta_rss_mb"]  # Primary delta metric


def benchmark_function[**P, T](func: Callable[P, T], iterations: int = 100) -> BenchmarkResult:
    """Benchmark a function with statistical analysis.

    Args:
        func: Function to benchmark (should be callable with no arguments)
        iterations: Number of iterations to run

    Returns:
        Benchmark result with statistical analysis

    Raises:
        MeasurementError: If benchmarking fails

    Example:
        >>> def test_function():
        ...     return sum(range(1000))
        >>> result = benchmark_function(test_function, iterations=50)
        >>> print(f"Average time: {result.avg_time_ms:.2f}ms")
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    try:
        times = []
        total_memory_delta = 0.0
        cpu_usage_samples = []

        # Warmup run
        try:
            func()
        except Exception as e:
            logging.debug(f"Warmup run failed (expected): {e}")

        # Benchmark runs
        for i in range(iterations):
            # Measure CPU usage before operation
            cpu_before = psutil.cpu_percent(interval=None)

            with measure_time() as timer, measure_memory() as memory:
                try:
                    func()
                except Exception as e:
                    raise MeasurementError(
                        f"Function failed during benchmarking iteration {i}: {e}"
                    ) from e

            # Measure CPU usage after operation
            cpu_after = psutil.cpu_percent(interval=None)
            cpu_usage_samples.append((cpu_before + cpu_after) / 2)

            times.append(timer["elapsed_seconds"])
            total_memory_delta += memory["delta_mb"]

        # Calculate statistics
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        throughput = iterations / total_time if total_time > 0 else 0.0
        avg_memory_delta = total_memory_delta / iterations
        avg_cpu_usage = statistics.mean(cpu_usage_samples) if cpu_usage_samples else 0.0

        return BenchmarkResult(
            function_name=getattr(func, "__name__", str(func)),
            iterations=iterations,
            total_time_seconds=total_time,
            avg_time_seconds=avg_time,
            min_time_seconds=min_time,
            max_time_seconds=max_time,
            std_dev_seconds=std_dev,
            throughput_ops_per_second=throughput,
            memory_usage_mb=avg_memory_delta,
            cpu_usage_percent=avg_cpu_usage,
        )

    except Exception as e:
        if isinstance(e, MeasurementError):
            raise
        raise MeasurementError(f"Failed to benchmark function: {e}") from e


# Performance Profiler Class
class PerformanceProfiler:
    """Performance profiler for tracking operation timing and memory usage."""

    def __init__(self, enable_memory_tracking: bool = True):
        """Initialize the performance profiler.

        Args:
            enable_memory_tracking: Whether to track memory usage during profiling
        """
        self.enable_memory_tracking = enable_memory_tracking
        self._active_operations: dict[str, dict[str, Any]] = {}
        self._completed_operations: dict[str, list[ProfileResult]] = defaultdict(list)
        self._operation_counts: dict[str, int] = defaultdict(int)

    def start_profiling(self, operation_name: str) -> None:
        """Start profiling an operation.

        Args:
            operation_name: Name of the operation being profiled

        Raises:
            ProfilingError: If operation is already being profiled
        """
        if operation_name in self._active_operations:
            raise ProfilingError(f"Operation '{operation_name}' is already being profiled")

        try:
            start_data = {
                "start_time": time.perf_counter(),
                "start_process_time": time.process_time(),
            }

            if self.enable_memory_tracking:
                try:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    start_data["start_memory_mb"] = memory_info.rss / (1024 * 1024)
                except (ImportError, AttributeError, OSError, Exception) as e:
                    # If memory tracking fails, continue without it
                    logging.debug(f"Memory tracking unavailable: {e}")

            self._active_operations[operation_name] = start_data

        except Exception as e:
            raise ProfilingError(
                f"Failed to start profiling operation '{operation_name}': {e}"
            ) from e

    def end_profiling(self, operation_name: str) -> ProfileResult:
        """End profiling an operation and return the result.

        Args:
            operation_name: Name of the operation being profiled

        Returns:
            Profile result for the operation

        Raises:
            ProfilingError: If operation is not being profiled
        """
        if operation_name not in self._active_operations:
            raise ProfilingError(f"Operation '{operation_name}' is not being profiled")

        try:
            start_data = self._active_operations.pop(operation_name)
            end_time = time.perf_counter()

            # Calculate timing
            total_time = end_time - start_data["start_time"]
            self._operation_counts[operation_name] += 1
            call_count = self._operation_counts[operation_name]
            avg_time_ms = total_time * 1000  # Convert to milliseconds

            # Calculate memory delta
            memory_delta_mb = 0.0
            if self.enable_memory_tracking and "start_memory_mb" in start_data:
                try:
                    process = psutil.Process()
                    end_memory_mb = process.memory_info().rss / (1024 * 1024)
                    memory_delta_mb = end_memory_mb - start_data["start_memory_mb"]
                except Exception:
                    memory_delta_mb = 0.0  # Fallback if memory measurement fails

            # Create profile result
            result = ProfileResult(
                operation_name=operation_name,
                total_time_seconds=total_time,
                call_count=call_count,
                avg_time_per_call_ms=avg_time_ms,
                memory_delta_mb=memory_delta_mb,
            )

            # Store completed operation
            self._completed_operations[operation_name].append(result)

            return result

        except Exception as e:
            # Clean up active operation on error
            self._active_operations.pop(operation_name, None)
            raise ProfilingError(
                f"Failed to end profiling operation '{operation_name}': {e}"
            ) from e

    def profile_function(self, func: Callable[P, T]) -> Callable[P, T]:
        """Decorator for profiling function calls.

        Args:
            func: Function to profile

        Returns:
            Decorated function that profiles each call

        Example:
            >>> profiler = PerformanceProfiler()
            >>> @profiler.profile_function
            ... def my_function(x, y):
            ...     return x + y
            >>> result = my_function(1, 2)
            >>> summary = profiler.get_profile_summary()
        """

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            operation_name = (
                f"{func.__module__}.{func.__name__}"
                if hasattr(func, "__module__")
                else func.__name__
            )

            self.start_profiling(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.end_profiling(operation_name)

        return wrapper

    def get_profile_summary(self) -> dict[str, ProfileResult]:
        """Get summary of all completed profiling operations.

        Returns:
            Dictionary mapping operation names to their latest profile results
        """
        summary = {}
        for operation_name, results in self._completed_operations.items():
            if results:
                # Get the most recent result, but aggregate statistics
                results[-1]

                # Calculate aggregate statistics across all calls
                total_time = sum(r.total_time_seconds for r in results)
                total_calls = sum(r.call_count for r in results)
                avg_time_per_call = (total_time / total_calls * 1000) if total_calls > 0 else 0.0
                avg_memory_delta = statistics.mean(r.memory_delta_mb for r in results)

                # Create aggregated result
                summary[operation_name] = ProfileResult(
                    operation_name=operation_name,
                    total_time_seconds=total_time,
                    call_count=total_calls,
                    avg_time_per_call_ms=avg_time_per_call,
                    memory_delta_mb=avg_memory_delta,
                )

        return summary

    def reset(self) -> None:
        """Reset all profiling data."""
        self._active_operations.clear()
        self._completed_operations.clear()
        self._operation_counts.clear()

    @contextmanager
    def profile_context(self, operation_name: str) -> Generator[None, None, None]:
        """Context manager for profiling a block of code.

        Args:
            operation_name: Name of the operation being profiled

        Example:
            >>> profiler = PerformanceProfiler()
            >>> with profiler.profile_context("data_processing"):
            ...     # Some data processing code
            ...     process_data()
        """
        self.start_profiling(operation_name)
        try:
            yield
        finally:
            self.end_profiling(operation_name)


class BenchmarkRunner:
    """Main benchmark runner for comprehensive performance testing."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.profiler = PerformanceProfiler(enable_memory_tracking=config.measure_memory)
        self._results_history: list[dict[str, Any]] = []

    def benchmark_generation(self, num_samples: int, config: Any = None) -> PerformanceMetrics:
        """Benchmark parameter generation operations.

        Args:
            num_samples: Number of samples to generate
            config: Augmentation configuration (optional)

        Returns:
            Performance metrics for generation operations

        Raises:
            MeasurementError: If benchmarking fails
        """
        try:
            # Import DPA functions dynamically to avoid circular imports
            from .dpa import AugmentationConfig, gen_augmentation_params

            # Use default config if none provided
            if config is None:
                config = AugmentationConfig()

            # Warmup runs
            for _ in range(self.config.warmup_iterations):
                try:
                    gen_augmentation_params(0, config)
                except Exception as e:
                    logging.debug(f"Generation warmup failed (expected): {e}")

            # Benchmark runs
            times = []
            memory_deltas = []
            cpu_samples = []

            for _i in range(self.config.iterations):
                # Measure CPU before operation
                cpu_before = psutil.cpu_percent(interval=None) if self.config.measure_cpu else 0.0

                with measure_time() as timer:
                    if self.config.measure_memory:
                        with measure_memory() as memory:
                            for sample_id in range(num_samples):
                                gen_augmentation_params(sample_id, config)
                        memory_deltas.append(memory["delta_mb"])
                    else:
                        for sample_id in range(num_samples):
                            gen_augmentation_params(sample_id, config)
                        memory_deltas.append(0.0)

                # Measure CPU after operation
                if self.config.measure_cpu:
                    cpu_after = psutil.cpu_percent(interval=None)
                    cpu_samples.append((cpu_before + cpu_after) / 2)
                else:
                    cpu_samples.append(0.0)

                times.append(timer["elapsed_seconds"])

            # Calculate metrics
            total_time = sum(times)
            avg_time = statistics.mean(times)
            throughput = (
                (num_samples * self.config.iterations) / total_time if total_time > 0 else 0.0
            )
            avg_latency_ms = (avg_time / num_samples * 1000) if num_samples > 0 else 0.0
            avg_memory_mb = statistics.mean(memory_deltas) if memory_deltas else 0.0
            avg_cpu_percent = statistics.mean(cpu_samples) if cpu_samples else 0.0

            return PerformanceMetrics(
                throughput_samples_per_second=throughput,
                avg_latency_ms=avg_latency_ms,
                memory_usage_mb=avg_memory_mb,
                cpu_usage_percent=avg_cpu_percent,
                total_time_seconds=total_time,
            )

        except Exception as e:
            raise MeasurementError(f"Failed to benchmark parameter generation: {e}") from e

    def benchmark_streaming(self, num_samples: int, chunk_size: int = 1000) -> PerformanceMetrics:
        """Benchmark streaming operations.

        Args:
            num_samples: Number of samples to stream
            chunk_size: Size of chunks for streaming

        Returns:
            Performance metrics for streaming operations

        Raises:
            MeasurementError: If benchmarking fails
        """
        try:
            # Import streaming functions dynamically
            from .dpa import AugmentationConfig, stream_augmentation_chain

            config = AugmentationConfig()

            # Warmup runs
            for _ in range(self.config.warmup_iterations):
                try:
                    # Consume a small stream for warmup
                    warmup_stream = stream_augmentation_chain(
                        min(100, num_samples), config, chunk_size=chunk_size
                    )
                    list(warmup_stream)  # Consume the generator
                except Exception as e:
                    logging.debug(f"Streaming warmup failed (expected): {e}")

            # Benchmark runs
            times = []
            memory_deltas = []
            cpu_samples = []

            for _i in range(self.config.iterations):
                # Measure CPU before operation
                cpu_before = psutil.cpu_percent(interval=None) if self.config.measure_cpu else 0.0

                with measure_time() as timer:
                    if self.config.measure_memory:
                        with measure_memory() as memory:
                            stream = stream_augmentation_chain(
                                num_samples, config, chunk_size=chunk_size
                            )
                            # Consume the entire stream
                            sum(1 for _ in stream)
                        memory_deltas.append(memory["delta_mb"])
                    else:
                        stream = stream_augmentation_chain(
                            num_samples, config, chunk_size=chunk_size
                        )
                        sum(1 for _ in stream)
                        memory_deltas.append(0.0)

                # Measure CPU after operation
                if self.config.measure_cpu:
                    cpu_after = psutil.cpu_percent(interval=None)
                    cpu_samples.append((cpu_before + cpu_after) / 2)
                else:
                    cpu_samples.append(0.0)

                times.append(timer["elapsed_seconds"])

            # Calculate metrics
            total_time = sum(times)
            avg_time = statistics.mean(times)
            throughput = (
                (num_samples * self.config.iterations) / total_time if total_time > 0 else 0.0
            )
            avg_latency_ms = (avg_time / num_samples * 1000) if num_samples > 0 else 0.0
            avg_memory_mb = statistics.mean(memory_deltas) if memory_deltas else 0.0
            avg_cpu_percent = statistics.mean(cpu_samples) if cpu_samples else 0.0

            return PerformanceMetrics(
                throughput_samples_per_second=throughput,
                avg_latency_ms=avg_latency_ms,
                memory_usage_mb=avg_memory_mb,
                cpu_usage_percent=avg_cpu_percent,
                total_time_seconds=total_time,
            )

        except Exception as e:
            raise MeasurementError(f"Failed to benchmark streaming operations: {e}") from e

    def benchmark_batch_processing(
        self, num_samples: int, batch_config: Any = None
    ) -> PerformanceMetrics:
        """Benchmark batch processing operations.

        Args:
            num_samples: Number of samples to process
            batch_config: Batch processing configuration

        Returns:
            Performance metrics for batch processing operations

        Raises:
            MeasurementError: If benchmarking fails
        """
        try:
            # Import batch processing components dynamically
            from .batch import BatchConfig, BatchProcessor, BatchStrategy
            from .dpa import AugmentationConfig, stream_augmentation_chain

            # Use default batch config if none provided
            if batch_config is None:
                batch_config = BatchConfig(
                    strategy=BatchStrategy.SEQUENTIAL, batch_size=100, max_memory_mb=500
                )

            aug_config = AugmentationConfig()

            # Warmup runs
            for _ in range(self.config.warmup_iterations):
                try:
                    processor = BatchProcessor(batch_config.strategy, batch_config)
                    param_stream = stream_augmentation_chain(min(100, num_samples), aug_config)
                    batch_stream = processor.process_stream(param_stream)
                    list(batch_stream)  # Consume the generator
                except Exception as e:
                    logging.debug(f"Batch processing warmup failed (expected): {e}")

            # Benchmark runs
            times = []
            memory_deltas = []
            cpu_samples = []

            for _i in range(self.config.iterations):
                # Measure CPU before operation
                cpu_before = psutil.cpu_percent(interval=None) if self.config.measure_cpu else 0.0

                with measure_time() as timer:
                    if self.config.measure_memory:
                        with measure_memory() as memory:
                            processor = BatchProcessor(batch_config.strategy, batch_config)
                            param_stream = stream_augmentation_chain(num_samples, aug_config)
                            batch_stream = processor.process_stream(param_stream)
                            # Consume all batches
                            sum(1 for batch in batch_stream)
                        memory_deltas.append(memory["delta_mb"])
                    else:
                        processor = BatchProcessor(batch_config.strategy, batch_config)
                        param_stream = stream_augmentation_chain(num_samples, aug_config)
                        batch_stream = processor.process_stream(param_stream)
                        sum(1 for batch in batch_stream)
                        memory_deltas.append(0.0)

                # Measure CPU after operation
                if self.config.measure_cpu:
                    cpu_after = psutil.cpu_percent(interval=None)
                    cpu_samples.append((cpu_before + cpu_after) / 2)
                else:
                    cpu_samples.append(0.0)

                times.append(timer["elapsed_seconds"])

            # Calculate metrics
            total_time = sum(times)
            avg_time = statistics.mean(times)
            throughput = (
                (num_samples * self.config.iterations) / total_time if total_time > 0 else 0.0
            )
            avg_latency_ms = (avg_time / num_samples * 1000) if num_samples > 0 else 0.0
            avg_memory_mb = statistics.mean(memory_deltas) if memory_deltas else 0.0
            avg_cpu_percent = statistics.mean(cpu_samples) if cpu_samples else 0.0

            return PerformanceMetrics(
                throughput_samples_per_second=throughput,
                avg_latency_ms=avg_latency_ms,
                memory_usage_mb=avg_memory_mb,
                cpu_usage_percent=avg_cpu_percent,
                total_time_seconds=total_time,
            )

        except Exception as e:
            raise MeasurementError(f"Failed to benchmark batch processing: {e}") from e

    def compare_configurations(self, configs: list[dict]) -> ComparisonReport:
        """Compare performance across different configurations.

        Args:
            configs: List of configuration dictionaries to compare.
                    Each dict should have 'name', 'type', and config-specific parameters.
                    Example: [
                        {'name': 'baseline', 'type': 'generation', 'num_samples': 1000},
                        {'name': 'optimized', 'type': 'streaming', 'num_samples': 1000, 'chunk_size': 500}
                    ]

        Returns:
            Comparison report with performance analysis

        Raises:
            MeasurementError: If comparison fails
        """
        if not configs:
            raise ValueError("At least one configuration must be provided")

        try:
            results = {}
            baseline_config = configs[0]["name"]

            # Run benchmarks for each configuration
            for config in configs:
                config_name = config["name"]
                config_type = config.get("type", "generation")

                if config_type == "generation":
                    num_samples = config.get("num_samples", 1000)
                    aug_config = config.get("aug_config", None)
                    metrics = self.benchmark_generation(num_samples, aug_config)

                elif config_type == "streaming":
                    num_samples = config.get("num_samples", 1000)
                    chunk_size = config.get("chunk_size", 1000)
                    metrics = self.benchmark_streaming(num_samples, chunk_size)

                elif config_type == "batch_processing":
                    num_samples = config.get("num_samples", 1000)
                    batch_config = config.get("batch_config", None)
                    metrics = self.benchmark_batch_processing(num_samples, batch_config)

                else:
                    raise ValueError(f"Unknown configuration type: {config_type}")

                results[config_name] = metrics

                # Store in history
                self._results_history.append(
                    {
                        "config_name": config_name,
                        "config": config,
                        "metrics": metrics,
                        "timestamp": time.time(),
                    }
                )

            # Calculate performance comparisons
            baseline_metrics = results[baseline_config]
            metrics_comparison = {}
            performance_improvements = {}

            for config_name, metrics in results.items():
                if config_name == baseline_config:
                    continue

                # Calculate relative performance changes
                throughput_change = (
                    (
                        (
                            metrics.throughput_samples_per_second
                            - baseline_metrics.throughput_samples_per_second
                        )
                        / baseline_metrics.throughput_samples_per_second
                        * 100
                    )
                    if baseline_metrics.throughput_samples_per_second > 0
                    else 0.0
                )

                latency_change = (
                    (
                        (metrics.avg_latency_ms - baseline_metrics.avg_latency_ms)
                        / baseline_metrics.avg_latency_ms
                        * 100
                    )
                    if baseline_metrics.avg_latency_ms > 0
                    else 0.0
                )

                memory_change = (
                    (
                        (metrics.memory_usage_mb - baseline_metrics.memory_usage_mb)
                        / baseline_metrics.memory_usage_mb
                        * 100
                    )
                    if baseline_metrics.memory_usage_mb > 0
                    else 0.0
                )

                cpu_change = (
                    (
                        (metrics.cpu_usage_percent - baseline_metrics.cpu_usage_percent)
                        / baseline_metrics.cpu_usage_percent
                        * 100
                    )
                    if baseline_metrics.cpu_usage_percent > 0
                    else 0.0
                )

                metrics_comparison[config_name] = {
                    "throughput_change_percent": throughput_change,
                    "latency_change_percent": latency_change,
                    "memory_change_percent": memory_change,
                    "cpu_change_percent": cpu_change,
                    "absolute_metrics": {
                        "throughput_samples_per_second": metrics.throughput_samples_per_second,
                        "avg_latency_ms": metrics.avg_latency_ms,
                        "memory_usage_mb": metrics.memory_usage_mb,
                        "cpu_usage_percent": metrics.cpu_usage_percent,
                        "total_time_seconds": metrics.total_time_seconds,
                    },
                }

                # Calculate overall performance improvement score
                # Positive throughput change is good, negative latency/memory/cpu changes are good
                improvement_score = (
                    throughput_change - latency_change - memory_change - cpu_change
                ) / 4
                performance_improvements[config_name] = improvement_score

            # Generate recommendations
            recommendations = self._generate_recommendations(
                results, metrics_comparison, performance_improvements
            )

            return ComparisonReport(
                baseline_config=baseline_config,
                comparison_configs=[config["name"] for config in configs[1:]],
                metrics_comparison=metrics_comparison,
                performance_improvements=performance_improvements,
                recommendations=recommendations,
            )

        except Exception as e:
            raise MeasurementError(f"Failed to compare configurations: {e}") from e

    def _generate_recommendations(
        self,
        results: dict[str, PerformanceMetrics],
        metrics_comparison: dict[str, dict],
        performance_improvements: dict[str, float],
    ) -> list[str]:
        """Generate performance optimization recommendations based on benchmark results.

        Args:
            results: Dictionary of configuration names to performance metrics
            metrics_comparison: Detailed metrics comparison data
            performance_improvements: Overall improvement scores

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Find best performing configuration
        if performance_improvements:
            best_config = max(
                performance_improvements.keys(), key=lambda k: performance_improvements[k]
            )
            best_score = performance_improvements[best_config]

            if best_score > 10:  # Significant improvement
                recommendations.append(
                    f"Configuration '{best_config}' shows significant performance improvement ({best_score:.1f}% overall)"
                )
            elif best_score > 0:
                recommendations.append(
                    f"Configuration '{best_config}' shows modest performance improvement ({best_score:.1f}% overall)"
                )

        # Analyze specific metrics
        for config_name, comparison in metrics_comparison.items():
            throughput_change = comparison["throughput_change_percent"]
            latency_change = comparison["latency_change_percent"]
            memory_change = comparison["memory_change_percent"]
            cpu_change = comparison["cpu_change_percent"]

            # Throughput recommendations
            if throughput_change > 20:
                recommendations.append(
                    f"Configuration '{config_name}' significantly improves throughput by {throughput_change:.1f}%"
                )
            elif throughput_change < -20:
                recommendations.append(
                    f"Configuration '{config_name}' significantly reduces throughput by {abs(throughput_change):.1f}% - consider alternatives"
                )

            # Latency recommendations
            if latency_change < -20:
                recommendations.append(
                    f"Configuration '{config_name}' significantly reduces latency by {abs(latency_change):.1f}%"
                )
            elif latency_change > 20:
                recommendations.append(
                    f"Configuration '{config_name}' increases latency by {latency_change:.1f}% - may impact responsiveness"
                )

            # Memory recommendations
            if memory_change < -20:
                recommendations.append(
                    f"Configuration '{config_name}' significantly reduces memory usage by {abs(memory_change):.1f}%"
                )
            elif memory_change > 50:
                recommendations.append(
                    f"Configuration '{config_name}' increases memory usage by {memory_change:.1f}% - monitor for memory constraints"
                )

            # CPU recommendations
            if cpu_change < -20:
                recommendations.append(
                    f"Configuration '{config_name}' significantly reduces CPU usage by {abs(cpu_change):.1f}%"
                )
            elif cpu_change > 50:
                recommendations.append(
                    f"Configuration '{config_name}' increases CPU usage by {cpu_change:.1f}% - may impact system performance"
                )

        # General recommendations based on patterns
        baseline_metrics = list(results.values())[0]  # First result is baseline

        if baseline_metrics.memory_usage_mb > 1000:
            recommendations.append(
                "High memory usage detected - consider using batch processing with memory limits"
            )

        if baseline_metrics.avg_latency_ms > 100:
            recommendations.append(
                "High latency detected - consider optimizing augmentation parameters or using streaming"
            )

        if baseline_metrics.throughput_samples_per_second < 100:
            recommendations.append(
                "Low throughput detected - consider batch processing or parameter optimization"
            )

        # Remove duplicates and limit recommendations
        recommendations = list(
            dict.fromkeys(recommendations)
        )  # Remove duplicates while preserving order
        return recommendations[:10]  # Limit to top 10 recommendations

    def generate_performance_report(
        self, comparison_report: ComparisonReport, format_type: str = "text"
    ) -> str:
        """Generate a comprehensive performance report.

        Args:
            comparison_report: Comparison report to format
            format_type: Output format ("text", "json", or "markdown")

        Returns:
            Formatted performance report
        """
        if format_type == "json":
            import json

            report_data = {
                "baseline_config": comparison_report.baseline_config,
                "comparison_configs": comparison_report.comparison_configs,
                "metrics_comparison": comparison_report.metrics_comparison,
                "performance_improvements": comparison_report.performance_improvements,
                "recommendations": comparison_report.recommendations,
                "system_info": get_system_info(),
                "benchmark_config": {
                    "iterations": self.config.iterations,
                    "warmup_iterations": self.config.warmup_iterations,
                    "measure_memory": self.config.measure_memory,
                    "measure_cpu": self.config.measure_cpu,
                },
            }
            return json.dumps(report_data, indent=2)

        elif format_type == "markdown":
            lines = [
                "# Performance Benchmark Report",
                "",
                f"**Baseline Configuration:** {comparison_report.baseline_config}",
                f"**Comparison Configurations:** {', '.join(comparison_report.comparison_configs)}",
                "",
                "## Performance Comparison",
                "",
            ]

            for config_name, metrics in comparison_report.metrics_comparison.items():
                lines.extend(
                    [
                        f"### {config_name}",
                        "",
                        f"- **Throughput Change:** {metrics['throughput_change_percent']:+.1f}%",
                        f"- **Latency Change:** {metrics['latency_change_percent']:+.1f}%",
                        f"- **Memory Change:** {metrics['memory_change_percent']:+.1f}%",
                        f"- **CPU Change:** {metrics['cpu_change_percent']:+.1f}%",
                        f"- **Overall Improvement:** {comparison_report.performance_improvements[config_name]:+.1f}%",
                        "",
                        "**Absolute Metrics:**",
                        f"- Throughput: {metrics['absolute_metrics']['throughput_samples_per_second']:.2f} samples/sec",
                        f"- Latency: {metrics['absolute_metrics']['avg_latency_ms']:.2f}ms",
                        f"- Memory: {metrics['absolute_metrics']['memory_usage_mb']:.2f}MB",
                        f"- CPU: {metrics['absolute_metrics']['cpu_usage_percent']:.2f}%",
                        "",
                    ]
                )

            if comparison_report.recommendations:
                lines.extend(["## Recommendations", ""])
                for i, rec in enumerate(comparison_report.recommendations, 1):
                    lines.append(f"{i}. {rec}")
                lines.append("")

            return "\n".join(lines)

        else:  # text format
            lines = [
                "Performance Benchmark Report",
                "=" * 50,
                f"Baseline Configuration: {comparison_report.baseline_config}",
                f"Comparison Configurations: {', '.join(comparison_report.comparison_configs)}",
                "",
                "Performance Comparison:",
                "-" * 30,
            ]

            for config_name, metrics in comparison_report.metrics_comparison.items():
                lines.extend(
                    [
                        f"\n{config_name}:",
                        f"  Throughput Change: {metrics['throughput_change_percent']:+.1f}%",
                        f"  Latency Change: {metrics['latency_change_percent']:+.1f}%",
                        f"  Memory Change: {metrics['memory_change_percent']:+.1f}%",
                        f"  CPU Change: {metrics['cpu_change_percent']:+.1f}%",
                        f"  Overall Improvement: {comparison_report.performance_improvements[config_name]:+.1f}%",
                        "",
                        "  Absolute Metrics:",
                        f"    Throughput: {metrics['absolute_metrics']['throughput_samples_per_second']:.2f} samples/sec",
                        f"    Latency: {metrics['absolute_metrics']['avg_latency_ms']:.2f}ms",
                        f"    Memory: {metrics['absolute_metrics']['memory_usage_mb']:.2f}MB",
                        f"    CPU: {metrics['absolute_metrics']['cpu_usage_percent']:.2f}%",
                    ]
                )

            if comparison_report.recommendations:
                lines.extend(["", "Recommendations:", "-" * 20])
                for i, rec in enumerate(comparison_report.recommendations, 1):
                    lines.append(f"{i}. {rec}")

            return "\n".join(lines)

    def get_results_history(self) -> list[dict[str, Any]]:
        """Get the history of all benchmark results.

        Returns:
            List of historical benchmark results
        """
        return self._results_history.copy()

    def clear_results_history(self) -> None:
        """Clear the benchmark results history."""
        self._results_history.clear()


# Utility functions for system monitoring
def get_system_info() -> dict[str, Any]:
    """Get comprehensive system information for benchmarking context.

    Returns:
        Dictionary with system information
    """
    try:
        cpu_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        }

        memory_info = psutil.virtual_memory()._asdict()

        return {
            "cpu": cpu_info,
            "memory": memory_info,
            "platform": {
                "python_version": __import__("sys").version,
                "platform": __import__("platform").platform(),
            },
        }
    except Exception as e:
        return {"error": f"Failed to get system info: {e}"}


def format_benchmark_result(result: BenchmarkResult, format_type: str = "text") -> str:
    """Format benchmark result for display.

    Args:
        result: Benchmark result to format
        format_type: Output format ("text", "json", or "csv")

    Returns:
        Formatted result string
    """
    if format_type == "json":
        import json

        return json.dumps(
            {
                "function_name": result.function_name,
                "iterations": result.iterations,
                "avg_time_ms": result.avg_time_ms,
                "min_time_ms": result.min_time_ms,
                "max_time_ms": result.max_time_ms,
                "std_dev_ms": result.std_dev_seconds * 1000,
                "throughput_ops_per_second": result.throughput_ops_per_second,
                "memory_usage_mb": result.memory_usage_mb,
                "cpu_usage_percent": result.cpu_usage_percent,
            },
            indent=2,
        )

    elif format_type == "csv":
        return (
            f"{result.function_name},{result.iterations},{result.avg_time_ms:.3f},"
            f"{result.min_time_ms:.3f},{result.max_time_ms:.3f},"
            f"{result.std_dev_seconds * 1000:.3f},{result.throughput_ops_per_second:.2f},"
            f"{result.memory_usage_mb:.2f},{result.cpu_usage_percent:.2f}"
        )

    else:  # text format
        return (
            f"Function: {result.function_name}\n"
            f"Iterations: {result.iterations}\n"
            f"Average time: {result.avg_time_ms:.3f}ms\n"
            f"Min time: {result.min_time_ms:.3f}ms\n"
            f"Max time: {result.max_time_ms:.3f}ms\n"
            f"Std deviation: {result.std_dev_seconds * 1000:.3f}ms\n"
            f"Throughput: {result.throughput_ops_per_second:.2f} ops/sec\n"
            f"Memory usage: {result.memory_usage_mb:.2f}MB\n"
            f"CPU usage: {result.cpu_usage_percent:.2f}%"
        )
