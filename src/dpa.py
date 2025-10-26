import hashlib
import json
import random
import statistics
import sys
import time
from collections.abc import Callable, Generator
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


# Import batch processing components
def _import_batch_components():
    """Dynamically import batch components."""
    try:
        from .batch import BatchConfig, BatchMetrics, BatchProcessor, BatchStrategy

        return True, BatchProcessor, BatchStrategy, BatchConfig, BatchMetrics
    except ImportError:
        try:
            # Try importing from the same directory
            import os
            import sys

            current_dir = os.path.dirname(__file__)
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            import batch

            return (
                True,
                batch.BatchProcessor,
                batch.BatchStrategy,
                batch.BatchConfig,
                batch.BatchMetrics,
            )
        except ImportError:
            return False, None, None, None, None


_BATCH_AVAILABLE, BatchProcessor, BatchStrategy, BatchConfig, BatchMetrics = (
    _import_batch_components()
)


# Import distributed components
def _import_distributed_components():
    """Dynamically import distributed components."""
    try:
        from .distributed import (
            DistributedRangeSplitter,
            RankAwareSeedGenerator,
            gen_distributed_augmentation_params,
            stream_distributed_augmentation_chain,
        )

        return (
            True,
            RankAwareSeedGenerator,
            DistributedRangeSplitter,
            gen_distributed_augmentation_params,
            stream_distributed_augmentation_chain,
        )
    except ImportError:
        try:
            # Try importing from the same directory
            import os
            import sys

            current_dir = os.path.dirname(__file__)
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            import distributed

            return (
                True,
                distributed.RankAwareSeedGenerator,
                distributed.DistributedRangeSplitter,
                distributed.gen_distributed_augmentation_params,
                distributed.stream_distributed_augmentation_chain,
            )
        except ImportError:
            return False, None, None, None, None


(
    _DISTRIBUTED_AVAILABLE,
    RankAwareSeedGenerator,
    DistributedRangeSplitter,
    gen_distributed_augmentation_params,
    stream_distributed_augmentation_chain,
) = _import_distributed_components()


# Import benchmark components
def _import_benchmark_components():
    """Dynamically import benchmark components."""
    try:
        from .benchmark import (
            BenchmarkRunner,
            PerformanceProfiler,
            benchmark_function,
            measure_memory,
            measure_time,
        )

        return (
            True,
            PerformanceProfiler,
            BenchmarkRunner,
            measure_time,
            measure_memory,
            benchmark_function,
        )
    except ImportError:
        try:
            # Try importing from the same directory
            import os
            import sys

            current_dir = os.path.dirname(__file__)
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            import benchmark

            return (
                True,
                benchmark.PerformanceProfiler,
                benchmark.BenchmarkRunner,
                benchmark.measure_time,
                benchmark.measure_memory,
                benchmark.benchmark_function,
            )
        except ImportError:
            return False, None, None, None, None, None


(
    _BENCHMARK_AVAILABLE,
    PerformanceProfiler,
    BenchmarkRunner,
    measure_time,
    measure_memory,
    benchmark_function,
) = _import_benchmark_components()


class StreamingError(Exception):
    """Base exception for streaming operations."""

    pass


class PartialWriteError(StreamingError):
    """Raised when streaming write operation is incomplete."""

    def __init__(self, message: str, samples_written: int) -> None:
        super().__init__(message)
        self.samples_written = samples_written


class StreamingIOError(StreamingError):
    """Raised for I/O errors during streaming operations."""

    pass


class GeneratorExhaustionError(StreamingError):
    """Raised when a generator is unexpectedly exhausted."""

    pass


class ResourceCleanupError(StreamingError):
    """Raised when resource cleanup fails during streaming operations."""

    pass


@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters."""

    rotation_range: tuple[float, float] = (-30, 30)
    brightness_range: tuple[float, float] = (0.8, 1.2)
    noise_range: tuple[float, float] = (0, 0.1)
    scale_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.7, 1.3)
    augmentation_depth: int = 10

    def __post_init__(self) -> None:
        """Validate ranges after initialization."""
        for name, (low, high) in [
            ("rotation", self.rotation_range),
            ("brightness", self.brightness_range),
            ("noise", self.noise_range),
            ("scale", self.scale_range),
            ("contrast", self.contrast_range),
        ]:
            if low >= high:
                raise ValueError(f"{name}_range: lower bound must be < upper bound")

        if self.augmentation_depth < 1:
            raise ValueError("augmentation_depth must be >= 1")


@dataclass
class StreamingMetadata:
    """Metadata for streaming operations."""

    total_samples: int | None = None  # None for unknown/infinite streams
    config: AugmentationConfig | None = None
    start_id: int = 0
    chunk_size: int | None = None
    buffer_size: int = 1000


@dataclass
class StreamingProgress:
    """Progress information for streaming operations."""

    samples_processed: int = 0
    current_sample_id: int = 0
    estimated_total: int | None = None
    start_time: float = 0.0

    def progress_percentage(self) -> float | None:
        """Calculate progress percentage if total is known."""
        if self.estimated_total is None:
            return None
        return (self.samples_processed / self.estimated_total) * 100

    def elapsed_time(self) -> float:
        """Calculate elapsed time since start."""
        return time.time() - self.start_time

    def estimated_time_remaining(self) -> float | None:
        """Estimate remaining time based on current progress."""
        if self.estimated_total is None or self.samples_processed == 0:
            return None

        elapsed = self.elapsed_time()
        if elapsed <= 0:
            return None

        rate = self.samples_processed / elapsed
        remaining_samples = self.estimated_total - self.samples_processed

        return remaining_samples / rate if rate > 0 else None


PRESETS = {
    "mild": AugmentationConfig(
        rotation_range=(-15, 15),
        brightness_range=(0.9, 1.1),
        noise_range=(0, 0.05),
        scale_range=(0.9, 1.1),
        contrast_range=(0.85, 1.15),
    ),
    "moderate": AugmentationConfig(
        rotation_range=(-30, 30),
        brightness_range=(0.8, 1.2),
        noise_range=(0, 0.1),
        scale_range=(0.8, 1.2),
        contrast_range=(0.7, 1.3),
    ),
    "aggressive": AugmentationConfig(
        rotation_range=(-45, 45),
        brightness_range=(0.7, 1.3),
        noise_range=(0, 0.15),
        scale_range=(0.7, 1.3),
        contrast_range=(0.6, 1.4),
    ),
}


def fib(n: int) -> int:
    """Compute nth Fibonacci number with memoization."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0
    if n == 1:
        return 1

    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr

    return curr


def gen_augmentation_seed(seed_id: int, augmentation_depth: int = 10) -> str:
    """
    Generate deterministic hash seed using Fibonacci chain.

    Args:
        seed_id: Unique sample identifier
        augmentation_depth: Number of hash iterations

    Returns:
        SHA256 hash as hexadecimal string

    Raises:
        ValueError: If seed_id is negative or augmentation_depth < 1
    """
    if seed_id < 0:
        raise ValueError("seed_id must be non-negative")
    if augmentation_depth < 1:
        raise ValueError("augmentation_depth must be >= 1")

    prev_hash = hashlib.sha256(str(seed_id).encode()).hexdigest()

    for i in range(augmentation_depth):
        try:
            fib_val = str(fib(i))
            combined = prev_hash + fib_val
            prev_hash = hashlib.sha256(combined.encode()).hexdigest()
        except Exception as e:
            raise RuntimeError(f"Error computing Fibonacci seed at depth {i}: {e}") from e

    return prev_hash


def gen_augmentation_params(
    seed_id: int, config: AugmentationConfig | None = None, rank: int | None = None
) -> dict[str, Any]:
    """
    Generate deterministic augmentation parameters.

    Args:
        seed_id: Unique sample identifier
        config: AugmentationConfig instance (uses defaults if None)
        rank: Optional distributed training rank for rank-aware seeding

    Returns:
        Dictionary with augmentation parameters

    Raises:
        ValueError: If seed_id is invalid or rank is invalid
        ImportError: If rank is provided but distributed module is not available
    """
    if seed_id < 0:
        raise ValueError("seed_id must be non-negative")

    if config is None:
        config = AugmentationConfig()

    # If rank is provided, use distributed parameter generation
    if rank is not None:
        if not _DISTRIBUTED_AVAILABLE:
            raise ImportError(
                "Distributed module not available. "
                "Please ensure the distributed module is properly installed."
            )
        return gen_distributed_augmentation_params(seed_id, rank, config)

    # Standard parameter generation (backward compatibility)
    hash_seed = gen_augmentation_seed(seed_id, config.augmentation_depth)
    seed_int = int(hash_seed, 16) % (2**32)
    random.seed(seed_int)

    return {
        "rotation": random.uniform(*config.rotation_range),
        "brightness": random.uniform(*config.brightness_range),
        "noise": random.uniform(*config.noise_range),
        "scale": random.uniform(*config.scale_range),
        "contrast": random.uniform(*config.contrast_range),
        "hash": hash_seed,
    }


def compute_statistics(
    params_list: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Compute statistics (mean, std, min, max) for augmentation parameters.

    Args:
        params_list: List of augmentation parameter dictionaries

    Returns:
        Dictionary with statistics for each parameter
    """
    stats = {}
    param_keys = ["rotation", "brightness", "noise", "scale", "contrast"]

    for key in param_keys:
        values = [p[key] for p in params_list]
        stats[key] = {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    return stats


def save_augmentation_chain(
    params_list: list[dict[str, Any]],
    filepath: str,
    config: AugmentationConfig | None = None,
    include_stats: bool = True,
) -> None:
    """
    Save augmentation chain to JSON file with optional statistics.

    Args:
        params_list: List of augmentation parameter dictionaries
        filepath: Path to save JSON file
        config: AugmentationConfig used (included in metadata)
        include_stats: If True, include statistics in output

    Raises:
        OSError: If file cannot be written
    """
    try:
        output = {
            "metadata": {
                "num_samples": len(params_list),
                "config": asdict(config) if config else None,
            },
            "augmentations": params_list,
        }

        if include_stats:
            output["statistics"] = compute_statistics(params_list)

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        sys.stdout.write(f"Saved {len(params_list)} augmentations to {filepath}\n")

    except OSError as e:
        raise OSError(f"Failed to save augmentation chain: {e}") from e


def load_augmentation_chain(filepath: str) -> list[dict[str, Any]]:
    """
    Load augmentation chain from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        List of augmentation parameter dictionaries

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is not valid JSON
    """
    try:
        with open(filepath) as f:
            data = json.load(f)

        sys.stdout.write(
            f"Loaded {data['metadata']['num_samples']} augmentations from {filepath}\n"
        )
        return data["augmentations"]

    except FileNotFoundError:
        raise FileNotFoundError(f"Augmentation file not found: {filepath}") from None
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {filepath}: {e}", "", 0) from e


def get_preset(preset_name: str) -> AugmentationConfig:
    """
    Get a preset configuration.

    Args:
        preset_name: Name of preset ("mild", "moderate", "aggressive")

    Returns:
        AugmentationConfig instance

    Raises:
        ValueError: If preset_name is not recognized
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(PRESETS.keys())}")

    return PRESETS[preset_name]


def stream_augmentation_chain(
    num_samples: int,
    config: AugmentationConfig | None = None,
    start_id: int = 0,
    chunk_size: int | None = None,
    batch_config: "BatchConfig | None" = None,
    rank: int | None = None,
    world_size: int | None = None,
    verbose: bool = False,
    progress_callback: Callable[[StreamingProgress, StreamingMetadata], None] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Generate augmentation parameters as a stream/generator.

    This function provides memory-efficient generation of augmentation parameters
    by yielding results one at a time instead of creating a complete list in memory.
    It now supports advanced batch processing strategies when batch_config is provided
    and distributed training capabilities when rank and world_size are specified.

    Args:
        num_samples: Number of samples to generate
        config: AugmentationConfig instance (uses defaults if None)
        start_id: Starting sample ID for custom ranges (default: 0)
        chunk_size: If specified, yields lists of parameters instead of individual items
                   (mutually exclusive with batch_config)
        batch_config: BatchConfig instance for advanced batch processing
                     (mutually exclusive with chunk_size)
        rank: Optional distributed training rank (0-based)
        world_size: Optional total number of processes in distributed training
        verbose: If True, print progress information
        progress_callback: Optional callback function for progress updates

    Yields:
        Individual parameter dictionaries when chunk_size=None and batch_config=None,
        lists of parameter dictionaries when chunk_size is specified,
        or batched parameter lists when batch_config is specified

    Raises:
        ValueError: If num_samples < 0 or start_id < 0, or if both chunk_size and batch_config are specified,
                   or if rank/world_size parameters are invalid
        ImportError: If batch_config is provided but batch processing module is not available,
                    or if rank is provided but distributed module is not available

    Examples:
        Basic streaming usage:
        >>> for params in stream_augmentation_chain(5):
        ...     print(f"Rotation: {params['rotation']:.2f}")

        Chunked streaming (backward compatibility):
        >>> for chunk in stream_augmentation_chain(100, chunk_size=10):
        ...     print(f"Processing batch of {len(chunk)} parameters")

        Advanced batch processing with memory optimization:
        >>> from src.batch import BatchConfig, BatchStrategy
        >>> batch_config = BatchConfig(strategy=BatchStrategy.MEMORY_OPTIMIZED, max_memory_mb=500)
        >>> for batch in stream_augmentation_chain(10000, batch_config=batch_config):
        ...     process_batch_efficiently(batch)

        Adaptive batching for performance optimization:
        >>> batch_config = BatchConfig(strategy=BatchStrategy.ADAPTIVE)
        >>> for batch in stream_augmentation_chain(50000, batch_config=batch_config, verbose=True):
        ...     train_model_batch(batch)

        With custom configuration:
        >>> config = AugmentationConfig(rotation_range=(-45, 45))
        >>> generator = stream_augmentation_chain(1000, config=config)
        >>> first_params = next(generator)

        Memory-efficient processing of large datasets:
        >>> # Process 1 million samples without loading all into memory
        >>> for params in stream_augmentation_chain(1_000_000, verbose=True):
        ...     # Process each parameter individually
        ...     process_sample(params)
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if start_id < 0:
        raise ValueError("start_id must be non-negative")

    # Validate distributed parameters
    if rank is not None or world_size is not None:
        if rank is None or world_size is None:
            raise ValueError("Both rank and world_size must be specified together")
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if rank >= world_size:
            raise ValueError(f"rank ({rank}) must be < world_size ({world_size})")
        if not _DISTRIBUTED_AVAILABLE:
            raise ImportError(
                "Distributed module not available. "
                "Please ensure the distributed module is properly installed."
            )

    # Check for mutually exclusive parameters
    if chunk_size is not None and batch_config is not None:
        raise ValueError("chunk_size and batch_config are mutually exclusive")

    # If distributed parameters are provided, delegate to distributed streaming
    if rank is not None and world_size is not None:
        # Note: distributed streaming doesn't support batch_config or progress_callback yet
        # This is a limitation that could be addressed in future versions
        if batch_config is not None:
            raise ValueError("batch_config is not yet supported with distributed streaming")
        if progress_callback is not None:
            raise ValueError("progress_callback is not yet supported with distributed streaming")

        yield from stream_distributed_augmentation_chain(
            num_samples=num_samples,
            rank=rank,
            world_size=world_size,
            config=config,
            base_seed=start_id,  # Use start_id as base_seed for consistency
            chunk_size=chunk_size,
            verbose=verbose,
        )
        return

    # If batch_config is provided, delegate to the batched streaming function
    if batch_config is not None:
        if not _BATCH_AVAILABLE:
            raise ImportError(
                "Batch processing module not available. "
                "Please ensure the batch module is properly installed."
            )
        yield from stream_augmentation_chain_batched(
            num_samples=num_samples,
            batch_config=batch_config,
            config=config,
            start_id=start_id,
            verbose=verbose,
            progress_callback=progress_callback,
        )
        return

    if config is None:
        config = AugmentationConfig()

    # Create progress tracking
    progress = create_streaming_progress(estimated_total=num_samples, start_id=start_id)
    metadata = create_streaming_metadata(
        total_samples=num_samples, config=config, start_id=start_id, chunk_size=chunk_size
    )

    if verbose and num_samples > 0:
        sys.stdout.write(f"Starting stream generation: {num_samples} samples from ID {start_id}\n")

    if chunk_size is None:
        # Yield individual parameters
        for i, sample_id in enumerate(range(start_id, start_id + num_samples)):
            params = gen_augmentation_params(sample_id, config, rank)

            # Update progress
            update_streaming_progress(progress, i + 1, sample_id)

            if verbose:
                if num_samples > 100 and (i + 1) % max(1, num_samples // 20) == 0:
                    # Show progress for large datasets (every 5%)
                    progress_info = format_progress_info(progress)
                    sys.stdout.write(f"Progress: {progress_info}\n")
                elif num_samples <= 100:
                    # Show individual samples for small datasets
                    sys.stdout.write(
                        f"Sample({sample_id}) -> rotation={params['rotation']:.2f}° "
                        f"brightness={params['brightness']:.2f} "
                        f"noise={params['noise']:.3f} "
                        f"scale={params['scale']:.2f} "
                        f"contrast={params['contrast']:.2f}\n"
                    )

            # Call progress callback if provided
            if progress_callback:
                progress_callback(progress, metadata)

            yield params
    else:
        # Yield chunks of parameters
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        chunk = []
        for i, sample_id in enumerate(range(start_id, start_id + num_samples)):
            params = gen_augmentation_params(sample_id, config, rank)
            chunk.append(params)

            if verbose and num_samples <= 100:
                sys.stdout.write(
                    f"Sample({sample_id}) -> rotation={params['rotation']:.2f}° "
                    f"brightness={params['brightness']:.2f} "
                    f"noise={params['noise']:.3f} "
                    f"scale={params['scale']:.2f} "
                    f"contrast={params['contrast']:.2f}\n"
                )

            if len(chunk) == chunk_size:
                # Update progress before yielding chunk
                update_streaming_progress(progress, i + 1, sample_id)

                if verbose and num_samples > 100 and (i + 1) % max(1, num_samples // 20) == 0:
                    progress_info = format_progress_info(progress)
                    sys.stdout.write(f"Progress: {progress_info}\n")

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(progress, metadata)

                yield chunk  # pyright: ignore[reportReturnType]
                chunk = []

        # Yield remaining items if any
        if chunk:
            # Update final progress
            update_streaming_progress(progress, num_samples, start_id + num_samples - 1)

            if progress_callback:
                progress_callback(progress, metadata)

            yield chunk  # pyright: ignore[reportReturnType]

    if verbose and num_samples > 0:
        final_progress = format_progress_info(progress)
        sys.stdout.write(f"Stream generation completed: {final_progress}\n")


def stream_augmentation_range(
    start_id: int,
    end_id: int,
    config: AugmentationConfig | None = None,
    chunk_size: int | None = None,
    batch_config: "BatchConfig | None" = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Stream parameters for a specific range of sample IDs.

    This function allows streaming augmentation parameters for a specific range
    of sample IDs, useful for processing subsets of data or resuming operations.
    It now supports advanced batch processing strategies when batch_config is provided
    and distributed training capabilities when rank and world_size are specified.

    Args:
        start_id: Starting sample ID (inclusive)
        end_id: Ending sample ID (exclusive)
        config: AugmentationConfig instance (uses defaults if None)
        chunk_size: If specified, yields lists of parameters instead of individual items
                   (mutually exclusive with batch_config)
        batch_config: BatchConfig instance for advanced batch processing
                     (mutually exclusive with chunk_size)
        rank: Optional distributed training rank (0-based)
        world_size: Optional total number of processes in distributed training

    Yields:
        Individual parameter dictionaries when chunk_size=None and batch_config=None,
        lists of parameter dictionaries when chunk_size is specified,
        or batched parameter lists when batch_config is specified

    Raises:
        ValueError: If start_id < 0, end_id < start_id, chunk_size <= 0, or if both chunk_size and batch_config are specified,
                   or if rank/world_size parameters are invalid
        ImportError: If batch_config is provided but batch processing module is not available,
                    or if rank is provided but distributed module is not available

    Examples:
        Stream a specific range:
        >>> for params in stream_augmentation_range(100, 200):
        ...     print(f"Sample {params.get('sample_id', 'unknown')}: {params['rotation']:.2f}")

        Process data in chunks for a range (backward compatibility):
        >>> for chunk in stream_augmentation_range(0, 1000, chunk_size=50):
        ...     batch_process(chunk)

        Advanced batch processing for a range:
        >>> from src.batch import BatchConfig, BatchStrategy
        >>> batch_config = BatchConfig(strategy=BatchStrategy.MEMORY_OPTIMIZED)
        >>> for batch in stream_augmentation_range(1000, 5000, batch_config=batch_config):
        ...     process_batch_efficiently(batch)

        Resume processing from a specific point:
        >>> # Resume from sample 5000 to 10000
        >>> generator = stream_augmentation_range(5000, 10000)
        >>> for params in generator:
        ...     if should_stop():
        ...         break
        ...     process_sample(params)
    """
    if start_id < 0:
        raise ValueError("start_id must be non-negative")
    if end_id < start_id:
        raise ValueError("end_id must be >= start_id")

    num_samples = end_id - start_id
    yield from stream_augmentation_chain(
        num_samples=num_samples,
        config=config,
        start_id=start_id,
        chunk_size=chunk_size,
        batch_config=batch_config,
        rank=rank,
        world_size=world_size,
        verbose=False,
    )


def compute_streaming_statistics(
    param_generator: Generator[dict[str, Any], None, None],
) -> dict[str, dict[str, float]]:
    """
    Compute statistics from a generator using online algorithms (Welford's method).

    This function computes statistics without consuming the original generator by
    creating a tee'd copy for computation while preserving the original. It uses
    Welford's online algorithm for numerically stable variance computation.

    Args:
        param_generator: Generator yielding augmentation parameters

    Returns:
        Dictionary with statistics for each parameter (mean, stdev, min, max, count)

    Note:
        This function uses Welford's online algorithm for numerically stable variance computation.
        The statistics computed match exactly with the existing compute_statistics() function.

    Examples:
        Compute statistics from a stream:
        >>> generator = stream_augmentation_chain(1000)
        >>> stats = compute_streaming_statistics(generator)
        >>> print(f"Mean rotation: {stats['rotation']['mean']:.2f}")

        Compare streaming vs batch statistics:
        >>> # Both should produce identical results
        >>> params_list = generate_augmentation_chain(100)
        >>> batch_stats = compute_statistics(params_list)
        >>>
        >>> generator = stream_augmentation_chain(100)
        >>> streaming_stats = compute_streaming_statistics(generator)
        >>> assert batch_stats == streaming_stats

        Memory-efficient statistics for large datasets:
        >>> # Compute stats for 1M samples without loading all into memory
        >>> large_generator = stream_augmentation_chain(1_000_000)
        >>> stats = compute_streaming_statistics(large_generator)
    """

    # Create two independent iterators from the generator
    # Note: This approach works for generators but requires consuming the generator
    # For true non-consuming behavior, we'd need the generator to be restartable
    param_keys = ["rotation", "brightness", "noise", "scale", "contrast"]

    # Initialize accumulators for Welford's algorithm
    counts = {key: 0 for key in param_keys}
    means = {key: 0.0 for key in param_keys}
    m2s = {key: 0.0 for key in param_keys}  # Sum of squares of differences from mean
    mins = {key: float("inf") for key in param_keys}
    maxs = {key: float("-inf") for key in param_keys}

    for params in param_generator:
        for key in param_keys:
            value = params[key]
            counts[key] += 1

            # Update min/max
            mins[key] = min(mins[key], value)
            maxs[key] = max(maxs[key], value)

            # Welford's online algorithm for mean and variance
            delta = value - means[key]
            means[key] += delta / counts[key]
            delta2 = value - means[key]
            m2s[key] += delta * delta2

    # Compute final statistics
    stats = {}
    for key in param_keys:
        count = counts[key]
        if count == 0:
            stats[key] = {
                "mean": 0.0,
                "stdev": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }
        else:
            # Use sample variance (n-1) for consistency with statistics.stdev()
            variance = m2s[key] / (count - 1) if count > 1 else 0.0
            stats[key] = {
                "mean": means[key],
                "stdev": variance**0.5,
                "min": mins[key],
                "max": maxs[key],
                "count": count,
            }

    return stats


def _compute_streaming_statistics(
    param_generator: Generator[dict[str, Any], None, None],
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """
    Compute statistics from a generator using online algorithms (Welford's method).

    Args:
        param_generator: Generator yielding augmentation parameters

    Returns:
        Tuple of (statistics dict, list of all parameters)

    Note:
        This function consumes the generator and returns both statistics and the full list.
        It uses Welford's online algorithm for numerically stable variance computation.
    """
    param_keys = ["rotation", "brightness", "noise", "scale", "contrast"]

    # Initialize accumulators for Welford's algorithm
    counts = {key: 0 for key in param_keys}
    means = {key: 0.0 for key in param_keys}
    m2s = {key: 0.0 for key in param_keys}  # Sum of squares of differences from mean
    mins = {key: float("inf") for key in param_keys}
    maxs = {key: float("-inf") for key in param_keys}

    all_params = []

    for params in param_generator:
        all_params.append(params)

        for key in param_keys:
            value = params[key]
            counts[key] += 1

            # Update min/max
            mins[key] = min(mins[key], value)
            maxs[key] = max(maxs[key], value)

            # Welford's online algorithm for mean and variance
            delta = value - means[key]
            means[key] += delta / counts[key]
            delta2 = value - means[key]
            m2s[key] += delta * delta2

    # Compute final statistics
    stats = {}
    for key in param_keys:
        count = counts[key]
        if count == 0:
            stats[key] = {
                "mean": 0.0,
                "stdev": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }
        else:
            # Use sample variance (n-1) for consistency with statistics.stdev()
            variance = m2s[key] / (count - 1) if count > 1 else 0.0
            stats[key] = {
                "mean": means[key],
                "stdev": variance**0.5,
                "min": mins[key],
                "max": maxs[key],
                "count": count,
            }

    return stats, all_params


def save_augmentation_stream(
    param_generator: Generator[dict[str, Any], None, None],
    filepath: str,
    config: AugmentationConfig | None = None,
    include_stats: bool = True,
    buffer_size: int = 1000,
    verbose: bool = False,
    progress_callback: Callable[[StreamingProgress, StreamingMetadata], None] | None = None,
) -> None:
    """
    Save streaming parameters to JSON file with incremental writing.

    This function enables memory-efficient saving of large datasets by writing
    parameters incrementally to disk rather than accumulating them in memory.

    Args:
        param_generator: Generator yielding augmentation parameters
        filepath: Path to save JSON file
        config: AugmentationConfig used (included in metadata)
        include_stats: If True, include statistics in output
        buffer_size: Number of parameters to buffer before writing
        verbose: If True, print progress information
        progress_callback: Optional callback function for progress updates

    Raises:
        StreamingIOError: If file cannot be written
        PartialWriteError: If streaming write operation is incomplete
        ResourceCleanupError: If resource cleanup fails

    Examples:
        Save a large dataset efficiently:
        >>> generator = stream_augmentation_chain(1_000_000)
        >>> save_augmentation_stream(generator, "large_dataset.json", verbose=True)

        Save without statistics for faster processing:
        >>> generator = stream_augmentation_chain(100_000)
        >>> save_augmentation_stream(
        ...     generator,
        ...     "fast_save.json",
        ...     include_stats=False,
        ...     buffer_size=5000
        ... )

        Save with custom configuration:
        >>> config = AugmentationConfig(rotation_range=(-90, 90))
        >>> generator = stream_augmentation_chain(10000, config=config)
        >>> save_augmentation_stream(generator, "custom_config.json", config=config)

        Save with progress tracking:
        >>> def progress_handler(progress, metadata):
        ...     if progress.progress_percentage():
        ...         print(f"Progress: {progress.progress_percentage():.1f}%")
        >>>
        >>> generator = stream_augmentation_chain(50000)
        >>> save_augmentation_stream(
        ...     generator,
        ...     "tracked_save.json",
        ...     progress_callback=progress_handler
        ... )
    """
    file_handle = None
    samples_written = 0

    try:
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        buffer = []

        # Create progress tracking (unknown total for streaming)
        progress = create_streaming_progress(estimated_total=None, start_id=0)
        metadata = create_streaming_metadata(
            total_samples=None, config=config, buffer_size=buffer_size
        )

        if verbose:
            sys.stdout.write(f"Starting streaming save to {filepath}\n")

        # If we need statistics, we must consume the entire generator first
        if include_stats:
            try:
                if verbose:
                    sys.stdout.write("Computing statistics from stream...\n")

                stats, all_params = _compute_streaming_statistics(param_generator)
                total_samples = len(all_params)

                # Update progress tracking with known total
                progress.estimated_total = total_samples
                update_streaming_progress(progress, total_samples, total_samples - 1)

                # Write the complete file with statistics
                output = {
                    "metadata": {
                        "num_samples": total_samples,
                        "config": asdict(config) if config else None,
                        "streaming": True,
                    },
                    "augmentations": all_params,
                    "statistics": stats,
                }

                file_handle = open(filepath, "w")
                try:
                    json.dump(output, file_handle, indent=2)
                    samples_written = total_samples
                finally:
                    safe_close_file(file_handle, filepath)
                    file_handle = None

                if verbose:
                    final_progress = format_progress_info(progress)
                    sys.stdout.write(
                        f"Saved {samples_written} augmentations to {filepath} ({final_progress})\n"
                    )
                else:
                    sys.stdout.write(f"Saved {samples_written} augmentations to {filepath}\n")

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(progress, metadata)

            except GeneratorExhaustionError:
                raise PartialWriteError(
                    "Generator exhausted unexpectedly during statistics computation", 0
                )
            except Exception as e:
                raise PartialWriteError(f"Failed to compute statistics and save: {e}", 0) from e

        else:
            # Stream without statistics - write incrementally
            file_handle = None
            try:
                file_handle = open(filepath, "w")

                # Write opening structure
                file_handle.write("{\n")
                file_handle.write('  "metadata": {\n')
                file_handle.write('    "num_samples": null,\n')
                file_handle.write('    "config": ')
                json.dump(asdict(config) if config else None, file_handle)
                file_handle.write(",\n")
                file_handle.write('    "streaming": true\n')
                file_handle.write("  },\n")
                file_handle.write('  "augmentations": [\n')

                first_item = True

                try:
                    for params in param_generator:
                        buffer.append(params)

                        # Write buffer when full
                        if len(buffer) >= buffer_size:
                            for _i, buffered_params in enumerate(buffer):
                                if not first_item:
                                    file_handle.write(",\n")
                                else:
                                    first_item = False

                                file_handle.write("    ")
                                json.dump(buffered_params, file_handle)
                                samples_written += 1

                            # Update progress
                            update_streaming_progress(
                                progress, samples_written, samples_written - 1
                            )

                            if verbose and samples_written % (buffer_size * 10) == 0:
                                progress_info = format_progress_info(progress)
                                sys.stdout.write(f"Streaming progress: {progress_info}\n")

                            # Call progress callback if provided
                            if progress_callback:
                                progress_callback(progress, metadata)

                            buffer = []
                            file_handle.flush()  # Ensure data is written

                    # Write remaining buffer
                    for _i, buffered_params in enumerate(buffer):
                        if not first_item:
                            file_handle.write(",\n")
                        else:
                            first_item = False

                        file_handle.write("    ")
                        json.dump(buffered_params, file_handle)
                        samples_written += 1

                    # Final progress update
                    update_streaming_progress(progress, samples_written, samples_written - 1)

                    # Close the JSON structure
                    file_handle.write("\n  ]\n")
                    file_handle.write("}\n")

                except StopIteration:
                    # Generator exhausted normally
                    pass
                except GeneratorExit:
                    # Generator was closed
                    if verbose:
                        sys.stdout.write("Generator was closed during streaming\n")
                except Exception as e:
                    raise PartialWriteError(
                        f"Failed during streaming write: {e}", samples_written
                    ) from e
                finally:
                    # Ensure generator is properly cleaned up
                    try:
                        safe_cleanup_generator(param_generator)
                    except ResourceCleanupError as cleanup_error:
                        if verbose:
                            sys.stdout.write(f"Warning: {cleanup_error}\n")

                if verbose:
                    final_progress = format_progress_info(progress)
                    sys.stdout.write(
                        f"Saved {samples_written} augmentations to {filepath} ({final_progress})\n"
                    )
                else:
                    sys.stdout.write(f"Saved {samples_written} augmentations to {filepath}\n")

                # Final progress callback
                if progress_callback:
                    progress_callback(progress, metadata)

            finally:
                if file_handle is not None:
                    safe_close_file(file_handle, filepath)

    except OSError as e:
        raise StreamingIOError(f"I/O error during streaming save: {e}") from e
    except Exception as e:
        # Ensure cleanup happens even on unexpected errors
        if file_handle is not None:
            try:
                safe_close_file(file_handle, filepath)
            except ResourceCleanupError as cleanup_error:
                if verbose:
                    sys.stdout.write(f"Warning during cleanup: {cleanup_error}\n")

        # Re-raise the original exception
        if isinstance(e, (StreamingError, PartialWriteError, StreamingIOError)):
            raise
        else:
            raise StreamingIOError(f"Unexpected error during streaming save: {e}") from e


def load_augmentation_stream(
    filepath: str,
    chunk_size: int = 1000,
    verbose: bool = False,
    progress_callback: Callable[[StreamingProgress, StreamingMetadata], None] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Load and stream parameters from JSON file without loading entire file into memory.

    This function provides memory-efficient loading of augmentation parameters
    from JSON files by yielding results incrementally rather than loading
    the entire file into memory at once.

    Args:
        filepath: Path to JSON file
        chunk_size: Number of parameters to yield at once (for chunked streaming)
        verbose: If True, print progress information
        progress_callback: Optional callback function for progress updates

    Yields:
        Individual parameter dictionaries when chunk_size=1,
        or lists of parameter dictionaries when chunk_size > 1

    Raises:
        FileNotFoundError: If file does not exist
        StreamingIOError: If file cannot be read or is invalid JSON
        ResourceCleanupError: If resource cleanup fails

    Examples:
        Load and process parameters one by one:
        >>> for params in load_augmentation_stream("dataset.json", chunk_size=1):
        ...     process_single_sample(params)

        Load in chunks for batch processing:
        >>> for chunk in load_augmentation_stream("dataset.json", chunk_size=100):
        ...     batch_process(chunk)

        Load with progress tracking:
        >>> def progress_handler(progress, metadata):
        ...     print(f"Loaded {progress.samples_processed} samples")
        >>>
        >>> for params in load_augmentation_stream(
        ...     "large_dataset.json",
        ...     verbose=True,
        ...     progress_callback=progress_handler
        ... ):
        ...     process_sample(params)

        Memory-efficient processing of large files:
        >>> # Process a 10GB file without loading it all into memory
        >>> for params in load_augmentation_stream("huge_dataset.json"):
        ...     if meets_criteria(params):
        ...         filtered_process(params)
    """
    file_handle = None

    try:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Augmentation file not found: {filepath}")

        file_handle = open(filepath)

        try:
            data = json.load(file_handle)
        except json.JSONDecodeError as e:
            raise StreamingIOError(f"Invalid JSON in {filepath}: {e}") from e
        finally:
            safe_close_file(file_handle, filepath)
            file_handle = None

        if "augmentations" not in data:
            raise StreamingIOError("Invalid augmentation file format: missing 'augmentations' key")

        augmentations = data["augmentations"]
        total_samples = len(augmentations)

        # Validate that augmentations is a list
        if not isinstance(augmentations, list):
            raise StreamingIOError(
                "Invalid augmentation file format: 'augmentations' must be a list"
            )

        # Create progress tracking
        progress = create_streaming_progress(estimated_total=total_samples, start_id=0)
        metadata = create_streaming_metadata(total_samples=total_samples, chunk_size=chunk_size)

        if verbose:
            sys.stdout.write(f"Loading {total_samples} augmentations from {filepath}\n")
        else:
            sys.stdout.write(f"Loading {total_samples} augmentations from {filepath}\n")

        try:
            if chunk_size <= 1:
                # Yield individual parameters
                for i, params in enumerate(augmentations):
                    # Validate parameter structure
                    if not isinstance(params, dict):
                        raise StreamingIOError(
                            f"Invalid parameter format at index {i}: expected dict, got {type(params)}"
                        )

                    update_streaming_progress(progress, i + 1, i)

                    if (
                        verbose
                        and total_samples > 100
                        and (i + 1) % max(1, total_samples // 20) == 0
                    ):
                        progress_info = format_progress_info(progress)
                        sys.stdout.write(f"Loading progress: {progress_info}\n")

                    if progress_callback:
                        progress_callback(progress, metadata)

                    yield params
            else:
                # Yield chunks of parameters
                chunk = []
                items_yielded = 0

                for i, params in enumerate(augmentations):
                    # Validate parameter structure
                    if not isinstance(params, dict):
                        raise StreamingIOError(
                            f"Invalid parameter format at index {i}: expected dict, got {type(params)}"
                        )

                    chunk.append(params)

                    if len(chunk) >= chunk_size:
                        items_yielded += len(chunk)
                        update_streaming_progress(progress, items_yielded, i)

                        if (
                            verbose
                            and total_samples > 100
                            and items_yielded % (chunk_size * 10) == 0
                        ):
                            progress_info = format_progress_info(progress)
                            sys.stdout.write(f"Loading progress: {progress_info}\n")

                        if progress_callback:
                            progress_callback(progress, metadata)

                        yield chunk  # pyright: ignore[reportReturnType]
                        chunk = []

                # Yield remaining items if any
                if chunk:
                    items_yielded += len(chunk)
                    update_streaming_progress(progress, items_yielded, len(augmentations) - 1)

                    if progress_callback:
                        progress_callback(progress, metadata)

                    yield chunk  # pyright: ignore[reportReturnType]

            if verbose:
                final_progress = format_progress_info(progress)
                sys.stdout.write(f"Loading completed: {final_progress}\n")

        except GeneratorExit:
            # Generator was closed
            if verbose:
                sys.stdout.write("Loading generator was closed\n")
        except Exception as e:
            raise StreamingIOError(f"Error during streaming load: {e}") from e

    except FileNotFoundError:
        raise
    except OSError as e:
        raise StreamingIOError(f"I/O error during streaming load: {e}") from e
    except Exception as e:
        if isinstance(e, StreamingIOError):
            raise
        else:
            raise StreamingIOError(f"Unexpected error during streaming load: {e}") from e
    finally:
        if file_handle is not None:
            try:
                safe_close_file(file_handle, filepath)
            except ResourceCleanupError as cleanup_error:
                if verbose:
                    sys.stdout.write(f"Warning during cleanup: {cleanup_error}\n")


def generator_to_list(
    param_generator: Generator[dict[str, Any], None, None],
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """
    Convert generator to list with optional size limit for compatibility.

    This utility function provides compatibility between streaming and batch APIs
    by converting a generator to a list, with optional size limiting to prevent
    memory issues with large datasets.

    Args:
        param_generator: Generator yielding augmentation parameters
        max_items: Maximum number of items to collect (None for unlimited)

    Returns:
        List of augmentation parameter dictionaries

    Raises:
        ValueError: If max_items is negative

    Examples:
        Convert entire generator to list:
        >>> generator = stream_augmentation_chain(100)
        >>> params_list = generator_to_list(generator)
        >>> print(f"Collected {len(params_list)} parameters")

        Limit collection size for safety:
        >>> large_generator = stream_augmentation_chain(1_000_000)
        >>> sample = generator_to_list(large_generator, max_items=1000)
        >>> print(f"Sampled {len(sample)} parameters from large dataset")

        Use with existing batch functions:
        >>> generator = stream_augmentation_chain(50)
        >>> params_list = generator_to_list(generator)
        >>> stats = compute_statistics(params_list)  # Use batch function
    """
    if max_items is not None and max_items < 0:
        raise ValueError("max_items must be non-negative")

    result = []
    for i, params in enumerate(param_generator):
        if max_items is not None and i >= max_items:
            break
        result.append(params)

    return result


def list_to_generator(
    params_list: list[dict[str, Any]],
) -> Generator[dict[str, Any], None, None]:
    """
    Convert list to generator for uniform interface.

    This utility function provides compatibility between batch and streaming APIs
    by converting a list to a generator, enabling uniform processing patterns.

    Args:
        params_list: List of augmentation parameter dictionaries

    Yields:
        Individual parameter dictionaries

    Examples:
        Convert batch data to streaming format:
        >>> params_list = generate_augmentation_chain(100)
        >>> generator = list_to_generator(params_list)
        >>> save_augmentation_stream(generator, "converted.json")

        Use batch data with streaming functions:
        >>> existing_data = load_augmentation_chain("existing.json")
        >>> generator = list_to_generator(existing_data)
        >>> stats = compute_streaming_statistics(generator)

        Chain with other streaming operations:
        >>> params_list = generate_augmentation_chain(50)
        >>> generator = list_to_generator(params_list)
        >>> for params in generator:
        ...     if params['rotation'] > 20:
        ...         process_high_rotation(params)
    """
    for params in params_list:
        yield from [params]


def create_streaming_metadata(
    total_samples: int | None = None,
    config: AugmentationConfig | None = None,
    start_id: int = 0,
    chunk_size: int | None = None,
    buffer_size: int = 1000,
) -> StreamingMetadata:
    """
    Create streaming metadata configuration.

    This function creates a StreamingMetadata instance with validated parameters
    for use in streaming operations.

    Args:
        total_samples: Total number of samples (None for unknown/infinite streams)
        config: AugmentationConfig instance
        start_id: Starting sample ID
        chunk_size: Chunk size for batched operations
        buffer_size: Buffer size for I/O operations

    Returns:
        StreamingMetadata instance

    Raises:
        ValueError: If parameters are invalid

    Examples:
        Create metadata for a known dataset size:
        >>> metadata = create_streaming_metadata(
        ...     total_samples=10000,
        ...     config=AugmentationConfig(),
        ...     buffer_size=500
        ... )

        Create metadata for streaming with unknown size:
        >>> metadata = create_streaming_metadata(
        ...     total_samples=None,  # Unknown size
        ...     chunk_size=100
        ... )
    """
    if start_id < 0:
        raise ValueError("start_id must be non-negative")
    if total_samples is not None and total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if buffer_size <= 0:
        raise ValueError("buffer_size must be positive")

    return StreamingMetadata(
        total_samples=total_samples,
        config=config,
        start_id=start_id,
        chunk_size=chunk_size,
        buffer_size=buffer_size,
    )


def create_streaming_progress(
    estimated_total: int | None = None,
    start_id: int = 0,
) -> StreamingProgress:
    """
    Create streaming progress tracker.

    This function creates a StreamingProgress instance initialized with the
    current time for tracking progress during streaming operations.

    Args:
        estimated_total: Estimated total number of samples
        start_id: Starting sample ID

    Returns:
        StreamingProgress instance initialized with current time

    Raises:
        ValueError: If parameters are invalid

    Examples:
        Create progress tracker for known dataset:
        >>> progress = create_streaming_progress(estimated_total=10000)
        >>> print(f"Started at: {progress.start_time}")

        Create progress tracker for unknown size:
        >>> progress = create_streaming_progress(estimated_total=None)
        >>> # Progress percentage will be None until total is known
    """
    if estimated_total is not None and estimated_total < 0:
        raise ValueError("estimated_total must be non-negative")
    if start_id < 0:
        raise ValueError("start_id must be non-negative")

    return StreamingProgress(
        samples_processed=0,
        current_sample_id=start_id,
        estimated_total=estimated_total,
        start_time=time.time(),
    )


def update_streaming_progress(
    progress: StreamingProgress,
    samples_processed: int,
    current_sample_id: int,
) -> None:
    """
    Update streaming progress tracker.

    Args:
        progress: StreamingProgress instance to update
        samples_processed: Number of samples processed so far
        current_sample_id: Current sample ID being processed

    Raises:
        ValueError: If parameters are invalid
    """
    if samples_processed < 0:
        raise ValueError("samples_processed must be non-negative")
    if current_sample_id < 0:
        raise ValueError("current_sample_id must be non-negative")

    progress.samples_processed = samples_processed
    progress.current_sample_id = current_sample_id


def estimate_memory_usage(
    num_samples: int,
    include_statistics: bool = True,
    bytes_per_param: int = 200,
) -> dict[str, int]:
    """
    Estimate memory usage for augmentation operations.

    Args:
        num_samples: Number of samples to process
        include_statistics: Whether statistics will be computed
        bytes_per_param: Estimated bytes per parameter dictionary

    Returns:
        Dictionary with memory estimates in bytes

    Raises:
        ValueError: If num_samples is negative
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")

    # Base memory for parameter dictionaries
    params_memory = num_samples * bytes_per_param

    # Additional memory for statistics computation (if enabled)
    stats_memory = 0
    if include_statistics:
        # Memory for accumulators and intermediate values
        stats_memory = 1024  # Small fixed overhead for statistics

    # Memory for JSON serialization (approximate)
    json_memory = int(params_memory * 1.2)  # 20% overhead for JSON formatting

    # Total memory estimate
    total_memory = params_memory + stats_memory + json_memory

    return {
        "parameters": params_memory,
        "statistics": stats_memory,
        "json_serialization": json_memory,
        "total": total_memory,
    }


def get_current_memory_usage() -> dict[str, int]:
    """
    Get current process memory usage.

    Returns:
        Dictionary with memory usage information in bytes

    Note:
        Requires psutil package. Returns empty dict if not available.
    """
    if not _PSUTIL_AVAILABLE:
        return {}

    try:
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss": memory_info.rss,  # Resident Set Size (physical memory)
            "vms": memory_info.vms,  # Virtual Memory Size
            "percent": int(process.memory_percent()),  # Percentage of system memory
        }
    except (AttributeError, OSError):
        # Method not supported on this platform or process access denied
        return {}


def format_memory_size(bytes_size: int) -> str:
    """
    Format memory size in human-readable format.

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB", "2.3 GB")
    """
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024**2:
        return f"{bytes_size / 1024:.1f} KB"
    elif bytes_size < 1024**3:
        return f"{bytes_size / (1024**2):.1f} MB"
    else:
        return f"{bytes_size / (1024**3):.1f} GB"


def format_progress_info(progress: StreamingProgress) -> str:
    """
    Format progress information as human-readable string.

    Args:
        progress: StreamingProgress instance

    Returns:
        Formatted progress string
    """
    parts = []

    # Add sample count
    if progress.estimated_total is not None:
        parts.append(f"{progress.samples_processed}/{progress.estimated_total}")
    else:
        parts.append(f"{progress.samples_processed}")

    # Add percentage if available
    percentage = progress.progress_percentage()
    if percentage is not None:
        parts.append(f"({percentage:.1f}%)")

    # Add elapsed time
    elapsed = progress.elapsed_time()
    parts.append(f"elapsed: {elapsed:.1f}s")

    # Add estimated time remaining if available
    eta = progress.estimated_time_remaining()
    if eta is not None:
        parts.append(f"ETA: {eta:.1f}s")

    return " ".join(parts)


def create_simple_progress_callback(
    verbose: bool = True,
) -> Callable[[StreamingProgress, StreamingMetadata], None]:
    """
    Create a simple progress callback function for streaming operations.

    Args:
        verbose: If True, print detailed progress information

    Returns:
        Callback function that accepts (progress, metadata) arguments
    """

    def progress_callback(progress: StreamingProgress, metadata: StreamingMetadata) -> None:
        """Simple progress callback that prints progress information."""
        if verbose:
            progress_info = format_progress_info(progress)
            operation = "streaming"
            if metadata.total_samples is not None:
                operation = f"processing {metadata.total_samples} samples"
            sys.stdout.write(f"[{operation}] {progress_info}\n")

    return progress_callback


def safe_close_file(file_handle: Any, filepath: str = "") -> None:
    """
    Safely close a file handle with error handling.

    Args:
        file_handle: File handle to close
        filepath: Optional filepath for error messages

    Raises:
        ResourceCleanupError: If file cannot be closed properly
    """
    if file_handle is None:
        return

    try:
        if hasattr(file_handle, "close"):
            file_handle.close()
    except Exception as e:
        error_msg = "Failed to close file handle"
        if filepath:
            error_msg += f" for {filepath}"
        raise ResourceCleanupError(f"{error_msg}: {e}") from e


def safe_cleanup_generator(generator: Generator[Any, None, None]) -> None:
    """
    Safely cleanup a generator by closing it.

    Args:
        generator: Generator to cleanup

    Raises:
        ResourceCleanupError: If generator cannot be closed properly
    """
    if generator is None:
        return

    try:
        if hasattr(generator, "close"):
            generator.close()
    except Exception as e:
        raise ResourceCleanupError(f"Failed to cleanup generator: {e}") from e


def recover_partial_write(
    filepath: str, expected_samples: int | None = None, verbose: bool = False
) -> dict[str, Any]:
    """
    Attempt to recover information from a partially written streaming file.

    Args:
        filepath: Path to the potentially corrupted file
        expected_samples: Expected number of samples (if known)
        verbose: If True, print recovery information

    Returns:
        Dictionary with recovery information including:
        - 'recoverable': bool indicating if file can be recovered
        - 'samples_found': number of samples successfully parsed
        - 'metadata': recovered metadata if available
        - 'error': error message if recovery failed

    Raises:
        StreamingIOError: If file cannot be accessed
    """
    recovery_info = {"recoverable": False, "samples_found": 0, "metadata": None, "error": None}

    try:
        if not Path(filepath).exists():
            recovery_info["error"] = "File does not exist"
            return recovery_info

        if verbose:
            sys.stdout.write(f"Attempting to recover partial write from {filepath}\n")

        with open(filepath) as f:
            content = f.read()

        # Try to parse as much JSON as possible
        try:
            # First, try to parse the complete file
            data = json.loads(content)
            recovery_info["recoverable"] = True
            recovery_info["metadata"] = data.get("metadata", {})

            if "augmentations" in data:
                recovery_info["samples_found"] = len(data["augmentations"])

            if verbose:
                sys.stdout.write(
                    f"File is complete with {recovery_info['samples_found']} samples\n"
                )

            return recovery_info

        except json.JSONDecodeError:
            # File is incomplete, try to recover what we can
            if verbose:
                sys.stdout.write("File appears to be incomplete, attempting partial recovery\n")

            # Look for the metadata section
            metadata_match = content.find('"metadata"')
            if metadata_match != -1:
                try:
                    # Try to extract metadata
                    metadata_start = content.find("{", metadata_match)
                    metadata_end = content.find("},", metadata_start)
                    if metadata_end != -1:
                        metadata_json = content[metadata_start : metadata_end + 1]
                        recovery_info["metadata"] = json.loads(metadata_json)
                        if verbose:
                            sys.stdout.write("Successfully recovered metadata\n")
                except json.JSONDecodeError:
                    if verbose:
                        sys.stdout.write("Could not recover metadata\n")

            # Count recoverable augmentation entries
            augmentations_start = content.find('"augmentations": [')
            if augmentations_start != -1:
                # Count complete JSON objects in the augmentations array
                samples_found = 0
                search_pos = augmentations_start

                while True:
                    # Look for complete parameter objects
                    obj_start = content.find('{"rotation"', search_pos)
                    if obj_start == -1:
                        break

                    obj_end = content.find("}", obj_start)
                    if obj_end == -1:
                        break

                    # Try to parse this object
                    try:
                        obj_json = content[obj_start : obj_end + 1]
                        json.loads(obj_json)
                        samples_found += 1
                        search_pos = obj_end + 1
                    except json.JSONDecodeError:
                        break

                recovery_info["samples_found"] = samples_found
                recovery_info["recoverable"] = samples_found > 0

                if verbose:
                    sys.stdout.write(f"Recovered {samples_found} complete samples\n")

            if not recovery_info["recoverable"]:
                recovery_info["error"] = "No recoverable data found in file"

    except OSError as e:
        recovery_info["error"] = f"I/O error during recovery: {e}"
        raise StreamingIOError(f"Cannot access file for recovery: {e}") from e

    return recovery_info


def resume_streaming_save(
    param_generator: Generator[dict[str, Any], None, None],
    filepath: str,
    resume_from_sample: int = 0,
    config: AugmentationConfig | None = None,
    include_stats: bool = True,
    buffer_size: int = 1000,
    verbose: bool = False,
) -> None:
    """
    Resume a streaming save operation from a specific sample ID.

    Args:
        param_generator: Generator yielding augmentation parameters
        filepath: Path to save JSON file
        resume_from_sample: Sample ID to resume from (skips earlier samples)
        config: AugmentationConfig used (included in metadata)
        include_stats: If True, include statistics in output
        buffer_size: Number of parameters to buffer before writing
        verbose: If True, print progress information

    Raises:
        StreamingIOError: If file cannot be written
        PartialWriteError: If streaming write operation fails
        ValueError: If resume_from_sample is negative
    """
    if resume_from_sample < 0:
        raise ValueError("resume_from_sample must be non-negative")

    if verbose:
        sys.stdout.write(f"Resuming streaming save from sample {resume_from_sample}\n")

    # Skip samples until we reach the resume point
    samples_skipped = 0
    resumed_generator = param_generator

    if resume_from_sample > 0:

        def skip_generator() -> Generator[dict[str, Any], None, None]:
            nonlocal samples_skipped
            for params in param_generator:
                if samples_skipped < resume_from_sample:
                    samples_skipped += 1
                    continue
                yield params

        resumed_generator = skip_generator()

        if verbose:
            sys.stdout.write(f"Skipping first {resume_from_sample} samples\n")

    # Use the regular save function with the resumed generator
    try:
        save_augmentation_stream(
            resumed_generator,
            filepath,
            config=config,
            include_stats=include_stats,
            buffer_size=buffer_size,
            verbose=verbose,
        )
    except Exception as e:
        raise PartialWriteError(
            f"Failed to resume streaming save from sample {resume_from_sample}: {e}",
            samples_skipped,
        ) from e


class StreamingContext:
    """
    Context manager for safe streaming operations with automatic resource cleanup.

    This context manager ensures that generators and file handles are properly
    cleaned up even if exceptions occur during streaming operations.
    """

    def __init__(self, verbose: bool = False) -> None:
        """
        Initialize streaming context.

        Args:
            verbose: If True, print cleanup information
        """
        self.verbose = verbose
        self.generators: list[Generator[Any, None, None]] = []
        self.file_handles: list[tuple[Any, str]] = []  # (handle, filepath) pairs
        self.cleanup_errors: list[Exception] = []

    def register_generator(self, generator: Generator[Any, None, None]) -> None:
        """
        Register a generator for cleanup.

        Args:
            generator: Generator to register for cleanup
        """
        if generator is not None:
            self.generators.append(generator)

    def register_file_handle(self, file_handle: Any, filepath: str = "") -> None:
        """
        Register a file handle for cleanup.

        Args:
            file_handle: File handle to register for cleanup
            filepath: Optional filepath for error messages
        """
        if file_handle is not None:
            self.file_handles.append((file_handle, filepath))

    def __enter__(self) -> "StreamingContext":
        """Enter the streaming context."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exit the streaming context and cleanup resources.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        """
        # Clean up generators
        for generator in self.generators:
            try:
                safe_cleanup_generator(generator)
                if self.verbose:
                    sys.stdout.write("Cleaned up generator\n")
            except ResourceCleanupError as e:
                self.cleanup_errors.append(e)
                if self.verbose:
                    sys.stdout.write(f"Warning: {e}\n")

        # Clean up file handles
        for file_handle, filepath in self.file_handles:
            try:
                safe_close_file(file_handle, filepath)
                if self.verbose and filepath:
                    sys.stdout.write(f"Closed file handle for {filepath}\n")
            except ResourceCleanupError as e:
                self.cleanup_errors.append(e)
                if self.verbose:
                    sys.stdout.write(f"Warning: {e}\n")

        # If there were cleanup errors and no original exception, raise cleanup error
        if self.cleanup_errors and exc_type is None:
            raise ResourceCleanupError(
                f"Resource cleanup failed with {len(self.cleanup_errors)} errors: "
                f"{'; '.join(str(e) for e in self.cleanup_errors)}"
            )


def with_streaming_context(
    operation: Callable[..., Any], *args: Any, verbose: bool = False, **kwargs: Any
) -> Any:
    """
    Execute a streaming operation within a safe context.

    Args:
        operation: Function to execute
        *args: Positional arguments for the operation
        verbose: If True, print context information
        **kwargs: Keyword arguments for the operation

    Returns:
        Result of the operation

    Raises:
        Any exception from the operation or ResourceCleanupError
    """
    with StreamingContext(verbose=verbose) as context:
        # Add context to kwargs if the operation accepts it
        if "streaming_context" in operation.__code__.co_varnames:
            kwargs["streaming_context"] = context

        return operation(*args, **kwargs)


def stream_augmentation_chain_batched(
    num_samples: int,
    batch_config: "BatchConfig | None" = None,
    config: AugmentationConfig | None = None,
    start_id: int = 0,
    verbose: bool = False,
    progress_callback: Callable[[StreamingProgress, StreamingMetadata], None] | None = None,
) -> Generator[list[dict[str, Any]], None, None]:
    """
    Generate augmentation parameters as batched streams using advanced batching strategies.

    This function extends the basic streaming API with sophisticated batch processing
    capabilities, including memory-aware batching, different batching strategies,
    and dynamic batch size adjustment.

    Args:
        num_samples: Number of samples to generate
        batch_config: BatchConfig instance for batch processing (uses defaults if None)
        config: AugmentationConfig instance (uses defaults if None)
        start_id: Starting sample ID for custom ranges (default: 0)
        verbose: If True, print progress information
        progress_callback: Optional callback function for progress updates

    Yields:
        Lists of parameter dictionaries (batches)

    Raises:
        ValueError: If num_samples < 0 or start_id < 0
        ImportError: If batch processing module is not available
        BatchProcessingError: If batch processing fails

    Examples:
        Basic batched streaming with sequential strategy:
        >>> from src.batch import BatchConfig, BatchStrategy
        >>> batch_config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=32)
        >>> for batch in stream_augmentation_chain_batched(1000, batch_config):
        ...     print(f"Processing batch of {len(batch)} parameters")

        Memory-optimized batching:
        >>> batch_config = BatchConfig(
        ...     strategy=BatchStrategy.MEMORY_OPTIMIZED,
        ...     max_memory_mb=500
        ... )
        >>> for batch in stream_augmentation_chain_batched(10000, batch_config):
        ...     process_batch_efficiently(batch)

        Adaptive batching that adjusts based on performance:
        >>> batch_config = BatchConfig(strategy=BatchStrategy.ADAPTIVE)
        >>> for batch in stream_augmentation_chain_batched(50000, batch_config, verbose=True):
        ...     # Batch size will adapt based on processing performance
        ...     train_model_batch(batch)

        Integration with existing streaming API (backward compatibility):
        >>> # This maintains the same interface as stream_augmentation_chain
        >>> # but with batching capabilities
        >>> for batch in stream_augmentation_chain_batched(1000):
        ...     for params in batch:  # Process individual parameters if needed
        ...         process_single_sample(params)
    """
    if not _BATCH_AVAILABLE:
        raise ImportError(
            "Batch processing module not available. "
            "Please ensure the batch module is properly installed."
        )

    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if start_id < 0:
        raise ValueError("start_id must be non-negative")

    # Use default batch config if none provided
    if batch_config is None:
        batch_config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=32)

    if config is None:
        config = AugmentationConfig()

    # Create the base parameter generator
    param_generator = stream_augmentation_chain(
        num_samples=num_samples,
        config=config,
        start_id=start_id,
        chunk_size=None,  # We'll handle batching ourselves
        verbose=False,  # We'll handle verbose output
        progress_callback=None,  # We'll handle progress callbacks
    )

    # Create batch processor
    batch_processor = BatchProcessor(batch_config.strategy, batch_config)

    # Create progress tracking
    progress = create_streaming_progress(estimated_total=num_samples, start_id=start_id)
    metadata = create_streaming_metadata(
        total_samples=num_samples,
        config=config,
        start_id=start_id,
        chunk_size=batch_config.batch_size,
    )

    if verbose:
        sys.stdout.write(
            f"Starting batched stream generation: {num_samples} samples from ID {start_id} "
            f"using {batch_config.strategy.value} strategy\n"
        )

    try:
        batch_count = 0
        samples_processed = 0

        # Process the stream with memory monitoring if configured
        if batch_config.strategy.value in ["memory_optimized", "adaptive"]:
            batch_stream = batch_processor.process_with_memory_monitoring(param_generator)
        else:
            batch_stream = batch_processor.process_stream(param_generator)

        for batch in batch_stream:
            batch_count += 1
            samples_processed += len(batch)

            # Update progress
            update_streaming_progress(progress, samples_processed, start_id + samples_processed - 1)

            if verbose:
                if (
                    num_samples > 100
                    and batch_count % max(1, (num_samples // batch_config.batch_size or 32) // 20)
                    == 0
                ):
                    # Show progress for large datasets (every 5% of batches)
                    progress_info = format_progress_info(progress)
                    sys.stdout.write(
                        f"Batch progress: {progress_info} (batch #{batch_count}, size: {len(batch)})\n"
                    )
                elif num_samples <= 100:
                    # Show individual batches for small datasets
                    sys.stdout.write(f"Batch #{batch_count}: {len(batch)} samples\n")

            # Call progress callback if provided
            if progress_callback:
                progress_callback(progress, metadata)

            yield batch

        if verbose:
            final_progress = format_progress_info(progress)
            sys.stdout.write(
                f"Batched stream generation completed: {final_progress} "
                f"({batch_count} batches, avg size: {samples_processed / batch_count:.1f})\n"
            )

    except Exception as e:
        from .batch import BatchProcessingError

        raise BatchProcessingError(f"Batch processing failed: {e}") from e


def stream_augmentation_range_batched(
    start_id: int,
    end_id: int,
    batch_config: "BatchConfig | None" = None,
    config: AugmentationConfig | None = None,
) -> Generator[list[dict[str, Any]], None, None]:
    """
    Stream batched parameters for a specific range of sample IDs.

    This function provides batched streaming for specific ID ranges, useful for
    distributed processing or resuming operations with advanced batching strategies.

    Args:
        start_id: Starting sample ID (inclusive)
        end_id: Ending sample ID (exclusive)
        batch_config: BatchConfig instance for batch processing (uses defaults if None)
        config: AugmentationConfig instance (uses defaults if None)

    Yields:
        Lists of parameter dictionaries (batches)

    Raises:
        ValueError: If start_id < 0, end_id < start_id
        ImportError: If batch processing module is not available
        BatchProcessingError: If batch processing fails

    Examples:
        Process a specific range with memory-optimized batching:
        >>> from src.batch import BatchConfig, BatchStrategy
        >>> batch_config = BatchConfig(
        ...     strategy=BatchStrategy.MEMORY_OPTIMIZED,
        ...     max_memory_mb=1000
        ... )
        >>> for batch in stream_augmentation_range_batched(1000, 5000, batch_config):
        ...     process_batch(batch)

        Resume processing from a specific point with adaptive batching:
        >>> batch_config = BatchConfig(strategy=BatchStrategy.ADAPTIVE)
        >>> for batch in stream_augmentation_range_batched(5000, 10000, batch_config):
        ...     if should_stop():
        ...         break
        ...     process_batch(batch)
    """
    if start_id < 0:
        raise ValueError("start_id must be non-negative")
    if end_id < start_id:
        raise ValueError("end_id must be >= start_id")

    num_samples = end_id - start_id
    yield from stream_augmentation_chain_batched(
        num_samples=num_samples,
        batch_config=batch_config,
        config=config,
        start_id=start_id,
        verbose=False,
    )


def process_augmentation_batches(
    param_generator: Generator[dict[str, Any], None, None],
    batch_config: "BatchConfig | None" = None,
    processor_func: Callable[[list[dict[str, Any]]], Any] | None = None,
    verbose: bool = False,
) -> Generator[Any, None, None]:
    """
    Process augmentation parameters in batches using a custom processing function.

    This function provides a high-level interface for batch processing of augmentation
    parameters with custom processing logic and advanced batching strategies.

    Args:
        param_generator: Generator yielding individual augmentation parameters
        batch_config: BatchConfig instance for batch processing (uses defaults if None)
        processor_func: Function to process each batch (identity function if None)
        verbose: If True, print processing information

    Yields:
        Results from processor_func applied to each batch

    Raises:
        ImportError: If batch processing module is not available
        BatchProcessingError: If batch processing fails

    Examples:
        Custom batch processing with memory monitoring:
        >>> from src.batch import BatchConfig, BatchStrategy
        >>>
        >>> def custom_processor(batch):
        ...     # Custom processing logic
        ...     return [transform_params(params) for params in batch]
        >>>
        >>> generator = stream_augmentation_chain(10000)
        >>> batch_config = BatchConfig(
        ...     strategy=BatchStrategy.MEMORY_OPTIMIZED,
        ...     max_memory_mb=500
        ... )
        >>>
        >>> for processed_batch in process_augmentation_batches(
        ...     generator, batch_config, custom_processor, verbose=True
        ... ):
        ...     save_processed_batch(processed_batch)

        Simple batching without custom processing:
        >>> generator = stream_augmentation_chain(1000)
        >>> batch_config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=50)
        >>>
        >>> for batch in process_augmentation_batches(generator, batch_config):
        ...     # batch is a list of parameter dictionaries
        ...     train_model_on_batch(batch)

        Adaptive batching for performance optimization:
        >>> generator = stream_augmentation_chain(100000)
        >>> batch_config = BatchConfig(strategy=BatchStrategy.ADAPTIVE)
        >>>
        >>> for batch in process_augmentation_batches(generator, batch_config, verbose=True):
        ...     # Batch size will adapt based on processing performance
        ...     result = expensive_processing(batch)
        ...     store_results(result)
    """
    if not _BATCH_AVAILABLE:
        raise ImportError(
            "Batch processing module not available. "
            "Please ensure the batch module is properly installed."
        )

    # Use default batch config if none provided
    if batch_config is None:
        batch_config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=32)

    # Use identity function if no processor provided
    if processor_func is None:
        processor_func = lambda batch: batch

    # Create batch processor
    batch_processor = BatchProcessor(batch_config.strategy, batch_config)

    if verbose:
        sys.stdout.write(
            f"Starting batch processing using {batch_config.strategy.value} strategy\n"
        )

    try:
        batch_count = 0

        # Process the stream with memory monitoring if configured
        if batch_config.strategy.value in ["memory_optimized", "adaptive"]:
            batch_stream = batch_processor.process_with_memory_monitoring(param_generator)
        else:
            batch_stream = batch_processor.process_stream(param_generator)

        for batch in batch_stream:
            batch_count += 1

            if verbose and batch_count % 10 == 0:
                sys.stdout.write(
                    f"Processed {batch_count} batches (current batch size: {len(batch)})\n"
                )

            # Apply custom processing function
            result = processor_func(batch)
            yield result

        if verbose:
            sys.stdout.write(f"Batch processing completed: {batch_count} batches processed\n")

    except Exception as e:
        from .batch import BatchProcessingError

        raise BatchProcessingError(f"Batch processing failed: {e}") from e


def get_batch_metrics(
    param_generator: Generator[dict[str, Any], None, None],
    batch_config: "BatchConfig | None" = None,
) -> "BatchMetrics":
    """
    Collect metrics from batch processing operations.

    This function processes a parameter generator using the specified batch configuration
    and returns detailed metrics about the batching performance.

    Args:
        param_generator: Generator yielding augmentation parameters
        batch_config: BatchConfig instance for batch processing (uses defaults if None)

    Returns:
        BatchMetrics instance with processing statistics

    Raises:
        ImportError: If batch processing module is not available
        BatchProcessingError: If batch processing fails

    Examples:
        Analyze batching performance:
        >>> from src.batch import BatchConfig, BatchStrategy
        >>> generator = stream_augmentation_chain(10000)
        >>> batch_config = BatchConfig(strategy=BatchStrategy.ADAPTIVE)
        >>>
        >>> metrics = get_batch_metrics(generator, batch_config)
        >>> print(f"Average batch size: {metrics.avg_batch_size:.1f}")
        >>> print(f"Throughput: {metrics.throughput_samples_per_second:.1f} samples/sec")
        >>> print(f"Memory usage: {metrics.memory_usage_mb:.1f} MB")

        Compare different batching strategies:
        >>> strategies = [BatchStrategy.SEQUENTIAL, BatchStrategy.MEMORY_OPTIMIZED, BatchStrategy.ADAPTIVE]
        >>> for strategy in strategies:
        ...     generator = stream_augmentation_chain(5000)
        ...     config = BatchConfig(strategy=strategy)
        ...     metrics = get_batch_metrics(generator, config)
        ...     print(f"{strategy.value}: {metrics.throughput_samples_per_second:.1f} samples/sec")
    """
    if not _BATCH_AVAILABLE:
        raise ImportError(
            "Batch processing module not available. "
            "Please ensure the batch module is properly installed."
        )

    # Use default batch config if none provided
    if batch_config is None:
        batch_config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=32)
    import time

    # Create batch processor
    batch_processor = BatchProcessor(batch_config.strategy, batch_config)

    start_time = time.time()
    total_batches = 0
    total_samples = 0
    batch_sizes = []

    try:
        # Process the stream with memory monitoring if configured
        if batch_config.strategy.value in ["memory_optimized", "adaptive"]:
            batch_stream = batch_processor.process_with_memory_monitoring(param_generator)
        else:
            batch_stream = batch_processor.process_stream(param_generator)

        for batch in batch_stream:
            total_batches += 1
            batch_size = len(batch)
            total_samples += batch_size
            batch_sizes.append(batch_size)

        end_time = time.time()
        processing_time = end_time - start_time

        # Calculate metrics
        avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
        throughput = total_samples / processing_time if processing_time > 0 else 0

        # Get memory usage (approximate)
        memory_usage = 0
        if _PSUTIL_AVAILABLE:
            try:
                import psutil

                process = psutil.Process()
                memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            except (AttributeError, OSError):
                pass

        return BatchMetrics(
            total_batches=total_batches,
            avg_batch_size=avg_batch_size,
            memory_usage_mb=memory_usage,
            processing_time_seconds=processing_time,
            throughput_samples_per_second=throughput,
        )

    except Exception as e:
        from .batch import BatchProcessingError

        raise BatchProcessingError(f"Failed to collect batch metrics: {e}") from e


def stream_augmentation_chain_with_memory_limit(
    num_samples: int,
    max_memory_mb: int = 1000,
    config: AugmentationConfig | None = None,
    start_id: int = 0,
    verbose: bool = False,
    progress_callback: Callable[[StreamingProgress, StreamingMetadata], None] | None = None,
) -> Generator[list[dict[str, Any]], None, None]:
    """
    Stream augmentation parameters with automatic memory-aware batching.

    This convenience function automatically configures memory-optimized batching
    to stay within specified memory limits, providing a simple interface for
    memory-constrained environments.

    Args:
        num_samples: Number of samples to generate
        max_memory_mb: Maximum memory usage allowed in MB
        config: AugmentationConfig instance (uses defaults if None)
        start_id: Starting sample ID for custom ranges (default: 0)
        verbose: If True, print progress information
        progress_callback: Optional callback function for progress updates

    Yields:
        Lists of parameter dictionaries (batches) sized to fit memory constraints

    Raises:
        ValueError: If num_samples < 0, start_id < 0, or max_memory_mb <= 0
        ImportError: If batch processing module is not available

    Examples:
        Memory-constrained processing:
        >>> for batch in stream_augmentation_chain_with_memory_limit(10000, max_memory_mb=500):
        ...     process_batch_within_memory_limit(batch)

        Large dataset processing with memory monitoring:
        >>> for batch in stream_augmentation_chain_with_memory_limit(
        ...     1_000_000, max_memory_mb=2000, verbose=True
        ... ):
        ...     train_model_batch(batch)
    """
    if not _BATCH_AVAILABLE:
        raise ImportError(
            "Batch processing module not available. "
            "Please ensure the batch module is properly installed."
        )

    if max_memory_mb <= 0:
        raise ValueError("max_memory_mb must be positive")

    # Create memory-optimized batch configuration
    batch_config = BatchConfig(
        strategy=BatchStrategy.MEMORY_OPTIMIZED,
        max_memory_mb=max_memory_mb,
        min_batch_size=1,
        adaptive_sizing=True,
    )

    yield from stream_augmentation_chain(
        num_samples=num_samples,
        config=config,
        start_id=start_id,
        batch_config=batch_config,
        verbose=verbose,
        progress_callback=progress_callback,
    )


def estimate_optimal_batch_size(
    sample_size_bytes: int = 200, max_memory_mb: int = 1000, safety_margin: float = 0.1
) -> int:
    """
    Estimate optimal batch size based on memory constraints and sample size.

    This function provides a quick way to estimate appropriate batch sizes
    for memory-constrained environments without requiring the full batch
    processing module.

    Args:
        sample_size_bytes: Estimated size of a single parameter dictionary in bytes
        max_memory_mb: Maximum memory usage allowed in MB
        safety_margin: Safety margin as fraction (0.1 = 10% margin)

    Returns:
        Estimated optimal batch size

    Raises:
        ValueError: If parameters are invalid

    Examples:
        Estimate batch size for typical parameters:
        >>> batch_size = estimate_optimal_batch_size(sample_size_bytes=200, max_memory_mb=1000)
        >>> print(f"Recommended batch size: {batch_size}")

        Estimate for large parameter dictionaries:
        >>> batch_size = estimate_optimal_batch_size(sample_size_bytes=1000, max_memory_mb=500)
        >>> for batch in stream_augmentation_chain(10000, chunk_size=batch_size):
        ...     process_batch(batch)
    """
    if sample_size_bytes <= 0:
        raise ValueError("sample_size_bytes must be positive")
    if max_memory_mb <= 0:
        raise ValueError("max_memory_mb must be positive")
    if safety_margin < 0 or safety_margin >= 1:
        raise ValueError("safety_margin must be between 0 and 1")

    # Convert to bytes and apply safety margin
    available_bytes = max_memory_mb * 1024 * 1024 * (1 - safety_margin)

    # Calculate how many samples can fit
    optimal_size = int(available_bytes / sample_size_bytes)

    # Ensure minimum batch size of 1
    return max(1, optimal_size)


def get_memory_usage_estimate(
    num_samples: int,
    include_statistics: bool = True,
    bytes_per_param: int = 200,
) -> dict[str, str]:
    """
    Get human-readable memory usage estimates for augmentation operations.

    This function extends the existing estimate_memory_usage function with
    human-readable formatting and additional context for batch processing.

    Args:
        num_samples: Number of samples to process
        include_statistics: Whether statistics will be computed
        bytes_per_param: Estimated bytes per parameter dictionary

    Returns:
        Dictionary with formatted memory estimates

    Examples:
        Check memory requirements before processing:
        >>> estimates = get_memory_usage_estimate(100000)
        >>> print(f"Total memory needed: {estimates['total']}")
        >>> print(f"Parameters: {estimates['parameters']}")

        Plan batch processing:
        >>> estimates = get_memory_usage_estimate(1000000, include_statistics=False)
        >>> if estimates['total_mb'] > 1000:  # More than 1GB
        ...     use_streaming_with_batching()
        ... else:
        ...     use_standard_processing()
    """
    raw_estimates = estimate_memory_usage(num_samples, include_statistics, bytes_per_param)

    formatted = {}
    for key, bytes_value in raw_estimates.items():
        formatted[key] = format_memory_size(bytes_value)
        formatted[f"{key}_mb"] = bytes_value / (1024 * 1024)  # Also provide MB values

    # Add recommendations
    total_mb = raw_estimates["total"] / (1024 * 1024)
    if total_mb > 1000:  # > 1GB
        formatted["recommendation"] = "Use streaming with memory-optimized batching"
        formatted["suggested_batch_size"] = estimate_optimal_batch_size(bytes_per_param, 500)
    elif total_mb > 100:  # > 100MB
        formatted["recommendation"] = "Consider using chunked streaming"
        formatted["suggested_chunk_size"] = min(1000, num_samples // 10)
    else:
        formatted["recommendation"] = "Standard processing should work fine"

    return formatted


def generate_augmentation_chain(
    num_samples: int,
    config: AugmentationConfig | None = None,
    verbose: bool = False,
    save_path: str | None = None,
) -> list[dict[str, Any]]:
    """
    Generate augmentation parameters for multiple samples.

    Args:
        num_samples: Number of samples to augment
        config: AugmentationConfig instance (uses defaults if None)
        verbose: If True, print results to stdout
        save_path: If provided, save results to JSON file

    Returns:
        List of augmentation parameter dictionaries

    Raises:
        ValueError: If num_samples < 0
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")

    if config is None:
        config = AugmentationConfig()

    results = []
    output = []

    for sample_id in range(num_samples):
        params = gen_augmentation_params(sample_id, config)
        results.append(params)

        if verbose:
            output.append(
                f"Sample({sample_id}) -> rotation={params['rotation']:.2f}° "
                f"brightness={params['brightness']:.2f} "
                f"noise={params['noise']:.3f} "
                f"scale={params['scale']:.2f} "
                f"contrast={params['contrast']:.2f}\n"
            )

    if verbose:
        sys.stdout.write("".join(output))

    if save_path:
        save_augmentation_chain(results, save_path, config, include_stats=True)

    return results


def convert_stream_to_batches(
    param_generator: Generator[dict[str, Any], None, None],
    batch_strategy: str = "sequential",
    batch_size: int = 32,
    max_memory_mb: int = 1000,
) -> Generator[list[dict[str, Any]], None, None]:
    """
    Convert a parameter stream to batched output using specified strategy.

    This function provides a bridge between streaming and batch processing,
    allowing existing streaming generators to be processed with advanced
    batching strategies.

    Args:
        param_generator: Generator yielding individual augmentation parameters
        batch_strategy: Batching strategy ("sequential", "round_robin", "memory_optimized", "adaptive")
        batch_size: Target batch size (may be adjusted by strategy)
        max_memory_mb: Maximum memory usage for memory-optimized strategies

    Yields:
        Lists of parameter dictionaries (batches)

    Raises:
        ImportError: If batch processing module is not available
        ValueError: If batch_strategy is invalid

    Examples:
        Convert existing stream to memory-optimized batches:
        >>> generator = stream_augmentation_chain(10000)
        >>> for batch in convert_stream_to_batches(
        ...     generator, batch_strategy="memory_optimized", max_memory_mb=500
        ... ):
        ...     process_batch_efficiently(batch)

        Convert with adaptive batching:
        >>> generator = stream_augmentation_range(1000, 5000)
        >>> for batch in convert_stream_to_batches(
        ...     generator, batch_strategy="adaptive"
        ... ):
        ...     # Batch size will adapt based on processing performance
        ...     train_model_batch(batch)
    """
    if not _BATCH_AVAILABLE:
        raise ImportError(
            "Batch processing module not available. "
            "Please ensure the batch module is properly installed."
        )

    # Map string strategy to enum
    strategy_map = {
        "sequential": BatchStrategy.SEQUENTIAL,
        "round_robin": BatchStrategy.ROUND_ROBIN,
        "memory_optimized": BatchStrategy.MEMORY_OPTIMIZED,
        "adaptive": BatchStrategy.ADAPTIVE,
    }

    if batch_strategy not in strategy_map:
        raise ValueError(
            f"Invalid batch_strategy: {batch_strategy}. Must be one of {list(strategy_map.keys())}"
        )

    # Create batch configuration
    batch_config = BatchConfig(
        strategy=strategy_map[batch_strategy],
        batch_size=batch_size,
        max_memory_mb=max_memory_mb,
        min_batch_size=1,
        adaptive_sizing=True,
    )

    # Create batch processor and process the stream
    batch_processor = BatchProcessor(batch_config.strategy, batch_config)

    if batch_strategy in ["memory_optimized", "adaptive"]:
        yield from batch_processor.process_with_memory_monitoring(param_generator)
    else:
        yield from batch_processor.process_stream(param_generator)


def get_batch_processing_metrics(
    num_samples: int,
    config: AugmentationConfig | None = None,
    batch_strategies: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, "BatchMetrics"]:
    """
    Compare performance of different batch processing strategies.

    This function benchmarks various batch processing strategies to help
    choose the optimal approach for specific use cases and hardware configurations.

    Args:
        num_samples: Number of samples to use for benchmarking
        config: AugmentationConfig instance (uses defaults if None)
        batch_strategies: List of strategies to compare (uses all if None)
        verbose: If True, print progress information

    Returns:
        Dictionary mapping strategy names to BatchMetrics

    Raises:
        ImportError: If batch processing module is not available

    Examples:
        Compare all strategies:
        >>> metrics = get_batch_processing_metrics(10000, verbose=True)
        >>> for strategy, metric in metrics.items():
        ...     print(f"{strategy}: {metric.throughput_samples_per_second:.1f} samples/sec")

        Compare specific strategies:
        >>> metrics = get_batch_processing_metrics(
        ...     5000, batch_strategies=["sequential", "memory_optimized"]
        ... )
        >>> best_strategy = max(metrics.items(), key=lambda x: x[1].throughput_samples_per_second)
        >>> print(f"Best strategy: {best_strategy[0]}")
    """
    if not _BATCH_AVAILABLE:
        raise ImportError(
            "Batch processing module not available. "
            "Please ensure the batch module is properly installed."
        )

    if batch_strategies is None:
        batch_strategies = ["sequential", "round_robin", "memory_optimized", "adaptive"]

    results = {}

    for strategy in batch_strategies:
        if verbose:
            print(f"Benchmarking {strategy} strategy...")

        try:
            # Create a fresh generator for each strategy
            generator = stream_augmentation_chain(num_samples, config=config, verbose=False)

            # Convert to batches and collect metrics
            batch_generator = convert_stream_to_batches(
                generator, batch_strategy=strategy, batch_size=32, max_memory_mb=1000
            )

            # Use the existing get_batch_metrics function
            metrics = get_batch_metrics(batch_generator)
            results[strategy] = metrics

            if verbose:
                print(
                    f"  {strategy}: {metrics.throughput_samples_per_second:.1f} samples/sec, "
                    f"avg batch size: {metrics.avg_batch_size:.1f}"
                )

        except Exception as e:
            if verbose:
                print(f"  {strategy}: failed with error: {e}")
            # Continue with other strategies
            continue

    return results


def profile_augmentation_generation(
    func: Callable[..., Any] | None = None,
    *,
    enable_memory_tracking: bool = True,
    operation_name: str | None = None,
) -> Callable[..., Any]:
    """
    Decorator to profile augmentation parameter generation functions.

    This decorator adds performance profiling to augmentation functions,
    measuring timing and memory usage without affecting functionality.

    Args:
        func: Function to decorate (when used as @profile_augmentation_generation)
        enable_memory_tracking: If True, track memory usage during execution
        operation_name: Custom name for the operation (uses function name if None)

    Returns:
        Decorated function with profiling capabilities

    Raises:
        ImportError: If benchmark module is not available

    Examples:
        Decorate a function for profiling:
        >>> @profile_augmentation_generation
        ... def my_custom_generator(num_samples):
        ...     return generate_augmentation_chain(num_samples)

        Use with custom operation name:
        >>> @profile_augmentation_generation(operation_name="custom_batch_processing")
        ... def process_batch(batch_size):
        ...     # Custom processing logic
        ...     pass

        Profile existing functions:
        >>> profiled_gen = profile_augmentation_generation(generate_augmentation_chain)
        >>> results = profiled_gen(1000)
    """
    if not _BENCHMARK_AVAILABLE:
        raise ImportError(
            "Benchmark module not available. "
            "Please ensure the benchmark module is properly installed."
        )

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        profiler = PerformanceProfiler(enable_memory_tracking=enable_memory_tracking)
        actual_operation_name = operation_name or f.__name__

        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profiler.start_profiling(actual_operation_name)
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                profile_result = profiler.end_profiling(actual_operation_name)
                # Store profile result for later retrieval
                if not hasattr(wrapper, "_profile_results"):
                    wrapper._profile_results = []
                wrapper._profile_results.append(profile_result)

        # Add method to get profile results
        def get_profile_results():
            return getattr(wrapper, "_profile_results", [])

        wrapper.get_profile_results = get_profile_results
        wrapper._profiler = profiler
        return wrapper

    # Support both @profile_augmentation_generation and @profile_augmentation_generation()
    if func is None:
        return decorator
    else:
        return decorator(func)


def benchmark_augmentation_performance(
    num_samples: int,
    config: AugmentationConfig | None = None,
    iterations: int = 10,
    warmup_iterations: int = 2,
    include_streaming: bool = True,
    include_batch_processing: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Comprehensive benchmarking of augmentation parameter generation performance.

    This function benchmarks various aspects of the augmentation system including
    parameter generation, streaming, and batch processing to identify performance
    characteristics and potential optimizations.

    Args:
        num_samples: Number of samples to use for benchmarking
        config: AugmentationConfig instance (uses defaults if None)
        iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations to exclude from results
        include_streaming: If True, benchmark streaming operations
        include_batch_processing: If True, benchmark batch processing
        verbose: If True, print progress information

    Returns:
        Dictionary with comprehensive benchmark results

    Raises:
        ImportError: If benchmark module is not available

    Examples:
        Basic performance benchmarking:
        >>> results = benchmark_augmentation_performance(1000, verbose=True)
        >>> print(f"Generation throughput: {results['generation']['throughput_samples_per_second']:.1f}")

        Detailed benchmarking with custom config:
        >>> config = AugmentationConfig(rotation_range=(-90, 90))
        >>> results = benchmark_augmentation_performance(
        ...     5000, config=config, iterations=20, include_batch_processing=True
        ... )
        >>> print(f"Best batch strategy: {results['batch_processing']['best_strategy']}")

        Quick performance check:
        >>> results = benchmark_augmentation_performance(
        ...     100, iterations=3, include_streaming=False, include_batch_processing=False
        ... )
        >>> print(f"Basic generation: {results['generation']['avg_time_ms']:.2f}ms per sample")
    """
    if not _BENCHMARK_AVAILABLE:
        raise ImportError(
            "Benchmark module not available. "
            "Please ensure the benchmark module is properly installed."
        )

    if config is None:
        config = AugmentationConfig()

    results = {}

    # Benchmark basic parameter generation
    if verbose:
        print("Benchmarking parameter generation...")

    def generation_benchmark():
        return generate_augmentation_chain(num_samples, config=config, verbose=False)

    gen_result = benchmark_function(generation_benchmark, iterations=iterations)
    results["generation"] = {
        "avg_time_seconds": gen_result.avg_time_seconds,
        "avg_time_ms": gen_result.avg_time_ms,
        "throughput_samples_per_second": num_samples / gen_result.avg_time_seconds,
        "min_time_ms": gen_result.min_time_ms,
        "max_time_ms": gen_result.max_time_ms,
        "std_dev_ms": gen_result.std_dev_seconds * 1000,
    }

    # Benchmark streaming operations
    if include_streaming:
        if verbose:
            print("Benchmarking streaming operations...")

        def streaming_benchmark():
            return list(stream_augmentation_chain(num_samples, config=config, verbose=False))

        stream_result = benchmark_function(streaming_benchmark, iterations=iterations)
        results["streaming"] = {
            "avg_time_seconds": stream_result.avg_time_seconds,
            "avg_time_ms": stream_result.avg_time_ms,
            "throughput_samples_per_second": num_samples / stream_result.avg_time_seconds,
            "overhead_vs_generation_percent": (
                (stream_result.avg_time_seconds - gen_result.avg_time_seconds)
                / gen_result.avg_time_seconds
            )
            * 100,
        }

    # Benchmark batch processing
    if include_batch_processing and _BATCH_AVAILABLE:
        if verbose:
            print("Benchmarking batch processing strategies...")

        batch_results = {}
        strategies = ["sequential", "memory_optimized"]

        for strategy in strategies:
            try:

                def batch_benchmark():
                    generator = stream_augmentation_chain(num_samples, config=config, verbose=False)
                    return list(convert_stream_to_batches(generator, batch_strategy=strategy))

                batch_result = benchmark_function(
                    batch_benchmark, iterations=max(1, iterations // 2)
                )
                batch_results[strategy] = {
                    "avg_time_seconds": batch_result.avg_time_seconds,
                    "throughput_samples_per_second": num_samples / batch_result.avg_time_seconds,
                }

                if verbose:
                    print(
                        f"  {strategy}: {batch_results[strategy]['throughput_samples_per_second']:.1f} samples/sec"
                    )

            except Exception as e:
                if verbose:
                    print(f"  {strategy}: failed with error: {e}")
                continue

        if batch_results:
            # Find best strategy
            best_strategy = max(
                batch_results.items(), key=lambda x: x[1]["throughput_samples_per_second"]
            )
            results["batch_processing"] = {
                "strategies": batch_results,
                "best_strategy": best_strategy[0],
                "best_throughput": best_strategy[1]["throughput_samples_per_second"],
            }

    # Add system information
    results["system_info"] = {
        "num_samples": num_samples,
        "iterations": iterations,
        "config": asdict(config),
    }

    # Add memory usage if available
    if _PSUTIL_AVAILABLE:
        memory_info = get_current_memory_usage()
        if memory_info:
            results["system_info"]["memory_usage_mb"] = memory_info.get("rss", 0) / (1024 * 1024)

    if verbose:
        print("Benchmarking completed!")
        print(
            f"Generation: {results['generation']['throughput_samples_per_second']:.1f} samples/sec"
        )
        if "streaming" in results:
            print(
                f"Streaming: {results['streaming']['throughput_samples_per_second']:.1f} samples/sec"
            )
        if "batch_processing" in results:
            print(
                f"Best batch strategy: {results['batch_processing']['best_strategy']} "
                f"({results['batch_processing']['best_throughput']:.1f} samples/sec)"
            )

    return results


def create_performance_profiler(
    enable_memory_tracking: bool = True, auto_report: bool = False
) -> "PerformanceProfiler":
    """
    Create a performance profiler instance for manual profiling.

    This function provides access to the underlying performance profiler
    for custom profiling scenarios and advanced performance analysis.

    Args:
        enable_memory_tracking: If True, track memory usage during operations
        auto_report: If True, automatically print reports after profiling

    Returns:
        PerformanceProfiler instance

    Raises:
        ImportError: If benchmark module is not available

    Examples:
        Manual profiling of custom operations:
        >>> profiler = create_performance_profiler()
        >>> profiler.start_profiling("custom_operation")
        >>> # ... custom code ...
        >>> result = profiler.end_profiling("custom_operation")
        >>> print(f"Operation took {result.avg_time_per_call_ms:.2f}ms")

        Context-based profiling:
        >>> profiler = create_performance_profiler(enable_memory_tracking=True)
        >>> with measure_time() as timer:
        ...     params = generate_augmentation_chain(1000)
        >>> print(f"Generation took {timer['elapsed']:.3f} seconds")
    """
    if not _BENCHMARK_AVAILABLE:
        raise ImportError(
            "Benchmark module not available. "
            "Please ensure the benchmark module is properly installed."
        )

    return PerformanceProfiler(enable_memory_tracking=enable_memory_tracking)


def get_performance_recommendations(
    benchmark_results: dict[str, Any],
    target_throughput: float | None = None,
    memory_limit_mb: int | None = None,
) -> list[str]:
    """
    Generate performance optimization recommendations based on benchmark results.

    This function analyzes benchmark results and provides actionable recommendations
    for optimizing augmentation pipeline performance based on specific constraints.

    Args:
        benchmark_results: Results from benchmark_augmentation_performance()
        target_throughput: Target throughput in samples per second (optional)
        memory_limit_mb: Memory limit in MB (optional)

    Returns:
        List of recommendation strings

    Examples:
        Get recommendations from benchmark results:
        >>> results = benchmark_augmentation_performance(10000)
        >>> recommendations = get_performance_recommendations(results, target_throughput=1000)
        >>> for rec in recommendations:
        ...     print(f"- {rec}")

        Memory-constrained recommendations:
        >>> recommendations = get_performance_recommendations(
        ...     results, memory_limit_mb=500
        ... )
        >>> print("\\n".join(recommendations))
    """
    recommendations = []

    if "generation" not in benchmark_results:
        recommendations.append("Run benchmark_augmentation_performance() to get baseline metrics")
        return recommendations

    gen_throughput = benchmark_results["generation"]["throughput_samples_per_second"]

    # Throughput recommendations
    if target_throughput and gen_throughput < target_throughput:
        shortfall = target_throughput - gen_throughput
        recommendations.append(
            f"Current throughput ({gen_throughput:.1f} samples/sec) is below target "
            f"({target_throughput:.1f} samples/sec) by {shortfall:.1f} samples/sec"
        )

        if "batch_processing" in benchmark_results:
            best_batch = benchmark_results["batch_processing"]["best_throughput"]
            if best_batch > gen_throughput:
                improvement = ((best_batch - gen_throughput) / gen_throughput) * 100
                recommendations.append(
                    f"Use {benchmark_results['batch_processing']['best_strategy']} batch processing "
                    f"for {improvement:.1f}% performance improvement"
                )

        recommendations.append("Consider using streaming with chunking for large datasets")
        recommendations.append("Profile individual augmentation operations to identify bottlenecks")

    # Memory recommendations
    if memory_limit_mb:
        system_memory = benchmark_results.get("system_info", {}).get("memory_usage_mb", 0)
        if system_memory > memory_limit_mb:
            recommendations.append(
                f"Current memory usage ({system_memory:.1f}MB) exceeds limit ({memory_limit_mb}MB)"
            )
            recommendations.append(
                "Use stream_augmentation_chain_with_memory_limit() for memory-constrained processing"
            )
            recommendations.append("Consider memory-optimized batch processing strategy")

    # General optimization recommendations
    if "streaming" in benchmark_results:
        streaming_overhead = benchmark_results["streaming"].get("overhead_vs_generation_percent", 0)
        if streaming_overhead > 10:  # More than 10% overhead
            recommendations.append(
                f"Streaming has {streaming_overhead:.1f}% overhead - consider batch processing for better performance"
            )
        elif streaming_overhead < 5:  # Low overhead
            recommendations.append(
                "Streaming has low overhead - good for memory-efficient processing"
            )

    # Configuration recommendations
    config = benchmark_results.get("system_info", {}).get("config", {})
    if config.get("augmentation_depth", 10) > 15:
        recommendations.append(
            "Consider reducing augmentation_depth for faster parameter generation"
        )

    if not recommendations:
        recommendations.append("Performance looks good! No specific optimizations needed.")

    return recommendations
