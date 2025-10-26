"""
Batch processing module for DPA library.

This module provides advanced batch processing strategies with memory awareness
and dynamic sizing capabilities for augmentation parameter generation.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import psutil

# Note: AugmentationConfig import removed to avoid circular imports
# It will be imported dynamically when needed


class BatchStrategy(Enum):
    """Enumeration of available batch processing strategies."""

    SEQUENTIAL = "sequential"
    ROUND_ROBIN = "round_robin"
    MEMORY_OPTIMIZED = "memory_optimized"
    ADAPTIVE = "adaptive"


# Exception Classes
class BatchProcessingError(Exception):
    """Base exception for batch processing operations."""

    pass


class MemoryLimitExceededError(BatchProcessingError):
    """Raised when memory usage exceeds configured limits."""

    pass


class InvalidBatchConfigError(BatchProcessingError):
    """Raised when batch configuration is invalid."""

    pass


# Data Models
@dataclass
class BatchConfig:
    """Configuration for batch processing operations."""

    strategy: BatchStrategy
    batch_size: int | None = None
    max_memory_mb: int = 1000
    min_batch_size: int = 1
    adaptive_sizing: bool = True

    def __post_init__(self):
        """Validate batch configuration after initialization."""
        if self.batch_size is not None and self.batch_size < self.min_batch_size:
            raise InvalidBatchConfigError(
                f"batch_size ({self.batch_size}) cannot be less than min_batch_size ({self.min_batch_size})"
            )

        if self.max_memory_mb <= 0:
            raise InvalidBatchConfigError("max_memory_mb must be positive")

        if self.min_batch_size <= 0:
            raise InvalidBatchConfigError("min_batch_size must be positive")


@dataclass
class BatchMetrics:
    """Metrics collected during batch processing operations."""

    total_batches: int
    avg_batch_size: float
    memory_usage_mb: float
    processing_time_seconds: float
    throughput_samples_per_second: float

    @property
    def total_samples(self) -> int:
        """Calculate total samples processed."""
        return int(self.total_batches * self.avg_batch_size)


@dataclass
class MemoryInfo:
    """Memory usage information."""

    available_mb: int
    used_mb: int
    total_mb: int
    percent_used: float

    @classmethod
    def current(cls) -> MemoryInfo:
        """Get current memory information."""
        memory = psutil.virtual_memory()
        return cls(
            available_mb=int(memory.available / (1024 * 1024)),
            used_mb=int(memory.used / (1024 * 1024)),
            total_mb=int(memory.total / (1024 * 1024)),
            percent_used=memory.percent,
        )


# Placeholder classes for future implementation
class MemoryAwareBatcher:
    """Memory-aware batch size calculator and monitor."""

    def __init__(self, max_memory_mb: int = 1000, min_batch_size: int = 1):
        """Initialize memory-aware batcher.

        Args:
            max_memory_mb: Maximum memory usage allowed in MB
            min_batch_size: Minimum batch size to maintain
        """
        self.max_memory_mb = max_memory_mb
        self.min_batch_size = min_batch_size
        self._safety_margin = 0.1  # 10% safety margin
        self._adjustment_factor = 0.8  # Reduce batch size by 20% when memory is high

    def calculate_optimal_batch_size(self, sample_size_bytes: int) -> int:
        """Calculate optimal batch size based on memory constraints.

        Args:
            sample_size_bytes: Estimated size of a single sample in bytes

        Returns:
            Optimal batch size
        """
        if sample_size_bytes <= 0:
            return self.min_batch_size

        # Get current memory info
        memory_info = MemoryInfo.current()

        # Calculate available memory with safety margin
        available_memory_bytes = (
            (memory_info.available_mb * (1 - self._safety_margin)) * 1024 * 1024
        )

        # Calculate how many samples can fit in available memory
        max_samples = int(available_memory_bytes / sample_size_bytes)

        # Ensure we don't exceed our configured maximum memory limit
        max_memory_bytes = self.max_memory_mb * 1024 * 1024
        max_samples_by_limit = int(max_memory_bytes / sample_size_bytes)

        # Take the minimum of the two constraints
        optimal_size = min(max_samples, max_samples_by_limit)

        # Ensure we meet minimum batch size requirement
        return max(self.min_batch_size, optimal_size)

    def monitor_memory_usage(self) -> dict[str, int]:
        """Monitor current memory usage.

        Returns:
            Dictionary with memory usage information
        """
        memory_info = MemoryInfo.current()
        return {
            "available_mb": memory_info.available_mb,
            "used_mb": memory_info.used_mb,
            "total_mb": memory_info.total_mb,
            "percent_used": int(memory_info.percent_used),
        }

    def adjust_batch_size(self, current_size: int, memory_usage: dict[str, int]) -> int:
        """Adjust batch size based on current memory usage.

        Args:
            current_size: Current batch size
            memory_usage: Current memory usage information

        Returns:
            Adjusted batch size
        """
        percent_used = memory_usage.get("percent_used", 0)
        available_mb = memory_usage.get("available_mb", 0)

        # If memory usage is high (>80%), reduce batch size
        if percent_used > 80:
            new_size = int(current_size * self._adjustment_factor)
            return max(self.min_batch_size, new_size)

        # If available memory is less than our max limit, reduce batch size
        elif available_mb < self.max_memory_mb:
            reduction_factor = available_mb / self.max_memory_mb
            new_size = int(current_size * reduction_factor)
            return max(self.min_batch_size, new_size)

        # If memory usage is low (<50%) and we have room, increase batch size slightly
        elif percent_used < 50 and available_mb > self.max_memory_mb * 2:
            new_size = int(current_size * 1.2)  # Increase by 20%
            return new_size

        # Otherwise, keep current size
        return current_size


class BatchProcessor:
    """Main batch processor that orchestrates batch processing operations."""

    def __init__(self, strategy: BatchStrategy, config: BatchConfig):
        """Initialize batch processor.

        Args:
            strategy: Batch processing strategy to use
            config: Batch processing configuration
        """
        self.strategy = strategy
        self.config = config
        self._memory_batcher = MemoryAwareBatcher(
            max_memory_mb=config.max_memory_mb, min_batch_size=config.min_batch_size
        )
        self._performance_history = []  # For adaptive strategy
        self._current_batch_size = config.batch_size or 32

    def _sequential_batching(self, param_generator: Generator) -> Generator[list[dict], None, None]:
        """Sequential batching strategy - simple consecutive grouping.

        Args:
            param_generator: Generator yielding augmentation parameters

        Yields:
            Lists of batched parameters
        """
        batch = []
        batch_size = self._current_batch_size

        for params in param_generator:
            batch.append(params)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield remaining items
        if batch:
            yield batch

    def _round_robin_batching(
        self, param_generator: Generator
    ) -> Generator[list[dict], None, None]:
        """Round-robin batching strategy - distribute items across multiple batches.

        Args:
            param_generator: Generator yielding augmentation parameters

        Yields:
            Lists of batched parameters
        """
        batch_size = self._current_batch_size
        batches = [[] for _ in range(batch_size)]
        batch_index = 0
        items_collected = 0

        for params in param_generator:
            batches[batch_index % batch_size].append(params)
            batch_index += 1
            items_collected += 1

            # When we have filled all batches with at least one item each
            if items_collected >= batch_size and all(len(batch) > 0 for batch in batches):
                # Yield the batch with the most items
                max_batch = max(batches, key=len)
                if max_batch:
                    yield max_batch
                    # Remove yielded items and reset
                    batches = [[] for _ in range(batch_size)]
                    batch_index = 0
                    items_collected = 0

        # Yield remaining non-empty batches
        for batch in batches:
            if batch:
                yield batch

    def _memory_optimized_batching(
        self, param_generator: Generator
    ) -> Generator[list[dict], None, None]:
        """Memory-optimized batching strategy - adjust batch size based on memory.

        Args:
            param_generator: Generator yielding augmentation parameters

        Yields:
            Lists of batched parameters
        """
        batch = []

        for params in param_generator:
            batch.append(params)

            # Check memory usage periodically
            if len(batch) % 10 == 0:  # Check every 10 items
                memory_usage = self._memory_batcher.monitor_memory_usage()
                optimal_size = self._memory_batcher.adjust_batch_size(len(batch), memory_usage)

                # If we've reached optimal size or memory is getting tight
                if len(batch) >= optimal_size or memory_usage.get("percent_used", 0) > 75:
                    yield batch
                    batch = []

            # Safety check - don't let batches grow too large
            elif len(batch) >= self._current_batch_size * 2:
                yield batch
                batch = []

        # Yield remaining items
        if batch:
            yield batch

    def _adaptive_batching(self, param_generator: Generator) -> Generator[list[dict], None, None]:
        """Adaptive batching strategy - adjust based on performance history.

        Args:
            param_generator: Generator yielding augmentation parameters

        Yields:
            Lists of batched parameters
        """
        import time

        batch = []

        for params in param_generator:
            start_time = time.time()
            batch.append(params)

            # Determine current optimal batch size based on performance history
            if self._performance_history:
                # Use average of recent performance to adjust batch size
                recent_performance = self._performance_history[-5:]  # Last 5 batches
                avg_time_per_item = sum(p["time_per_item"] for p in recent_performance) / len(
                    recent_performance
                )

                # If processing is slow, reduce batch size
                if avg_time_per_item > 0.1:  # More than 100ms per item
                    target_size = max(self.config.min_batch_size, self._current_batch_size // 2)
                # If processing is fast, increase batch size
                elif avg_time_per_item < 0.01:  # Less than 10ms per item
                    target_size = min(self._current_batch_size * 2, 1000)  # Cap at 1000
                else:
                    target_size = self._current_batch_size
            else:
                target_size = self._current_batch_size

            # Also consider memory constraints
            memory_usage = self._memory_batcher.monitor_memory_usage()
            memory_adjusted_size = self._memory_batcher.adjust_batch_size(target_size, memory_usage)

            final_batch_size = min(target_size, memory_adjusted_size)

            if len(batch) >= final_batch_size:
                end_time = time.time()
                processing_time = end_time - start_time

                # Record performance metrics
                self._performance_history.append(
                    {
                        "batch_size": len(batch),
                        "processing_time": processing_time,
                        "time_per_item": processing_time / len(batch) if batch else 0,
                    }
                )

                # Keep only recent history
                if len(self._performance_history) > 20:
                    self._performance_history = self._performance_history[-20:]

                yield batch
                batch = []

        # Yield remaining items
        if batch:
            yield batch

    def process_stream(self, param_generator: Generator) -> Generator[list[dict], None, None]:
        """Process parameter stream into batches using the configured strategy.

        Args:
            param_generator: Generator yielding augmentation parameters

        Yields:
            Lists of batched parameters
        """
        if self.strategy == BatchStrategy.SEQUENTIAL:
            yield from self._sequential_batching(param_generator)
        elif self.strategy == BatchStrategy.ROUND_ROBIN:
            yield from self._round_robin_batching(param_generator)
        elif self.strategy == BatchStrategy.MEMORY_OPTIMIZED:
            yield from self._memory_optimized_batching(param_generator)
        elif self.strategy == BatchStrategy.ADAPTIVE:
            yield from self._adaptive_batching(param_generator)
        else:
            raise BatchProcessingError(f"Unknown batch strategy: {self.strategy}")

    def process_with_memory_monitoring(
        self, param_generator: Generator
    ) -> Generator[list[dict], None, None]:
        """Process parameter stream with memory monitoring.

        Args:
            param_generator: Generator yielding augmentation parameters

        Yields:
            Lists of batched parameters with memory monitoring
        """
        for batch in self.process_stream(param_generator):
            # Monitor memory before yielding batch
            memory_usage = self._memory_batcher.monitor_memory_usage()

            # Check if we're approaching memory limits
            if memory_usage.get("percent_used", 0) > 90:
                raise MemoryLimitExceededError(
                    f"Memory usage ({memory_usage['percent_used']}%) is critically high"
                )

            # Adjust future batch sizes if memory is getting tight
            if memory_usage.get("percent_used", 0) > 80:
                self._current_batch_size = self._memory_batcher.adjust_batch_size(
                    self._current_batch_size, memory_usage
                )

            yield batch


# Utility functions
def get_memory_usage() -> MemoryInfo:
    """Get current system memory usage information.

    Returns:
        Current memory information
    """
    return MemoryInfo.current()


@contextmanager
def memory_monitor(max_memory_mb: int):
    """Context manager for monitoring memory usage during operations.

    Args:
        max_memory_mb: Maximum allowed memory usage in MB

    Raises:
        MemoryLimitExceededError: If memory usage exceeds limit
    """
    initial_memory = get_memory_usage()
    try:
        yield initial_memory
    finally:
        final_memory = get_memory_usage()
        if final_memory.used_mb > max_memory_mb:
            raise MemoryLimitExceededError(
                f"Memory usage ({final_memory.used_mb}MB) exceeded limit ({max_memory_mb}MB)"
            )
