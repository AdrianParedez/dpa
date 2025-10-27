"""
Unit tests for batch processing module.

Tests memory monitoring, different batching strategies, and dynamic batch size adjustment.
"""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.batch import (
    BatchConfig,
    BatchMetrics,
    BatchProcessingError,
    BatchProcessor,
    BatchStrategy,
    InvalidBatchConfigError,
    MemoryAwareBatcher,
    MemoryInfo,
    MemoryLimitExceededError,
    get_memory_usage,
    memory_monitor,
)


class TestBatchConfig:
    """Test BatchConfig data model."""

    def test_valid_config(self):
        """Test valid batch configuration."""
        config = BatchConfig(
            strategy=BatchStrategy.SEQUENTIAL, batch_size=32, max_memory_mb=1000, min_batch_size=1
        )
        assert config.strategy == BatchStrategy.SEQUENTIAL
        assert config.batch_size == 32
        assert config.max_memory_mb == 1000
        assert config.min_batch_size == 1

    def test_invalid_batch_size_zero(self):
        """Test invalid batch size configuration with zero value."""
        with pytest.raises(InvalidBatchConfigError, match="batch_size \\(0\\) must be positive"):
            BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=0, min_batch_size=1)

    def test_invalid_batch_size_negative(self):
        """Test invalid batch size configuration with negative value."""
        with pytest.raises(InvalidBatchConfigError, match="batch_size \\(-5\\) must be positive"):
            BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=-5, min_batch_size=1)

    def test_batch_size_less_than_min(self):
        """Test batch size less than min_batch_size."""
        with pytest.raises(
            InvalidBatchConfigError,
            match="batch_size \\(2\\) cannot be less than min_batch_size \\(5\\)",
        ):
            BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=2, min_batch_size=5)

    def test_batch_size_none_valid(self):
        """Test that batch_size=None is valid."""
        config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=None, min_batch_size=1)
        assert config.batch_size is None

    def test_invalid_memory_limit(self):
        """Test invalid memory limit configuration."""
        with pytest.raises(InvalidBatchConfigError):
            BatchConfig(strategy=BatchStrategy.SEQUENTIAL, max_memory_mb=-100)


class TestMemoryInfo:
    """Test MemoryInfo data model."""

    @patch("psutil.virtual_memory")
    def test_current_memory_info(self, mock_memory):
        """Test current memory information retrieval."""
        # Mock psutil memory object
        mock_memory.return_value = MagicMock(
            available=1024 * 1024 * 1024,  # 1GB
            used=512 * 1024 * 1024,  # 512MB
            total=2048 * 1024 * 1024,  # 2GB
            percent=25.0,
        )

        memory_info = MemoryInfo.current()
        assert memory_info.available_mb == 1024
        assert memory_info.used_mb == 512
        assert memory_info.total_mb == 2048
        assert memory_info.percent_used == 25.0


class TestMemoryAwareBatcher:
    """Test MemoryAwareBatcher class."""

    def test_initialization(self):
        """Test MemoryAwareBatcher initialization."""
        batcher = MemoryAwareBatcher(max_memory_mb=500, min_batch_size=2)
        assert batcher.max_memory_mb == 500
        assert batcher.min_batch_size == 2

    @patch("src.batch.MemoryInfo.current")
    def test_calculate_optimal_batch_size(self, mock_memory):
        """Test optimal batch size calculation."""
        # Mock memory with 1GB available
        mock_memory.return_value = MemoryInfo(
            available_mb=1024, used_mb=512, total_mb=2048, percent_used=25.0
        )

        batcher = MemoryAwareBatcher(max_memory_mb=500, min_batch_size=1)

        # Test with 1MB per sample
        sample_size = 1024 * 1024  # 1MB
        batch_size = batcher.calculate_optimal_batch_size(sample_size)

        # Should be limited by max_memory_mb (500MB)
        assert batch_size == 500

    def test_calculate_optimal_batch_size_zero_sample(self):
        """Test optimal batch size with zero sample size."""
        batcher = MemoryAwareBatcher(min_batch_size=5)
        batch_size = batcher.calculate_optimal_batch_size(0)
        assert batch_size == 5

    def test_calculate_optimal_batch_size_negative_sample(self):
        """Test optimal batch size with negative sample size."""
        batcher = MemoryAwareBatcher(min_batch_size=3)
        batch_size = batcher.calculate_optimal_batch_size(-100)
        assert batch_size == 3

    def test_calculate_optimal_batch_size_division_by_zero_protection(self):
        """Test division-by-zero protection in calculate_optimal_batch_size."""
        batcher = MemoryAwareBatcher(min_batch_size=10)

        # Test with zero sample size
        batch_size = batcher.calculate_optimal_batch_size(0)
        assert batch_size == 10

        # Test with very small sample size that could cause issues
        batch_size = batcher.calculate_optimal_batch_size(1)  # 1 byte
        assert batch_size >= 10  # Should at least return min_batch_size

    @patch("src.batch.MemoryInfo.current")
    def test_monitor_memory_usage(self, mock_memory):
        """Test memory usage monitoring."""
        mock_memory.return_value = MemoryInfo(
            available_mb=1024, used_mb=512, total_mb=2048, percent_used=25.0
        )

        batcher = MemoryAwareBatcher()
        usage = batcher.monitor_memory_usage()

        assert usage["available_mb"] == 1024
        assert usage["used_mb"] == 512
        assert usage["total_mb"] == 2048
        assert usage["percent_used"] == 25

    def test_adjust_batch_size_high_memory(self):
        """Test batch size adjustment with high memory usage."""
        batcher = MemoryAwareBatcher(min_batch_size=2)

        # High memory usage should reduce batch size
        memory_usage = {"percent_used": 85, "available_mb": 100}
        adjusted_size = batcher.adjust_batch_size(100, memory_usage)

        # Should be reduced by adjustment factor (0.8)
        assert adjusted_size == 80

    def test_adjust_batch_size_low_memory(self):
        """Test batch size adjustment with low memory usage."""
        batcher = MemoryAwareBatcher(max_memory_mb=500)

        # Low memory usage should increase batch size
        memory_usage = {"percent_used": 30, "available_mb": 1200}
        adjusted_size = batcher.adjust_batch_size(100, memory_usage)

        # Should be increased by 20%
        assert adjusted_size == 120

    def test_adjust_batch_size_minimum_constraint(self):
        """Test batch size adjustment respects minimum constraint."""
        batcher = MemoryAwareBatcher(min_batch_size=10, max_memory_mb=100)

        # Very high memory usage with small current batch size
        memory_usage = {"percent_used": 95, "available_mb": 50}
        adjusted_size = batcher.adjust_batch_size(12, memory_usage)

        # 12 * 0.8 = 9.6 -> 9, but min is 10, so should be 10
        assert adjusted_size == 10


class TestBatchProcessor:
    """Test BatchProcessor class."""

    def create_test_generator(self, count: int) -> Generator[dict, None, None]:
        """Create a test parameter generator."""
        for i in range(count):
            yield {"sample_id": i, "param": f"value_{i}"}

    def test_sequential_batching(self):
        """Test sequential batching strategy."""
        config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=3)
        processor = BatchProcessor(BatchStrategy.SEQUENTIAL, config)

        # Generate 10 items, expect batches of size 3, 3, 3, 1
        batches = list(processor.process_stream(self.create_test_generator(10)))

        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1

    def test_round_robin_batching(self):
        """Test round-robin batching strategy."""
        config = BatchConfig(strategy=BatchStrategy.ROUND_ROBIN, batch_size=3)
        processor = BatchProcessor(BatchStrategy.ROUND_ROBIN, config)

        # Generate items and test round-robin distribution
        batches = list(processor.process_stream(self.create_test_generator(6)))

        # Should have at least one batch
        assert len(batches) >= 1
        # All batches should have items
        for batch in batches:
            assert len(batch) > 0

    @patch("src.batch.MemoryInfo.current")
    def test_memory_optimized_batching(self, mock_memory):
        """Test memory-optimized batching strategy."""
        # Mock memory with moderate usage
        mock_memory.return_value = MemoryInfo(
            available_mb=1024, used_mb=512, total_mb=2048, percent_used=50.0
        )

        config = BatchConfig(strategy=BatchStrategy.MEMORY_OPTIMIZED, batch_size=10)
        processor = BatchProcessor(BatchStrategy.MEMORY_OPTIMIZED, config)

        batches = list(processor.process_stream(self.create_test_generator(25)))

        # Should produce batches
        assert len(batches) > 0
        # All batches should have items
        for batch in batches:
            assert len(batch) > 0

    def test_adaptive_batching(self):
        """Test adaptive batching strategy."""
        config = BatchConfig(strategy=BatchStrategy.ADAPTIVE, batch_size=5)
        processor = BatchProcessor(BatchStrategy.ADAPTIVE, config)

        batches = list(processor.process_stream(self.create_test_generator(15)))

        # Should produce batches
        assert len(batches) > 0
        # Performance history should be recorded
        assert len(processor._performance_history) > 0

    def test_invalid_strategy_validation_at_init(self):
        """Test strategy validation during BatchProcessor initialization."""
        config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=5)

        # Test with invalid strategy type (not BatchStrategy enum)
        with pytest.raises(InvalidBatchConfigError, match="strategy must be a BatchStrategy enum"):
            BatchProcessor("invalid_strategy", config)

    def test_unknown_strategy_error(self):
        """Test error handling for unknown batch strategy."""
        config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=5)
        processor = BatchProcessor(BatchStrategy.SEQUENTIAL, config)
        processor.strategy = "unknown_strategy"  # Force unknown strategy

        with pytest.raises(BatchProcessingError):
            list(processor.process_stream(self.create_test_generator(5)))

    @patch("src.batch.MemoryInfo.current")
    def test_memory_monitoring_critical_level(self, mock_memory):
        """Test memory monitoring with critical memory levels."""
        # Mock very high memory usage
        mock_memory.return_value = MemoryInfo(
            available_mb=100, used_mb=1900, total_mb=2048, percent_used=95.0
        )

        config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=5)
        processor = BatchProcessor(BatchStrategy.SEQUENTIAL, config)

        with pytest.raises(MemoryLimitExceededError):
            list(processor.process_with_memory_monitoring(self.create_test_generator(10)))


class TestUtilityFunctions:
    """Test utility functions."""

    @patch("psutil.virtual_memory")
    def test_get_memory_usage(self, mock_memory):
        """Test get_memory_usage function."""
        mock_memory.return_value = MagicMock(
            available=1024 * 1024 * 1024,
            used=512 * 1024 * 1024,
            total=2048 * 1024 * 1024,
            percent=25.0,
        )

        memory_info = get_memory_usage()
        assert memory_info.available_mb == 1024
        assert memory_info.used_mb == 512

    @patch("src.batch.get_memory_usage")
    def test_memory_monitor_context_manager(self, mock_get_memory):
        """Test memory monitor context manager."""
        # Mock memory usage below limit
        mock_get_memory.return_value = MemoryInfo(
            available_mb=1024, used_mb=400, total_mb=2048, percent_used=20.0
        )

        # Should not raise exception
        with memory_monitor(max_memory_mb=500) as initial_memory:
            assert initial_memory.used_mb == 400

    @patch("src.batch.get_memory_usage")
    def test_memory_monitor_exceeds_limit(self, mock_get_memory):
        """Test memory monitor when limit is exceeded."""
        # Mock memory usage that exceeds limit
        mock_get_memory.side_effect = [
            MemoryInfo(available_mb=1024, used_mb=400, total_mb=2048, percent_used=20.0),  # Initial
            MemoryInfo(available_mb=500, used_mb=600, total_mb=2048, percent_used=30.0),  # Final
        ]

        with pytest.raises(MemoryLimitExceededError):
            with memory_monitor(max_memory_mb=500):
                pass


class TestSafeMathematicalOperations:
    """Test safe mathematical operation utilities."""

    def test_safe_division_normal_cases(self):
        """Test safe division with normal cases."""
        from src.batch import safe_division

        # Normal division
        assert safe_division(10.0, 2.0) == 5.0
        assert safe_division(15.0, 3.0) == 5.0
        assert safe_division(7.0, 2.0) == 3.5

    def test_safe_division_zero_denominator(self):
        """Test safe division with zero denominator."""
        from src.batch import safe_division

        # Division by zero with default fallback
        assert safe_division(10.0, 0.0) == 0.0
        assert safe_division(0.0, 0.0) == 0.0

        # Division by zero with custom fallback
        assert safe_division(10.0, 0.0, fallback=float("inf")) == float("inf")
        assert safe_division(10.0, 0.0, fallback=-1.0) == -1.0

    def test_safe_division_edge_cases(self):
        """Test safe division with edge cases."""
        from src.batch import safe_division

        # Very small denominator
        result = safe_division(1.0, 1e-10)
        assert result == 1e10

        # Negative values
        assert safe_division(-10.0, 2.0) == -5.0
        assert safe_division(10.0, -2.0) == -5.0
        assert safe_division(-10.0, -2.0) == 5.0


class TestBatchMetrics:
    """Test BatchMetrics data model."""

    def test_batch_metrics_creation(self):
        """Test BatchMetrics creation and properties."""
        metrics = BatchMetrics(
            total_batches=10,
            avg_batch_size=32.5,
            memory_usage_mb=256.0,
            processing_time_seconds=5.2,
            throughput_samples_per_second=62.5,
        )

        assert metrics.total_batches == 10
        assert metrics.avg_batch_size == 32.5
        assert metrics.total_samples == 325  # 10 * 32.5 rounded
        assert metrics.memory_usage_mb == 256.0
        assert metrics.processing_time_seconds == 5.2
        assert metrics.throughput_samples_per_second == 62.5
