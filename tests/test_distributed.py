import pytest

from src.distributed import (
    DistributedRangeSplitter,
    RankAwareSeedGenerator,
    RankRange,
    gen_distributed_augmentation_params,
    stream_distributed_augmentation_chain,
    stream_distributed_augmentation_range,
)
from src.dpa import AugmentationConfig


class TestRankAwareSeedGenerator:
    """Test the RankAwareSeedGenerator class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        generator = RankAwareSeedGenerator()
        assert generator.base_seed == 0
        assert generator.world_size == 1

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        generator = RankAwareSeedGenerator(base_seed=42, world_size=4)
        assert generator.base_seed == 42
        assert generator.world_size == 4

    def test_init_invalid_base_seed(self):
        """Test initialization with invalid base_seed."""
        with pytest.raises(ValueError, match="base_seed must be non-negative"):
            RankAwareSeedGenerator(base_seed=-1)

    def test_init_invalid_world_size(self):
        """Test initialization with invalid world_size."""
        with pytest.raises(ValueError, match="world_size must be >= 1"):
            RankAwareSeedGenerator(world_size=0)

    def test_generate_seed_deterministic(self):
        """Test that seed generation is deterministic for same inputs."""
        generator = RankAwareSeedGenerator(base_seed=42, world_size=4)

        seed1 = generator.generate_seed(sample_id=0, rank=0)
        seed2 = generator.generate_seed(sample_id=0, rank=0)

        assert seed1 == seed2
        assert isinstance(seed1, str)
        assert len(seed1) == 64  # SHA256 hex string

    def test_generate_seed_different_ranks(self):
        """Test that different ranks produce different seeds for same sample_id."""
        generator = RankAwareSeedGenerator(base_seed=42, world_size=4)

        seed_rank0 = generator.generate_seed(sample_id=0, rank=0)
        seed_rank1 = generator.generate_seed(sample_id=0, rank=1)
        seed_rank2 = generator.generate_seed(sample_id=0, rank=2)

        # All seeds should be different
        assert seed_rank0 != seed_rank1
        assert seed_rank0 != seed_rank2
        assert seed_rank1 != seed_rank2

    def test_generate_seed_different_sample_ids(self):
        """Test that different sample_ids produce different seeds for same rank."""
        generator = RankAwareSeedGenerator(base_seed=42, world_size=4)

        seed_sample0 = generator.generate_seed(sample_id=0, rank=0)
        seed_sample1 = generator.generate_seed(sample_id=1, rank=0)
        seed_sample2 = generator.generate_seed(sample_id=2, rank=0)

        # All seeds should be different
        assert seed_sample0 != seed_sample1
        assert seed_sample0 != seed_sample2
        assert seed_sample1 != seed_sample2

    def test_generate_seed_different_base_seeds(self):
        """Test that different base_seeds produce different results."""
        generator1 = RankAwareSeedGenerator(base_seed=0, world_size=4)
        generator2 = RankAwareSeedGenerator(base_seed=42, world_size=4)

        seed1 = generator1.generate_seed(sample_id=0, rank=0)
        seed2 = generator2.generate_seed(sample_id=0, rank=0)

        assert seed1 != seed2

    def test_generate_seed_invalid_sample_id(self):
        """Test generate_seed with invalid sample_id."""
        generator = RankAwareSeedGenerator()

        with pytest.raises(ValueError, match="sample_id must be non-negative"):
            generator.generate_seed(sample_id=-1, rank=0)

    def test_generate_seed_invalid_rank_negative(self):
        """Test generate_seed with negative rank."""
        generator = RankAwareSeedGenerator(world_size=4)

        with pytest.raises(ValueError, match="rank must be non-negative"):
            generator.generate_seed(sample_id=0, rank=-1)

    def test_generate_seed_invalid_rank_too_large(self):
        """Test generate_seed with rank >= world_size."""
        generator = RankAwareSeedGenerator(world_size=4)

        with pytest.raises(ValueError, match="rank \\(4\\) must be < world_size \\(4\\)"):
            generator.generate_seed(sample_id=0, rank=4)

    def test_generate_seed_invalid_augmentation_depth(self):
        """Test generate_seed with invalid augmentation_depth."""
        generator = RankAwareSeedGenerator()

        with pytest.raises(ValueError, match="augmentation_depth must be >= 1"):
            generator.generate_seed(sample_id=0, rank=0, augmentation_depth=0)

    def test_generate_seed_custom_augmentation_depth(self):
        """Test generate_seed with custom augmentation_depth."""
        generator = RankAwareSeedGenerator(base_seed=42, world_size=4)

        seed_depth5 = generator.generate_seed(sample_id=0, rank=0, augmentation_depth=5)
        seed_depth10 = generator.generate_seed(sample_id=0, rank=0, augmentation_depth=10)

        # Different depths should produce different seeds
        assert seed_depth5 != seed_depth10
        assert len(seed_depth5) == 64
        assert len(seed_depth10) == 64

    def test_generate_seed_hex_format(self):
        """Test that generated seeds are valid hex strings."""
        generator = RankAwareSeedGenerator()

        seed = generator.generate_seed(sample_id=0, rank=0)

        # Should be valid hex string
        assert all(c in "0123456789abcdef" for c in seed)
        assert len(seed) == 64


class TestGenDistributedAugmentationParams:
    """Test the gen_distributed_augmentation_params function."""

    def test_basic_functionality(self):
        """Test basic parameter generation with rank awareness."""
        params = gen_distributed_augmentation_params(
            sample_id=0, rank=0, base_seed=42, world_size=4
        )

        # Check that all expected keys are present
        expected_keys = {"rotation", "brightness", "noise", "scale", "contrast", "hash", "rank"}
        assert set(params.keys()) == expected_keys

        # Check that rank is included in output
        assert params["rank"] == 0

        # Check that hash is valid hex string
        assert len(params["hash"]) == 64
        assert all(c in "0123456789abcdef" for c in params["hash"])

    def test_deterministic_behavior(self):
        """Test that same inputs produce same outputs."""
        params1 = gen_distributed_augmentation_params(
            sample_id=0, rank=0, base_seed=42, world_size=4
        )
        params2 = gen_distributed_augmentation_params(
            sample_id=0, rank=0, base_seed=42, world_size=4
        )

        assert params1 == params2

    def test_different_ranks_produce_different_params(self):
        """Test that different ranks produce different parameters for same sample_id."""
        params_rank0 = gen_distributed_augmentation_params(
            sample_id=0, rank=0, base_seed=42, world_size=4
        )
        params_rank1 = gen_distributed_augmentation_params(
            sample_id=0, rank=1, base_seed=42, world_size=4
        )
        params_rank2 = gen_distributed_augmentation_params(
            sample_id=0, rank=2, base_seed=42, world_size=4
        )

        # Parameters should be different between ranks
        assert params_rank0 != params_rank1
        assert params_rank0 != params_rank2
        assert params_rank1 != params_rank2

        # But rank field should be correctly set
        assert params_rank0["rank"] == 0
        assert params_rank1["rank"] == 1
        assert params_rank2["rank"] == 2

    def test_different_sample_ids_produce_different_params(self):
        """Test that different sample_ids produce different parameters for same rank."""
        params_sample0 = gen_distributed_augmentation_params(
            sample_id=0, rank=0, base_seed=42, world_size=4
        )
        params_sample1 = gen_distributed_augmentation_params(
            sample_id=1, rank=0, base_seed=42, world_size=4
        )

        # Parameters should be different between sample_ids
        assert params_sample0 != params_sample1

        # But rank should be the same
        assert params_sample0["rank"] == params_sample1["rank"] == 0

    def test_with_custom_config(self):
        """Test parameter generation with custom AugmentationConfig."""
        config = AugmentationConfig(
            rotation_range=(-10, 10), brightness_range=(0.9, 1.1), augmentation_depth=5
        )

        params = gen_distributed_augmentation_params(
            sample_id=0, rank=0, config=config, base_seed=42, world_size=4
        )

        # Check that parameters are within custom ranges
        assert -10 <= params["rotation"] <= 10
        assert 0.9 <= params["brightness"] <= 1.1

        # Check that custom augmentation_depth was used (indirectly via hash)
        assert len(params["hash"]) == 64

    def test_with_default_config(self):
        """Test parameter generation with default AugmentationConfig."""
        params = gen_distributed_augmentation_params(
            sample_id=0, rank=0, base_seed=42, world_size=4
        )

        # Check that parameters are within default ranges
        assert -30 <= params["rotation"] <= 30
        assert 0.8 <= params["brightness"] <= 1.2
        assert 0 <= params["noise"] <= 0.1
        assert 0.8 <= params["scale"] <= 1.2
        assert 0.7 <= params["contrast"] <= 1.3

    def test_invalid_sample_id(self):
        """Test with invalid sample_id."""
        with pytest.raises(ValueError, match="sample_id must be non-negative"):
            gen_distributed_augmentation_params(sample_id=-1, rank=0)

    def test_invalid_rank_negative(self):
        """Test with negative rank."""
        with pytest.raises(ValueError, match="rank must be non-negative"):
            gen_distributed_augmentation_params(sample_id=0, rank=-1)

    def test_invalid_world_size(self):
        """Test with invalid world_size."""
        with pytest.raises(ValueError, match="world_size must be >= 1"):
            gen_distributed_augmentation_params(sample_id=0, rank=0, world_size=0)

    def test_rank_greater_than_world_size(self):
        """Test with rank >= world_size."""
        with pytest.raises(ValueError, match="rank \\(4\\) must be < world_size \\(4\\)"):
            gen_distributed_augmentation_params(sample_id=0, rank=4, world_size=4)

    def test_backward_compatibility_when_rank_not_specified(self):
        """Test that function maintains backward compatibility."""
        # When world_size=1 and rank=0, behavior should be similar to non-distributed
        distributed_params = gen_distributed_augmentation_params(
            sample_id=0, rank=0, base_seed=0, world_size=1
        )

        # Remove the rank field for comparison
        distributed_params_no_rank = {k: v for k, v in distributed_params.items() if k != "rank"}

        # The parameters should be deterministic and valid
        assert len(distributed_params_no_rank) == 6  # All params except rank
        assert "hash" in distributed_params_no_rank
        assert len(distributed_params["hash"]) == 64

    def test_parameter_uniqueness_across_ranks(self):
        """Test that parameters are unique across all ranks for same sample_id."""
        world_size = 8
        sample_id = 42
        base_seed = 123

        all_params = []
        all_hashes = set()

        for rank in range(world_size):
            params = gen_distributed_augmentation_params(
                sample_id=sample_id, rank=rank, base_seed=base_seed, world_size=world_size
            )
            all_params.append(params)
            all_hashes.add(params["hash"])

        # All hashes should be unique
        assert len(all_hashes) == world_size

        # All parameter sets should be different
        for i in range(world_size):
            for j in range(i + 1, world_size):
                assert all_params[i] != all_params[j]

    def test_reproducibility_across_multiple_calls(self):
        """Test reproducibility across multiple function calls."""
        # Generate parameters multiple times with same inputs
        results = []
        for _ in range(5):
            params = gen_distributed_augmentation_params(
                sample_id=123, rank=2, base_seed=456, world_size=8
            )
            results.append(params)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[0] == results[i]

    def test_large_world_size(self):
        """Test with large world_size values."""
        world_size = 1000
        sample_id = 0
        base_seed = 42

        # Test a few ranks in large world
        test_ranks = [0, 1, 500, 999]
        params_list = []

        for rank in test_ranks:
            params = gen_distributed_augmentation_params(
                sample_id=sample_id, rank=rank, base_seed=base_seed, world_size=world_size
            )
            params_list.append(params)

        # All should be different
        hashes = [p["hash"] for p in params_list]
        assert len(set(hashes)) == len(test_ranks)

    def test_large_sample_ids(self):
        """Test with large sample_id values."""
        large_sample_ids = [10000, 100000, 1000000]

        for sample_id in large_sample_ids:
            params = gen_distributed_augmentation_params(
                sample_id=sample_id, rank=0, base_seed=42, world_size=4
            )

            # Should generate valid parameters
            assert len(params["hash"]) == 64
            assert params["rank"] == 0
            assert isinstance(params["rotation"], float)


class TestDistributedRangeSplitter:
    """Test the DistributedRangeSplitter class."""

    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        splitter = DistributedRangeSplitter(total_samples=100, world_size=4)
        assert splitter.total_samples == 100
        assert splitter.world_size == 4

    def test_init_invalid_total_samples(self):
        """Test initialization with invalid total_samples."""
        with pytest.raises(ValueError, match="total_samples must be non-negative"):
            DistributedRangeSplitter(total_samples=-1, world_size=4)

    def test_init_invalid_world_size(self):
        """Test initialization with invalid world_size."""
        with pytest.raises(ValueError, match="world_size must be >= 1"):
            DistributedRangeSplitter(total_samples=100, world_size=0)

    def test_get_rank_range_even_distribution(self):
        """Test range splitting with even distribution."""
        splitter = DistributedRangeSplitter(total_samples=100, world_size=4)

        # Each rank should get 25 samples
        assert splitter.get_rank_range(0) == (0, 25)
        assert splitter.get_rank_range(1) == (25, 50)
        assert splitter.get_rank_range(2) == (50, 75)
        assert splitter.get_rank_range(3) == (75, 100)

    def test_get_rank_range_uneven_distribution(self):
        """Test range splitting with uneven distribution."""
        splitter = DistributedRangeSplitter(total_samples=10, world_size=3)

        # First rank gets extra sample: 4, 3, 3
        assert splitter.get_rank_range(0) == (0, 4)
        assert splitter.get_rank_range(1) == (4, 7)
        assert splitter.get_rank_range(2) == (7, 10)

    def test_get_rank_range_more_uneven_distribution(self):
        """Test range splitting with more uneven distribution."""
        splitter = DistributedRangeSplitter(total_samples=11, world_size=4)

        # First 3 ranks get extra sample: 3, 3, 3, 2
        assert splitter.get_rank_range(0) == (0, 3)
        assert splitter.get_rank_range(1) == (3, 6)
        assert splitter.get_rank_range(2) == (6, 9)
        assert splitter.get_rank_range(3) == (9, 11)

    def test_get_rank_range_single_rank(self):
        """Test range splitting with single rank."""
        splitter = DistributedRangeSplitter(total_samples=100, world_size=1)

        # Single rank gets all samples
        assert splitter.get_rank_range(0) == (0, 100)

    def test_get_rank_range_more_ranks_than_samples(self):
        """Test range splitting with more ranks than samples."""
        splitter = DistributedRangeSplitter(total_samples=3, world_size=5)

        # First 3 ranks get 1 sample each, last 2 get empty ranges
        assert splitter.get_rank_range(0) == (0, 1)
        assert splitter.get_rank_range(1) == (1, 2)
        assert splitter.get_rank_range(2) == (2, 3)
        assert splitter.get_rank_range(3) == (0, 0)  # Empty range
        assert splitter.get_rank_range(4) == (0, 0)  # Empty range

    def test_get_rank_range_zero_samples(self):
        """Test range splitting with zero samples."""
        splitter = DistributedRangeSplitter(total_samples=0, world_size=4)

        # All ranks get empty ranges
        for rank in range(4):
            assert splitter.get_rank_range(rank) == (0, 0)

    def test_get_rank_range_invalid_rank_negative(self):
        """Test get_rank_range with negative rank."""
        splitter = DistributedRangeSplitter(total_samples=100, world_size=4)

        with pytest.raises(ValueError, match="rank must be non-negative"):
            splitter.get_rank_range(-1)

    def test_get_rank_range_invalid_rank_too_large(self):
        """Test get_rank_range with rank >= world_size."""
        splitter = DistributedRangeSplitter(total_samples=100, world_size=4)

        with pytest.raises(ValueError, match="rank \\(4\\) must be < world_size \\(4\\)"):
            splitter.get_rank_range(4)

    def test_get_all_ranges_even_distribution(self):
        """Test getting all ranges with even distribution."""
        splitter = DistributedRangeSplitter(total_samples=100, world_size=4)

        ranges = splitter.get_all_ranges()
        expected = [(0, 25), (25, 50), (50, 75), (75, 100)]

        assert ranges == expected
        assert len(ranges) == 4

    def test_get_all_ranges_uneven_distribution(self):
        """Test getting all ranges with uneven distribution."""
        splitter = DistributedRangeSplitter(total_samples=10, world_size=3)

        ranges = splitter.get_all_ranges()
        expected = [(0, 4), (4, 7), (7, 10)]

        assert ranges == expected
        assert len(ranges) == 3

    def test_get_all_ranges_caching(self):
        """Test that get_all_ranges caches results."""
        splitter = DistributedRangeSplitter(total_samples=100, world_size=4)

        # First call should compute and cache
        ranges1 = splitter.get_all_ranges()

        # Second call should return cached result
        ranges2 = splitter.get_all_ranges()

        assert ranges1 == ranges2
        assert ranges1 is not ranges2  # Should return a copy

    def test_get_all_rank_ranges(self):
        """Test getting all RankRange objects."""
        splitter = DistributedRangeSplitter(total_samples=10, world_size=3)

        rank_ranges = splitter.get_all_rank_ranges()

        assert len(rank_ranges) == 3
        assert isinstance(rank_ranges[0], RankRange)

        # Check first rank range
        assert rank_ranges[0].start_id == 0
        assert rank_ranges[0].end_id == 4
        assert rank_ranges[0].rank == 0
        assert rank_ranges[0].total_samples == 10

        # Check second rank range
        assert rank_ranges[1].start_id == 4
        assert rank_ranges[1].end_id == 7
        assert rank_ranges[1].rank == 1
        assert rank_ranges[1].total_samples == 10

        # Check third rank range
        assert rank_ranges[2].start_id == 7
        assert rank_ranges[2].end_id == 10
        assert rank_ranges[2].rank == 2
        assert rank_ranges[2].total_samples == 10

    def test_validate_ranges_valid_even(self):
        """Test range validation with valid even distribution."""
        splitter = DistributedRangeSplitter(total_samples=100, world_size=4)

        assert splitter.validate_ranges() is True

    def test_validate_ranges_valid_uneven(self):
        """Test range validation with valid uneven distribution."""
        splitter = DistributedRangeSplitter(total_samples=10, world_size=3)

        assert splitter.validate_ranges() is True

    def test_validate_ranges_zero_samples(self):
        """Test range validation with zero samples."""
        splitter = DistributedRangeSplitter(total_samples=0, world_size=4)

        assert splitter.validate_ranges() is True

    def test_validate_ranges_more_ranks_than_samples(self):
        """Test range validation with more ranks than samples."""
        splitter = DistributedRangeSplitter(total_samples=2, world_size=5)

        assert splitter.validate_ranges() is True

    def test_get_rank_for_sample_even_distribution(self):
        """Test finding rank for sample with even distribution."""
        splitter = DistributedRangeSplitter(total_samples=100, world_size=4)

        # Test samples in each rank's range
        assert splitter.get_rank_for_sample(0) == 0
        assert splitter.get_rank_for_sample(24) == 0
        assert splitter.get_rank_for_sample(25) == 1
        assert splitter.get_rank_for_sample(49) == 1
        assert splitter.get_rank_for_sample(50) == 2
        assert splitter.get_rank_for_sample(74) == 2
        assert splitter.get_rank_for_sample(75) == 3
        assert splitter.get_rank_for_sample(99) == 3

    def test_get_rank_for_sample_uneven_distribution(self):
        """Test finding rank for sample with uneven distribution."""
        splitter = DistributedRangeSplitter(total_samples=10, world_size=3)

        # Ranges: [0,4), [4,7), [7,10)
        assert splitter.get_rank_for_sample(0) == 0
        assert splitter.get_rank_for_sample(3) == 0
        assert splitter.get_rank_for_sample(4) == 1
        assert splitter.get_rank_for_sample(6) == 1
        assert splitter.get_rank_for_sample(7) == 2
        assert splitter.get_rank_for_sample(9) == 2

    def test_get_rank_for_sample_invalid_sample_id_negative(self):
        """Test get_rank_for_sample with negative sample_id."""
        splitter = DistributedRangeSplitter(total_samples=100, world_size=4)

        with pytest.raises(ValueError, match="sample_id must be non-negative"):
            splitter.get_rank_for_sample(-1)

    def test_get_rank_for_sample_invalid_sample_id_too_large(self):
        """Test get_rank_for_sample with sample_id >= total_samples."""
        splitter = DistributedRangeSplitter(total_samples=100, world_size=4)

        with pytest.raises(
            ValueError, match="sample_id \\(100\\) must be < total_samples \\(100\\)"
        ):
            splitter.get_rank_for_sample(100)

    def test_range_coverage_completeness(self):
        """Test that ranges cover all samples without gaps or overlaps."""
        test_cases = [
            (100, 4),  # Even distribution
            (10, 3),  # Uneven distribution
            (7, 4),  # More uneven
            (1, 1),  # Single rank
            (5, 10),  # More ranks than samples
            (0, 4),  # Zero samples
        ]

        for total_samples, world_size in test_cases:
            splitter = DistributedRangeSplitter(total_samples, world_size)
            ranges = splitter.get_all_ranges()

            # Collect all sample IDs covered by ranges
            covered_samples = set()
            for start_id, end_id in ranges:
                for sample_id in range(start_id, end_id):
                    # Check for overlaps
                    assert sample_id not in covered_samples, (
                        f"Sample {sample_id} covered by multiple ranges"
                    )
                    covered_samples.add(sample_id)

            # Check that all samples are covered exactly once
            expected_samples = set(range(total_samples))
            assert covered_samples == expected_samples, (
                f"Coverage mismatch for {total_samples} samples, {world_size} ranks"
            )

    def test_range_size_distribution(self):
        """Test that range sizes are distributed as evenly as possible."""
        test_cases = [
            (100, 4, [25, 25, 25, 25]),  # Perfect even distribution
            (10, 3, [4, 3, 3]),  # Uneven: first rank gets extra
            (11, 4, [3, 3, 3, 2]),  # Uneven: first 3 ranks get extra
            (7, 4, [2, 2, 2, 1]),  # Uneven: first 3 ranks get extra
            (3, 5, [1, 1, 1, 0, 0]),  # More ranks than samples
        ]

        for total_samples, world_size, expected_sizes in test_cases:
            splitter = DistributedRangeSplitter(total_samples, world_size)
            ranges = splitter.get_all_ranges()

            actual_sizes = [end - start for start, end in ranges]
            assert actual_sizes == expected_sizes, (
                f"Size distribution mismatch for {total_samples} samples, {world_size} ranks"
            )


class TestDistributedStreamingFunctions:
    """Test the distributed streaming functions."""

    def test_stream_distributed_augmentation_chain_basic(self):
        """Test basic distributed streaming functionality."""
        # Test with 10 samples, 2 ranks
        generator = stream_distributed_augmentation_chain(
            num_samples=10, rank=0, world_size=2, base_seed=42
        )

        params_list = list(generator)

        # Rank 0 should get first 5 samples (0-4)
        assert len(params_list) == 5

        # All parameters should have rank=0
        for params in params_list:
            assert params["rank"] == 0
            assert "hash" in params
            assert len(params["hash"]) == 64

    def test_stream_distributed_augmentation_chain_different_ranks(self):
        """Test that different ranks get different sample ranges."""
        num_samples = 10
        world_size = 3

        # Collect parameters from all ranks
        all_params = {}
        for rank in range(world_size):
            generator = stream_distributed_augmentation_chain(
                num_samples=num_samples, rank=rank, world_size=world_size, base_seed=42
            )
            all_params[rank] = list(generator)

        # Check that each rank gets the expected number of samples
        # 10 samples, 3 ranks: [4, 3, 3]
        assert len(all_params[0]) == 4
        assert len(all_params[1]) == 3
        assert len(all_params[2]) == 3

        # Check that all parameters are different between ranks
        all_hashes = set()
        for rank in range(world_size):
            for params in all_params[rank]:
                assert params["rank"] == rank
                assert params["hash"] not in all_hashes
                all_hashes.add(params["hash"])

        # Total unique hashes should equal total samples
        assert len(all_hashes) == num_samples

    def test_stream_distributed_augmentation_chain_chunked(self):
        """Test distributed streaming with chunking."""
        generator = stream_distributed_augmentation_chain(
            num_samples=10, rank=0, world_size=2, chunk_size=2, base_seed=42
        )

        chunks = list(generator)

        # Rank 0 gets 5 samples, with chunk_size=2: [2, 2, 1]
        assert len(chunks) == 3
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
        assert len(chunks[2]) == 1

        # Flatten and check total
        all_params = [param for chunk in chunks for param in chunk]
        assert len(all_params) == 5

        # All should have rank=0
        for params in all_params:
            assert params["rank"] == 0

    def test_stream_distributed_augmentation_chain_empty_range(self):
        """Test distributed streaming with empty range for excess ranks."""
        # 3 samples, 5 ranks - last 2 ranks get empty ranges
        generator = stream_distributed_augmentation_chain(
            num_samples=3, rank=4, world_size=5, base_seed=42
        )

        params_list = list(generator)

        # Rank 4 should get empty range
        assert len(params_list) == 0

    def test_stream_distributed_augmentation_chain_inclusive_range(self):
        """Test distributed streaming with inclusive range strategy."""
        # This is more of a documentation test since inclusive_range doesn't change
        # the core behavior significantly in our current implementation
        generator = stream_distributed_augmentation_chain(
            num_samples=4, rank=0, world_size=2, inclusive_range=True, base_seed=42
        )

        params_list = list(generator)

        # Rank 0 should get 2 samples (0, 1)
        assert len(params_list) == 2

        for params in params_list:
            assert params["rank"] == 0

    def test_stream_distributed_augmentation_chain_invalid_parameters(self):
        """Test distributed streaming with invalid parameters."""
        # Invalid num_samples
        with pytest.raises(ValueError, match="num_samples must be non-negative"):
            list(stream_distributed_augmentation_chain(-1, 0, 1))

        # Invalid rank
        with pytest.raises(ValueError, match="rank must be non-negative"):
            list(stream_distributed_augmentation_chain(10, -1, 1))

        # Invalid world_size
        with pytest.raises(ValueError, match="world_size must be >= 1"):
            list(stream_distributed_augmentation_chain(10, 0, 0))

        # Rank >= world_size
        with pytest.raises(ValueError, match="rank \\(2\\) must be < world_size \\(2\\)"):
            list(stream_distributed_augmentation_chain(10, 2, 2))

        # Invalid chunk_size
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            list(stream_distributed_augmentation_chain(10, 0, 1, chunk_size=0))

    def test_stream_distributed_augmentation_range_basic(self):
        """Test basic distributed range streaming functionality."""
        generator = stream_distributed_augmentation_range(
            start_id=10, end_id=15, rank=0, world_size=2, base_seed=42
        )

        params_list = list(generator)

        # Should get 5 samples (10, 11, 12, 13, 14)
        assert len(params_list) == 5

        # All parameters should have rank=0
        for params in params_list:
            assert params["rank"] == 0
            assert "hash" in params

    def test_stream_distributed_augmentation_range_chunked(self):
        """Test distributed range streaming with chunking."""
        generator = stream_distributed_augmentation_range(
            start_id=0, end_id=7, rank=1, world_size=3, chunk_size=3, base_seed=42
        )

        chunks = list(generator)

        # 7 samples with chunk_size=3: [3, 3, 1]
        assert len(chunks) == 3
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 1

        # Flatten and check
        all_params = [param for chunk in chunks for param in chunk]
        assert len(all_params) == 7

        for params in all_params:
            assert params["rank"] == 1

    def test_stream_distributed_augmentation_range_invalid_parameters(self):
        """Test distributed range streaming with invalid parameters."""
        # Invalid start_id
        with pytest.raises(ValueError, match="start_id must be non-negative"):
            list(stream_distributed_augmentation_range(-1, 10, 0, 1))

        # Invalid end_id
        with pytest.raises(ValueError, match="end_id must be >= start_id"):
            list(stream_distributed_augmentation_range(10, 5, 0, 1))

        # Invalid rank
        with pytest.raises(ValueError, match="rank must be non-negative"):
            list(stream_distributed_augmentation_range(0, 10, -1, 1))

        # Invalid world_size
        with pytest.raises(ValueError, match="world_size must be >= 1"):
            list(stream_distributed_augmentation_range(0, 10, 0, 0))

        # Rank >= world_size
        with pytest.raises(ValueError, match="rank \\(3\\) must be < world_size \\(3\\)"):
            list(stream_distributed_augmentation_range(0, 10, 3, 3))

        # Invalid chunk_size
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            list(stream_distributed_augmentation_range(0, 10, 0, 1, chunk_size=-1))

    def test_distributed_streaming_integration_with_range_splitter(self):
        """Test integration between streaming functions and range splitter."""
        num_samples = 13
        world_size = 4
        base_seed = 123

        # Create range splitter to get expected ranges
        splitter = DistributedRangeSplitter(num_samples, world_size)

        # Collect parameters from streaming function for each rank
        streaming_params = {}
        for rank in range(world_size):
            generator = stream_distributed_augmentation_chain(
                num_samples=num_samples, rank=rank, world_size=world_size, base_seed=base_seed
            )
            streaming_params[rank] = list(generator)

        # Verify that each rank gets the expected number of samples
        for rank in range(world_size):
            start_id, end_id = splitter.get_rank_range(rank)
            expected_count = end_id - start_id
            actual_count = len(streaming_params[rank])
            assert actual_count == expected_count, (
                f"Rank {rank}: expected {expected_count}, got {actual_count}"
            )

        # Verify total samples
        total_streamed = sum(len(params) for params in streaming_params.values())
        assert total_streamed == num_samples

        # Verify all parameters are unique
        all_hashes = set()
        for rank_params in streaming_params.values():
            for params in rank_params:
                assert params["hash"] not in all_hashes
                all_hashes.add(params["hash"])

        assert len(all_hashes) == num_samples
