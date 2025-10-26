"""
Integration tests for distributed training functionality.

Tests end-to-end distributed parameter generation, multi-rank coordination,
range splitting, and reproducibility across distributed runs as specified
in requirements 1.1, 1.2, 1.3, 4.1, and 4.2.
"""

import time

from src.distributed import (
    DistributedRangeSplitter,
    RankAwareSeedGenerator,
    gen_distributed_augmentation_params,
    stream_distributed_augmentation_chain,
    stream_distributed_augmentation_range,
)
from src.dpa import AugmentationConfig


class TestDistributedTrainingIntegration:
    """Integration tests for distributed training functionality."""

    def test_end_to_end_distributed_parameter_generation(self):
        """Test end-to-end distributed parameter generation across multiple ranks."""
        # Test configuration
        num_samples = 100
        world_size = 4
        base_seed = 42

        # Generate parameters for all ranks
        all_params = {}
        all_hashes = set()

        for rank in range(world_size):
            rank_params = []

            # Use range splitter to get this rank's sample range
            splitter = DistributedRangeSplitter(num_samples, world_size)
            start_id, end_id = splitter.get_rank_range(rank)

            # Generate parameters for this rank's samples
            for sample_id in range(start_id, end_id):
                params = gen_distributed_augmentation_params(
                    sample_id=sample_id, rank=rank, base_seed=base_seed, world_size=world_size
                )
                rank_params.append(params)

                # Verify parameter structure
                assert "rotation" in params
                assert "brightness" in params
                assert "noise" in params
                assert "scale" in params
                assert "contrast" in params
                assert "hash" in params
                assert "rank" in params
                assert params["rank"] == rank

                # Ensure hash uniqueness across all ranks
                assert params["hash"] not in all_hashes
                all_hashes.add(params["hash"])

            all_params[rank] = rank_params

        # Verify total sample coverage
        total_generated = sum(len(params) for params in all_params.values())
        assert total_generated == num_samples

        # Verify all hashes are unique
        assert len(all_hashes) == num_samples

        # Verify parameter ranges are within expected bounds
        for rank_params in all_params.values():
            for params in rank_params:
                assert -30 <= params["rotation"] <= 30
                assert 0.8 <= params["brightness"] <= 1.2
                assert 0 <= params["noise"] <= 0.1
                assert 0.8 <= params["scale"] <= 1.2
                assert 0.7 <= params["contrast"] <= 1.3

    def test_multi_rank_coordination_and_range_splitting(self):
        """Test multi-rank coordination and range splitting functionality."""
        test_cases = [
            (100, 4),  # Even distribution
            (97, 4),  # Uneven distribution
            (10, 3),  # Small dataset
            (1000, 8),  # Large world size
            (5, 10),  # More ranks than samples
        ]

        for num_samples, world_size in test_cases:
            # Create range splitter
            splitter = DistributedRangeSplitter(num_samples, world_size)

            # Verify range splitting correctness
            all_ranges = splitter.get_all_ranges()
            assert len(all_ranges) == world_size

            # Verify no gaps or overlaps
            covered_samples = set()
            for _rank, (start_id, end_id) in enumerate(all_ranges):
                # Verify range validity
                assert start_id >= 0
                assert end_id >= start_id
                assert end_id <= num_samples

                # Check for overlaps
                for sample_id in range(start_id, end_id):
                    assert sample_id not in covered_samples
                    covered_samples.add(sample_id)

            # Verify complete coverage
            expected_samples = set(range(num_samples))
            assert covered_samples == expected_samples

            # Test coordination between ranks using streaming
            all_streamed_params = {}
            total_streamed = 0

            for rank in range(world_size):
                generator = stream_distributed_augmentation_chain(
                    num_samples=num_samples, rank=rank, world_size=world_size, base_seed=42
                )
                params_list = list(generator)
                all_streamed_params[rank] = params_list
                total_streamed += len(params_list)

                # Verify this rank got the expected range
                start_id, end_id = splitter.get_rank_range(rank)
                expected_count = end_id - start_id
                assert len(params_list) == expected_count

            # Verify total coordination
            assert total_streamed == num_samples

    def test_reproducibility_across_distributed_runs(self):
        """Test reproducibility across multiple distributed runs."""
        # Test configuration
        num_samples = 50
        world_size = 3
        base_seed = 123

        # Run 1: Generate parameters for all ranks
        run1_params = {}
        for rank in range(world_size):
            generator = stream_distributed_augmentation_chain(
                num_samples=num_samples, rank=rank, world_size=world_size, base_seed=base_seed
            )
            run1_params[rank] = list(generator)

        # Run 2: Generate parameters again with same configuration
        run2_params = {}
        for rank in range(world_size):
            generator = stream_distributed_augmentation_chain(
                num_samples=num_samples, rank=rank, world_size=world_size, base_seed=base_seed
            )
            run2_params[rank] = list(generator)

        # Verify exact reproducibility
        for rank in range(world_size):
            assert len(run1_params[rank]) == len(run2_params[rank])

            for i, (params1, params2) in enumerate(zip(run1_params[rank], run2_params[rank], strict=False)):
                # All parameters should be identical
                assert params1 == params2, f"Rank {rank}, sample {i}: parameters differ"

        # Test reproducibility with individual parameter generation
        for rank in range(world_size):
            splitter = DistributedRangeSplitter(num_samples, world_size)
            start_id, end_id = splitter.get_rank_range(rank)

            for sample_id in range(start_id, end_id):
                # Generate same parameters multiple times
                params1 = gen_distributed_augmentation_params(
                    sample_id=sample_id, rank=rank, base_seed=base_seed, world_size=world_size
                )

                params2 = gen_distributed_augmentation_params(
                    sample_id=sample_id, rank=rank, base_seed=base_seed, world_size=world_size
                )

                assert params1 == params2, (
                    f"Non-reproducible parameters for rank {rank}, sample {sample_id}"
                )

    def test_distributed_parameter_uniqueness_across_ranks(self):
        """Test that parameters are unique across ranks for same sample_id."""
        # Test configuration
        sample_id = 42
        world_size = 8
        base_seed = 456

        # Generate parameters for same sample_id across all ranks
        all_params = []
        all_hashes = set()

        for rank in range(world_size):
            params = gen_distributed_augmentation_params(
                sample_id=sample_id, rank=rank, base_seed=base_seed, world_size=world_size
            )
            all_params.append(params)

            # Verify hash uniqueness
            assert params["hash"] not in all_hashes
            all_hashes.add(params["hash"])

            # Verify rank is correctly set
            assert params["rank"] == rank

        # Verify all parameters are different
        for i in range(world_size):
            for j in range(i + 1, world_size):
                assert all_params[i] != all_params[j]

                # Specifically check that augmentation values differ
                # (they should be different due to different seeds)
                params_i = all_params[i]
                params_j = all_params[j]

                # At least one augmentation parameter should be different
                different = (
                    params_i["rotation"] != params_j["rotation"]
                    or params_i["brightness"] != params_j["brightness"]
                    or params_i["noise"] != params_j["noise"]
                    or params_i["scale"] != params_j["scale"]
                    or params_i["contrast"] != params_j["contrast"]
                )
                assert different, f"Parameters for ranks {i} and {j} are too similar"

    def test_distributed_streaming_with_custom_config(self):
        """Test distributed streaming with custom augmentation configuration."""
        # Custom configuration with narrow ranges for easier testing
        config = AugmentationConfig(
            rotation_range=(-10, 10),
            brightness_range=(0.9, 1.1),
            noise_range=(0, 0.05),
            scale_range=(0.95, 1.05),
            contrast_range=(0.9, 1.1),
            augmentation_depth=5,
        )

        num_samples = 20
        world_size = 2
        base_seed = 789

        # Generate parameters for both ranks
        all_params = {}
        for rank in range(world_size):
            generator = stream_distributed_augmentation_chain(
                num_samples=num_samples,
                rank=rank,
                world_size=world_size,
                config=config,
                base_seed=base_seed,
            )
            all_params[rank] = list(generator)

        # Verify parameters are within custom ranges
        for rank_params in all_params.values():
            for params in rank_params:
                assert -10 <= params["rotation"] <= 10
                assert 0.9 <= params["brightness"] <= 1.1
                assert 0 <= params["noise"] <= 0.05
                assert 0.95 <= params["scale"] <= 1.05
                assert 0.9 <= params["contrast"] <= 1.1

        # Verify total samples and uniqueness
        total_params = sum(len(params) for params in all_params.values())
        assert total_params == num_samples

        all_hashes = set()
        for rank_params in all_params.values():
            for params in rank_params:
                assert params["hash"] not in all_hashes
                all_hashes.add(params["hash"])

    def test_distributed_range_streaming_integration(self):
        """Test integration between distributed range streaming and coordination."""
        # Test configuration
        total_samples = 1000
        world_size = 4
        base_seed = 999

        # Create range splitter
        splitter = DistributedRangeSplitter(total_samples, world_size)

        # Test streaming specific ranges for each rank
        all_params = {}
        for rank in range(world_size):
            start_id, end_id = splitter.get_rank_range(rank)

            # Use range streaming function
            generator = stream_distributed_augmentation_range(
                start_id=start_id,
                end_id=end_id,
                rank=rank,
                world_size=world_size,
                base_seed=base_seed,
            )
            all_params[rank] = list(generator)

            # Verify correct number of samples
            expected_count = end_id - start_id
            assert len(all_params[rank]) == expected_count

        # Verify total coverage and uniqueness
        total_generated = sum(len(params) for params in all_params.values())
        assert total_generated == total_samples

        all_hashes = set()
        for rank_params in all_params.values():
            for params in rank_params:
                assert params["hash"] not in all_hashes
                all_hashes.add(params["hash"])

        assert len(all_hashes) == total_samples

    def test_distributed_chunked_streaming_coordination(self):
        """Test distributed streaming with chunking across multiple ranks."""
        # Test configuration
        num_samples = 60
        world_size = 3
        chunk_size = 5
        base_seed = 111

        # Generate chunked parameters for all ranks
        all_chunks = {}
        total_items = 0

        for rank in range(world_size):
            generator = stream_distributed_augmentation_chain(
                num_samples=num_samples,
                rank=rank,
                world_size=world_size,
                chunk_size=chunk_size,
                base_seed=base_seed,
            )
            chunks = list(generator)
            all_chunks[rank] = chunks

            # Count total items for this rank
            rank_items = sum(len(chunk) for chunk in chunks)
            total_items += rank_items

            # Verify chunk sizes
            for _i, chunk in enumerate(chunks[:-1]):  # All but last chunk
                assert len(chunk) == chunk_size

            # Last chunk may be smaller
            if chunks:
                assert len(chunks[-1]) <= chunk_size

        # Verify total coordination
        assert total_items == num_samples

        # Verify uniqueness across all chunks and ranks
        all_hashes = set()
        for rank_chunks in all_chunks.values():
            for chunk in rank_chunks:
                for params in chunk:
                    assert params["hash"] not in all_hashes
                    all_hashes.add(params["hash"])

        assert len(all_hashes) == num_samples

    def test_distributed_edge_cases_integration(self):
        """Test distributed functionality with edge cases."""
        # Test case 1: Single rank (world_size=1)
        generator = stream_distributed_augmentation_chain(
            num_samples=10, rank=0, world_size=1, base_seed=42
        )
        params_list = list(generator)
        assert len(params_list) == 10

        for params in params_list:
            assert params["rank"] == 0

        # Test case 2: Zero samples
        generator = stream_distributed_augmentation_chain(
            num_samples=0, rank=0, world_size=4, base_seed=42
        )
        params_list = list(generator)
        assert len(params_list) == 0

        # Test case 3: More ranks than samples
        all_params = {}
        for rank in range(5):
            generator = stream_distributed_augmentation_chain(
                num_samples=3, rank=rank, world_size=5, base_seed=42
            )
            all_params[rank] = list(generator)

        # First 3 ranks should get 1 sample each, last 2 should get 0
        assert len(all_params[0]) == 1
        assert len(all_params[1]) == 1
        assert len(all_params[2]) == 1
        assert len(all_params[3]) == 0
        assert len(all_params[4]) == 0

        # Verify total samples
        total_samples = sum(len(params) for params in all_params.values())
        assert total_samples == 3

    def test_distributed_performance_and_scalability(self):
        """Test distributed functionality performance and scalability."""
        # Test with larger configurations to ensure scalability
        test_configs = [
            (1000, 4),  # Medium dataset, small world
            (10000, 8),  # Large dataset, medium world
            (100, 16),  # Small dataset, large world
        ]

        for num_samples, world_size in test_configs:
            start_time = time.time()

            # Generate parameters for all ranks
            total_generated = 0
            all_hashes = set()

            for rank in range(world_size):
                generator = stream_distributed_augmentation_chain(
                    num_samples=num_samples, rank=rank, world_size=world_size, base_seed=42
                )

                rank_params = list(generator)
                total_generated += len(rank_params)

                # Collect hashes for uniqueness verification
                for params in rank_params:
                    all_hashes.add(params["hash"])

            end_time = time.time()
            generation_time = end_time - start_time

            # Verify correctness
            assert total_generated == num_samples
            assert len(all_hashes) == num_samples

            # Performance should be reasonable (less than 10 seconds for these sizes)
            assert generation_time < 10.0, (
                f"Generation took too long: {generation_time:.2f}s for {num_samples} samples, {world_size} ranks"
            )

            # Calculate throughput
            throughput = num_samples / generation_time if generation_time > 0 else float("inf")
            assert throughput > 100, f"Throughput too low: {throughput:.2f} samples/sec"

    def test_distributed_seed_generator_integration(self):
        """Test integration of RankAwareSeedGenerator with distributed functions."""
        # Test configuration
        base_seed = 12345
        world_size = 4
        sample_id = 100

        # Create seed generator
        seed_generator = RankAwareSeedGenerator(base_seed=base_seed, world_size=world_size)

        # Generate seeds for all ranks
        seeds = {}
        for rank in range(world_size):
            seed = seed_generator.generate_seed(sample_id=sample_id, rank=rank)
            seeds[rank] = seed

            # Verify seed format
            assert len(seed) == 64  # SHA256 hex string
            assert all(c in "0123456789abcdef" for c in seed)

        # Verify all seeds are unique
        unique_seeds = set(seeds.values())
        assert len(unique_seeds) == world_size

        # Test integration with parameter generation
        for rank in range(world_size):
            params = gen_distributed_augmentation_params(
                sample_id=sample_id, rank=rank, base_seed=base_seed, world_size=world_size
            )

            # The hash in params should match the seed from generator
            expected_seed = seed_generator.generate_seed(sample_id=sample_id, rank=rank)
            assert params["hash"] == expected_seed
            assert params["rank"] == rank

    def test_distributed_range_splitter_integration(self):
        """Test integration of DistributedRangeSplitter with streaming functions."""
        # Test configuration
        num_samples = 77
        world_size = 5
        base_seed = 54321

        # Create range splitter
        splitter = DistributedRangeSplitter(num_samples, world_size)

        # Verify range splitting
        assert splitter.validate_ranges()

        # Get all rank ranges
        rank_ranges = splitter.get_all_rank_ranges()
        assert len(rank_ranges) == world_size

        # Test streaming integration
        streaming_results = {}
        for rank in range(world_size):
            # Get expected range for this rank
            expected_range = rank_ranges[rank]
            assert expected_range.rank == rank
            assert expected_range.total_samples == num_samples

            # Stream parameters for this rank
            generator = stream_distributed_augmentation_chain(
                num_samples=num_samples, rank=rank, world_size=world_size, base_seed=base_seed
            )
            params_list = list(generator)
            streaming_results[rank] = params_list

            # Verify count matches expected range
            expected_count = expected_range.end_id - expected_range.start_id
            assert len(params_list) == expected_count

        # Verify total coverage
        total_streamed = sum(len(params) for params in streaming_results.values())
        assert total_streamed == num_samples

        # Test sample-to-rank mapping
        for sample_id in range(num_samples):
            expected_rank = splitter.get_rank_for_sample(sample_id)

            # Verify this sample_id would be processed by the expected rank
            expected_range = rank_ranges[expected_rank]
            assert expected_range.start_id <= sample_id < expected_range.end_id
