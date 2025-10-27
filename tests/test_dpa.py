import json
import tempfile
from pathlib import Path

import pytest

from src.dpa import (
    PRESETS,
    AugmentationConfig,
    GeneratorExhaustionError,
    PartialWriteError,
    ResourceCleanupError,
    StreamingContext,
    StreamingError,
    StreamingIOError,
    compute_statistics,
    compute_streaming_statistics,
    create_streaming_metadata,
    create_streaming_progress,
    estimate_memory_usage,
    fib,
    format_memory_size,
    format_progress_info,
    gen_augmentation_params,
    gen_augmentation_seed,
    generate_augmentation_chain,
    generator_to_list,
    get_current_memory_usage,
    get_preset,
    list_to_generator,
    load_augmentation_chain,
    load_augmentation_stream,
    recover_partial_write,
    resume_streaming_save,
    safe_cleanup_generator,
    safe_close_file,
    save_augmentation_chain,
    save_augmentation_stream,
    stream_augmentation_chain,
    stream_augmentation_range,
    update_streaming_progress,
    with_streaming_context,
)


class TestAugmentationConfig:
    def test_default_config(self):
        config = AugmentationConfig()
        assert config.rotation_range == (-30, 30)
        assert config.brightness_range == (0.8, 1.2)
        assert config.augmentation_depth == 10

    def test_invalid_range_lower_greater_than_upper(self):
        with pytest.raises(ValueError):
            AugmentationConfig(rotation_range=(30, 10))

    def test_invalid_augmentation_depth(self):
        with pytest.raises(ValueError):
            AugmentationConfig(augmentation_depth=0)

    def test_custom_config(self):
        config = AugmentationConfig(
            rotation_range=(-45, 45),
            brightness_range=(0.7, 1.3),
        )
        assert config.rotation_range == (-45, 45)
        assert config.brightness_range == (0.7, 1.3)


class TestFibonacci:
    def test_fib_zero(self):
        assert fib(0) == 0

    def test_fib_one(self):
        assert fib(1) == 1

    def test_fib_sequence(self):
        assert fib(2) == 1
        assert fib(3) == 2
        assert fib(4) == 3
        assert fib(5) == 5
        assert fib(10) == 55

    def test_fib_negative(self):
        with pytest.raises(ValueError):
            fib(-1)

    def test_fib_iterative_performance(self):
        # Test that the iterative implementation works for larger values
        # that would cause stack overflow with recursion
        result = fib(50)
        assert result == 12586269025  # Known 50th Fibonacci number

    def test_fib_performance_large_values(self):
        import time

        # Test performance with various input sizes
        test_values = [100, 200, 500]

        for n in test_values:
            start_time = time.perf_counter()
            result = fib(n)
            end_time = time.perf_counter()

            # Verify the computation completes quickly (should be well under 1 second)
            execution_time = end_time - start_time
            assert execution_time < 1.0, f"fib({n}) took {execution_time:.4f}s, expected < 1.0s"

            # Verify result is positive (basic sanity check)
            assert result > 0, f"fib({n}) should be positive"


class TestAugmentationSeed:
    def test_gen_seed_deterministic(self):
        seed1 = gen_augmentation_seed(0)
        seed2 = gen_augmentation_seed(0)
        assert seed1 == seed2

    def test_gen_seed_different_ids(self):
        seed0 = gen_augmentation_seed(0)
        seed1 = gen_augmentation_seed(1)
        assert seed0 != seed1

    def test_gen_seed_negative_id(self):
        with pytest.raises(ValueError):
            gen_augmentation_seed(-1)

    def test_gen_seed_invalid_depth(self):
        with pytest.raises(ValueError):
            gen_augmentation_seed(0, augmentation_depth=0)

    def test_gen_seed_hex_format(self):
        seed = gen_augmentation_seed(0)
        assert isinstance(seed, str)
        assert len(seed) == 64
        assert all(c in "0123456789abcdef" for c in seed)


class TestAugmentationParams:
    def test_gen_params_deterministic(self):
        params1 = gen_augmentation_params(0)
        params2 = gen_augmentation_params(0)
        assert params1 == params2

    def test_gen_params_different_ids(self):
        params0 = gen_augmentation_params(0)
        params1 = gen_augmentation_params(1)
        assert params0 != params1

    def test_gen_params_negative_id(self):
        with pytest.raises(ValueError):
            gen_augmentation_params(-1)

    def test_gen_params_keys(self):
        params = gen_augmentation_params(0)
        expected_keys = {"rotation", "brightness", "noise", "scale", "contrast", "hash"}
        assert set(params.keys()) == expected_keys

    def test_gen_params_within_range(self):
        config = AugmentationConfig()
        params = gen_augmentation_params(0, config)

        assert config.rotation_range[0] <= params["rotation"] <= config.rotation_range[1]
        assert config.brightness_range[0] <= params["brightness"] <= config.brightness_range[1]
        assert config.noise_range[0] <= params["noise"] <= config.noise_range[1]
        assert config.scale_range[0] <= params["scale"] <= config.scale_range[1]
        assert config.contrast_range[0] <= params["contrast"] <= config.contrast_range[1]

    def test_gen_params_custom_config(self):
        config = AugmentationConfig(rotation_range=(-10, 10))
        for _ in range(100):
            params = gen_augmentation_params(0, config)
            assert -10 <= params["rotation"] <= 10


class TestStatistics:
    def test_compute_statistics_single_value(self):
        params_list = [
            {"rotation": 5, "brightness": 1.0, "noise": 0.05, "scale": 1.0, "contrast": 1.0}
        ]
        stats = compute_statistics(params_list)

        assert stats["rotation"]["mean"] == 5
        assert stats["rotation"]["min"] == 5
        assert stats["rotation"]["max"] == 5
        assert stats["rotation"]["count"] == 1

    def test_compute_statistics_multiple_values(self):
        params_list = [
            {"rotation": 0, "brightness": 1.0, "noise": 0.0, "scale": 1.0, "contrast": 1.0},
            {"rotation": 10, "brightness": 1.0, "noise": 0.0, "scale": 1.0, "contrast": 1.0},
        ]
        stats = compute_statistics(params_list)

        assert stats["rotation"]["mean"] == 5
        assert stats["rotation"]["min"] == 0
        assert stats["rotation"]["max"] == 10
        assert stats["rotation"]["count"] == 2

    def test_compute_statistics_keys(self):
        params_list = [gen_augmentation_params(i) for i in range(5)]
        stats = compute_statistics(params_list)

        expected_keys = {"rotation", "brightness", "noise", "scale", "contrast"}
        assert set(stats.keys()) == expected_keys

    def test_compute_statistics_stdev(self):
        params_list = [
            {"rotation": 0, "brightness": 1.0, "noise": 0.0, "scale": 1.0, "contrast": 1.0},
        ]
        stats = compute_statistics(params_list)
        assert stats["rotation"]["stdev"] == 0.0

    def test_compute_streaming_statistics_basic(self):
        """Test basic streaming statistics computation."""
        # Create a generator
        generator = stream_augmentation_chain(5)
        stats = compute_streaming_statistics(generator)

        # Verify structure
        expected_keys = {"rotation", "brightness", "noise", "scale", "contrast"}
        assert set(stats.keys()) == expected_keys

        for key in expected_keys:
            assert "mean" in stats[key]
            assert "stdev" in stats[key]
            assert "min" in stats[key]
            assert "max" in stats[key]
            assert "count" in stats[key]
            assert stats[key]["count"] == 5

    def test_compute_streaming_statistics_matches_batch(self):
        """Test that streaming statistics match batch statistics exactly."""
        # Generate same data both ways
        params_list = [gen_augmentation_params(i) for i in range(10)]
        batch_stats = compute_statistics(params_list)

        # Create generator with same data
        def param_generator():
            for params in params_list:
                yield from [params]

        streaming_stats = compute_streaming_statistics(param_generator())

        # Compare all statistics (should be nearly identical, allowing for floating-point precision)
        for key in ["rotation", "brightness", "noise", "scale", "contrast"]:
            assert abs(streaming_stats[key]["mean"] - batch_stats[key]["mean"]) < 1e-12
            assert abs(streaming_stats[key]["stdev"] - batch_stats[key]["stdev"]) < 1e-12
            assert abs(streaming_stats[key]["min"] - batch_stats[key]["min"]) < 1e-12
            assert abs(streaming_stats[key]["max"] - batch_stats[key]["max"]) < 1e-12
            assert streaming_stats[key]["count"] == batch_stats[key]["count"]

    def test_compute_streaming_statistics_single_value(self):
        """Test streaming statistics with single value."""

        def single_param_generator():
            yield {"rotation": 5, "brightness": 1.0, "noise": 0.05, "scale": 1.0, "contrast": 1.0}

        stats = compute_streaming_statistics(single_param_generator())

        assert stats["rotation"]["mean"] == 5
        assert stats["rotation"]["min"] == 5
        assert stats["rotation"]["max"] == 5
        assert stats["rotation"]["count"] == 1
        assert stats["rotation"]["stdev"] == 0.0

    def test_compute_streaming_statistics_empty_generator(self):
        """Test streaming statistics with empty generator."""

        def empty_generator():
            return
            yield  # This line never executes

        stats = compute_streaming_statistics(empty_generator())

        for key in ["rotation", "brightness", "noise", "scale", "contrast"]:
            assert stats[key]["mean"] == 0.0
            assert stats[key]["stdev"] == 0.0
            assert stats[key]["min"] == 0.0
            assert stats[key]["max"] == 0.0
            assert stats[key]["count"] == 0

    def test_compute_streaming_statistics_large_dataset(self):
        """Test streaming statistics with larger dataset."""
        # Use streaming generator for large dataset
        generator = stream_augmentation_chain(100)
        streaming_stats = compute_streaming_statistics(generator)

        # Compare with batch computation
        batch_params = generate_augmentation_chain(100)
        batch_stats = compute_statistics(batch_params)

        # Should match exactly
        for key in ["rotation", "brightness", "noise", "scale", "contrast"]:
            assert abs(streaming_stats[key]["mean"] - batch_stats[key]["mean"]) < 1e-12
            assert abs(streaming_stats[key]["stdev"] - batch_stats[key]["stdev"]) < 1e-12
            assert streaming_stats[key]["min"] == batch_stats[key]["min"]
            assert streaming_stats[key]["max"] == batch_stats[key]["max"]
            assert streaming_stats[key]["count"] == batch_stats[key]["count"]

    def test_compute_streaming_statistics_with_config(self):
        """Test streaming statistics with custom configuration."""
        config = AugmentationConfig(rotation_range=(-10, 10))
        generator = stream_augmentation_chain(20, config=config)
        stats = compute_streaming_statistics(generator)

        # Verify statistics are within expected ranges
        assert -10 <= stats["rotation"]["min"] <= 10
        assert -10 <= stats["rotation"]["max"] <= 10
        assert stats["rotation"]["count"] == 20


class TestPersistence:
    def test_save_and_load_augmentation_chain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_augmentations.json"
            params_list = [gen_augmentation_params(i) for i in range(5)]

            save_augmentation_chain(params_list, str(filepath))
            loaded = load_augmentation_chain(str(filepath))

            assert loaded == params_list

    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "test.json"
            params_list = [gen_augmentation_params(0)]

            save_augmentation_chain(params_list, str(filepath))
            assert filepath.exists()

    def test_save_with_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            params_list = [gen_augmentation_params(i) for i in range(5)]

            save_augmentation_chain(params_list, str(filepath), include_stats=True)

            with open(filepath) as f:
                data = json.load(f)

            assert "statistics" in data
            assert "rotation" in data["statistics"]

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_augmentation_chain("nonexistent_file.json")

    def test_save_includes_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            config = AugmentationConfig(rotation_range=(-45, 45))
            params_list = [gen_augmentation_params(0, config)]

            save_augmentation_chain(params_list, str(filepath), config=config)

            with open(filepath) as f:
                data = json.load(f)

            assert data["metadata"]["config"] is not None
            assert data["metadata"]["config"]["rotation_range"] == [-45, 45]


class TestPresets:
    def test_get_preset_mild(self):
        config = get_preset("mild")
        assert config.rotation_range == (-15, 15)
        assert config.brightness_range == (0.9, 1.1)

    def test_get_preset_moderate(self):
        config = get_preset("moderate")
        assert config.rotation_range == (-30, 30)

    def test_get_preset_aggressive(self):
        config = get_preset("aggressive")
        assert config.rotation_range == (-45, 45)

    def test_get_preset_invalid(self):
        with pytest.raises(ValueError):
            get_preset("nonexistent")

    def test_presets_exist(self):
        assert "mild" in PRESETS
        assert "moderate" in PRESETS
        assert "aggressive" in PRESETS


class TestGenerateChain:
    def test_generate_chain_deterministic(self):
        results1 = generate_augmentation_chain(5)
        results2 = generate_augmentation_chain(5)
        assert results1 == results2

    def test_generate_chain_length(self):
        results = generate_augmentation_chain(10)
        assert len(results) == 10

    def test_generate_chain_negative_samples(self):
        with pytest.raises(ValueError):
            generate_augmentation_chain(-1)

    def test_generate_chain_zero_samples(self):
        results = generate_augmentation_chain(0)
        assert results == []

    def test_generate_chain_with_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            results = generate_augmentation_chain(5, save_path=str(filepath))

            assert filepath.exists()
            assert len(results) == 5

    def test_generate_chain_with_preset(self):
        config = get_preset("aggressive")
        results = generate_augmentation_chain(3, config=config)

        for params in results:
            assert -45 <= params["rotation"] <= 45


class TestIntegration:
    def test_full_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "augmentations.json"
            config = get_preset("moderate")

            results = generate_augmentation_chain(10, config, save_path=str(filepath))
            loaded = load_augmentation_chain(str(filepath))
            stats = compute_statistics(loaded)

            assert len(results) == 10
            assert len(loaded) == 10
            assert len(stats) == 5
            assert all(stat["count"] == 10 for stat in stats.values())

    def test_reproducibility_across_runs(self):
        config = get_preset("mild")
        results1 = generate_augmentation_chain(20, config)
        results2 = generate_augmentation_chain(20, config)

        assert results1 == results2

    def test_augmentation_seed_integration_consistency(self):
        """Test that gen_augmentation_seed produces identical results with iterative Fibonacci."""
        # Test various seed IDs and depths to ensure consistency
        test_cases = [
            (0, 1),
            (0, 5),
            (0, 10),
            (0, 20),
            (1, 1),
            (1, 10),
            (1, 25),
            (42, 15),
            (100, 30),
            (999, 50),
        ]

        # Store results for consistency verification
        results = {}

        for seed_id, depth in test_cases:
            # Generate seed multiple times to verify deterministic behavior
            seed1 = gen_augmentation_seed(seed_id, depth)
            seed2 = gen_augmentation_seed(seed_id, depth)

            # Verify deterministic behavior
            assert seed1 == seed2, (
                f"Non-deterministic behavior for seed_id={seed_id}, depth={depth}"
            )

            # Store for cross-verification
            results[(seed_id, depth)] = seed1

            # Verify hash format
            assert len(seed1) == 64, f"Invalid hash length for seed_id={seed_id}, depth={depth}"
            assert all(c in "0123456789abcdef" for c in seed1), (
                f"Invalid hex format for seed_id={seed_id}, depth={depth}"
            )

    def test_augmentation_params_integration_with_fibonacci(self):
        """Test that augmentation parameter generation works correctly with iterative Fibonacci."""
        # Test with various configurations and depths
        configs = [
            AugmentationConfig(augmentation_depth=1),
            AugmentationConfig(augmentation_depth=10),
            AugmentationConfig(augmentation_depth=25),
            AugmentationConfig(augmentation_depth=50),
        ]

        for config in configs:
            for seed_id in [0, 1, 42, 100]:
                # Generate parameters multiple times to verify consistency
                params1 = gen_augmentation_params(seed_id, config)
                params2 = gen_augmentation_params(seed_id, config)

                # Verify deterministic behavior
                assert params1 == params2, (
                    f"Non-deterministic params for seed_id={seed_id}, depth={config.augmentation_depth}"
                )

                # Verify all expected keys are present
                expected_keys = {"rotation", "brightness", "noise", "scale", "contrast", "hash"}
                assert set(params1.keys()) == expected_keys

                # Verify parameters are within expected ranges
                assert config.rotation_range[0] <= params1["rotation"] <= config.rotation_range[1]
                assert (
                    config.brightness_range[0]
                    <= params1["brightness"]
                    <= config.brightness_range[1]
                )
                assert config.noise_range[0] <= params1["noise"] <= config.noise_range[1]
                assert config.scale_range[0] <= params1["scale"] <= config.scale_range[1]
                assert config.contrast_range[0] <= params1["contrast"] <= config.contrast_range[1]

    def test_edge_cases_large_augmentation_depths(self):
        """Test edge cases with large augmentation depths that might have caused stack overflow."""
        # Test depths that would previously cause issues with recursive implementation
        large_depths = [50, 75, 100, 150, 200]

        for depth in large_depths:
            # Test with multiple seed IDs
            for seed_id in [0, 1, 10, 100]:
                try:
                    # This should complete without stack overflow
                    seed = gen_augmentation_seed(seed_id, depth)

                    # Verify valid hash format
                    assert len(seed) == 64
                    assert all(c in "0123456789abcdef" for c in seed)

                    # Test parameter generation with large depth
                    config = AugmentationConfig(augmentation_depth=depth)
                    params = gen_augmentation_params(seed_id, config)

                    # Verify parameters are valid
                    assert isinstance(params["rotation"], float)
                    assert isinstance(params["brightness"], float)
                    assert isinstance(params["noise"], float)
                    assert isinstance(params["scale"], float)
                    assert isinstance(params["contrast"], float)
                    assert isinstance(params["hash"], str)

                except Exception as e:
                    pytest.fail(f"Failed with depth={depth}, seed_id={seed_id}: {e}")

    def test_boundary_conditions_typical_ranges(self):
        """Test boundary conditions for typical augmentation depth ranges (1-50)."""
        # Test boundary values and typical ranges
        test_depths = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

        # Store results to verify uniqueness across different parameters
        all_seeds = set()

        for depth in test_depths:
            for seed_id in range(10):  # Test first 10 seed IDs
                seed = gen_augmentation_seed(seed_id, depth)

                # Verify uniqueness (different seed_id or depth should produce different results)
                key = (seed_id, depth)
                assert seed not in all_seeds or (seed_id == 0 and depth == 1), (
                    f"Duplicate seed found for {key}"
                )
                all_seeds.add(seed)

                # Test parameter generation
                config = AugmentationConfig(augmentation_depth=depth)
                params = gen_augmentation_params(seed_id, config)

                # Verify hash consistency
                assert params["hash"] == seed

                # Verify parameter ranges
                assert -30 <= params["rotation"] <= 30  # Default range
                assert 0.8 <= params["brightness"] <= 1.2  # Default range
                assert 0 <= params["noise"] <= 0.1  # Default range
                assert 0.8 <= params["scale"] <= 1.2  # Default range
                assert 0.7 <= params["contrast"] <= 1.3  # Default range

    def test_maximum_practical_input_values(self):
        """Test behavior with maximum practical input values."""
        # Test with very large seed IDs and reasonable depths
        large_seed_ids = [10000, 100000, 1000000, 2**31 - 1]  # Large but practical values

        for seed_id in large_seed_ids:
            for depth in [1, 10, 25, 50]:
                try:
                    seed = gen_augmentation_seed(seed_id, depth)
                    assert len(seed) == 64
                    assert all(c in "0123456789abcdef" for c in seed)

                    # Test parameter generation
                    params = gen_augmentation_params(seed_id)
                    assert "hash" in params
                    assert len(params["hash"]) == 64

                except Exception as e:
                    pytest.fail(f"Failed with large seed_id={seed_id}, depth={depth}: {e}")

    def test_fibonacci_integration_specific_values(self):
        """Test that specific Fibonacci values are correctly integrated into seed generation."""
        # Test that the Fibonacci sequence is correctly used in seed generation
        # by verifying that different depths produce different results
        seed_id = 42

        seeds_by_depth = {}
        for depth in range(1, 21):  # Test depths 1-20
            seed = gen_augmentation_seed(seed_id, depth)
            seeds_by_depth[depth] = seed

        # Verify that different depths produce different seeds
        unique_seeds = set(seeds_by_depth.values())
        assert len(unique_seeds) == len(seeds_by_depth), (
            "Different depths should produce different seeds"
        )

        # Verify that the same depth always produces the same seed
        for depth in [1, 5, 10, 15, 20]:
            seed1 = gen_augmentation_seed(seed_id, depth)
            seed2 = gen_augmentation_seed(seed_id, depth)
            assert seed1 == seed2, f"Same depth {depth} should produce identical seeds"


class TestStreamingGenerators:
    def test_stream_augmentation_chain_basic(self):
        """Test basic streaming functionality."""
        generator = stream_augmentation_chain(5)
        results = list(generator)

        assert len(results) == 5
        for i, params in enumerate(results):
            expected = gen_augmentation_params(i)
            assert params == expected

    def test_stream_augmentation_chain_deterministic(self):
        """Test that streaming produces deterministic results."""
        results1 = list(stream_augmentation_chain(5))
        results2 = list(stream_augmentation_chain(5))
        assert results1 == results2

    def test_stream_augmentation_chain_matches_batch(self):
        """Test that streaming produces identical results to batch generation."""
        batch_results = generate_augmentation_chain(10)
        stream_results = list(stream_augmentation_chain(10))
        assert stream_results == batch_results

    def test_stream_augmentation_chain_with_config(self):
        """Test streaming with custom configuration."""
        config = AugmentationConfig(rotation_range=(-10, 10))
        results = list(stream_augmentation_chain(5, config=config))

        assert len(results) == 5
        for params in results:
            assert -10 <= params["rotation"] <= 10

    def test_stream_augmentation_chain_with_start_id(self):
        """Test streaming with custom start_id."""
        results = list(stream_augmentation_chain(3, start_id=5))

        expected = [gen_augmentation_params(i) for i in range(5, 8)]
        assert results == expected

    def test_stream_augmentation_chain_chunked(self):
        """Test chunked streaming."""
        chunk_size = 3
        generator = stream_augmentation_chain(10, chunk_size=chunk_size)
        chunks = list(generator)

        # Should have 4 chunks: [3, 3, 3, 1]
        assert len(chunks) == 4
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 3
        assert len(chunks[3]) == 1

        # Flatten and compare with batch results
        flattened = [item for chunk in chunks for item in chunk]
        batch_results = generate_augmentation_chain(10)
        assert flattened == batch_results

    def test_stream_augmentation_chain_chunked_exact_division(self):
        """Test chunked streaming when num_samples divides evenly by chunk_size."""
        chunk_size = 5
        generator = stream_augmentation_chain(10, chunk_size=chunk_size)
        chunks = list(generator)

        assert len(chunks) == 2
        assert len(chunks[0]) == 5
        assert len(chunks[1]) == 5

    def test_stream_augmentation_chain_zero_samples(self):
        """Test streaming with zero samples."""
        results = list(stream_augmentation_chain(0))
        assert results == []

    def test_stream_augmentation_chain_negative_samples(self):
        """Test streaming with negative samples raises error."""
        with pytest.raises(ValueError, match="num_samples must be non-negative"):
            list(stream_augmentation_chain(-1))

    def test_stream_augmentation_chain_negative_start_id(self):
        """Test streaming with negative start_id raises error."""
        with pytest.raises(ValueError, match="start_id must be non-negative"):
            list(stream_augmentation_chain(5, start_id=-1))

    def test_stream_augmentation_chain_invalid_chunk_size(self):
        """Test streaming with invalid chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            list(stream_augmentation_chain(5, chunk_size=0))

    def test_stream_augmentation_range_basic(self):
        """Test basic range streaming functionality."""
        results = list(stream_augmentation_range(2, 7))
        expected = [gen_augmentation_params(i) for i in range(2, 7)]
        assert results == expected

    def test_stream_augmentation_range_with_config(self):
        """Test range streaming with custom configuration."""
        config = AugmentationConfig(rotation_range=(-5, 5))
        results = list(stream_augmentation_range(0, 3, config=config))

        assert len(results) == 3
        for params in results:
            assert -5 <= params["rotation"] <= 5

    def test_stream_augmentation_range_chunked(self):
        """Test chunked range streaming."""
        chunks = list(stream_augmentation_range(1, 8, chunk_size=3))

        assert len(chunks) == 3  # [3, 3, 1]
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 1

        # Verify content
        flattened = [item for chunk in chunks for item in chunk]
        expected = [gen_augmentation_params(i) for i in range(1, 8)]
        assert flattened == expected

    def test_stream_augmentation_range_empty_range(self):
        """Test range streaming with empty range."""
        results = list(stream_augmentation_range(5, 5))
        assert results == []

    def test_stream_augmentation_range_invalid_range(self):
        """Test range streaming with invalid range raises error."""
        with pytest.raises(ValueError, match="end_id must be >= start_id"):
            list(stream_augmentation_range(5, 3))

    def test_stream_augmentation_range_negative_start(self):
        """Test range streaming with negative start_id raises error."""
        with pytest.raises(ValueError, match="start_id must be non-negative"):
            list(stream_augmentation_range(-1, 5))

    def test_streaming_memory_efficiency(self):
        """Test that streaming doesn't consume excessive memory."""
        # This test verifies that we can create a large generator without
        # immediately consuming memory for all results
        large_generator = stream_augmentation_chain(100000)

        # Generator should be created without error
        assert hasattr(large_generator, "__next__")

        # Take just a few items to verify it works
        first_few = []
        for i, params in enumerate(large_generator):
            first_few.append(params)
            if i >= 4:  # Take first 5 items
                break

        assert len(first_few) == 5
        expected = [gen_augmentation_params(i) for i in range(5)]
        assert first_few == expected

    def test_streaming_early_termination(self):
        """Test that streaming supports early termination."""
        generator = stream_augmentation_chain(1000)

        # Take only first 3 items
        results = []
        for i, _params in enumerate(generator):
            results.append(_params)
            if i >= 2:
                break

        assert len(results) == 3
        expected = [gen_augmentation_params(i) for i in range(3)]
        assert results == expected

    def test_streaming_with_presets(self):
        """Test streaming with preset configurations."""
        for preset_name in ["mild", "moderate", "aggressive"]:
            config = get_preset(preset_name)
            stream_results = list(stream_augmentation_chain(5, config=config))
            batch_results = generate_augmentation_chain(5, config=config)
            assert stream_results == batch_results


class TestStreamingIO:
    def test_save_augmentation_stream_basic(self):
        """Test basic streaming save functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "stream_test.json"

            # Create a generator
            generator = stream_augmentation_chain(5)

            # Save using streaming
            save_augmentation_stream(generator, str(filepath), include_stats=False)

            # Verify file exists and has correct structure
            assert filepath.exists()

            with open(filepath) as f:
                data = json.load(f)

            assert "metadata" in data
            assert "augmentations" in data
            assert data["metadata"]["streaming"] is True
            assert len(data["augmentations"]) == 5

    def test_save_augmentation_stream_with_stats(self):
        """Test streaming save with statistics computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "stream_stats_test.json"

            # Create a generator
            generator = stream_augmentation_chain(10)

            # Save with statistics
            save_augmentation_stream(generator, str(filepath), include_stats=True)

            # Verify file structure
            with open(filepath) as f:
                data = json.load(f)

            assert "statistics" in data
            assert "rotation" in data["statistics"]
            assert data["metadata"]["num_samples"] == 10
            assert len(data["augmentations"]) == 10

    def test_save_augmentation_stream_with_config(self):
        """Test streaming save with configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "stream_config_test.json"
            config = AugmentationConfig(rotation_range=(-10, 10))

            # Create a generator with config
            generator = stream_augmentation_chain(3, config=config)

            # Save with config
            save_augmentation_stream(generator, str(filepath), config=config, include_stats=False)

            # Verify config is saved
            with open(filepath) as f:
                data = json.load(f)

            assert data["metadata"]["config"] is not None
            assert data["metadata"]["config"]["rotation_range"] == [-10, 10]

    def test_save_augmentation_stream_creates_directory(self):
        """Test that streaming save creates directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "nested" / "test.json"

            generator = stream_augmentation_chain(2)
            save_augmentation_stream(generator, str(filepath), include_stats=False)

            assert filepath.exists()

    def test_save_augmentation_stream_buffer_size(self):
        """Test streaming save with different buffer sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "buffer_test.json"

            # Test with small buffer size
            generator = stream_augmentation_chain(7)
            save_augmentation_stream(generator, str(filepath), buffer_size=3, include_stats=False)

            # Verify all data is saved correctly
            with open(filepath) as f:
                data = json.load(f)

            assert len(data["augmentations"]) == 7

    def test_load_augmentation_stream_basic(self):
        """Test basic streaming load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "load_test.json"

            # First save some data using regular save
            params_list = [gen_augmentation_params(i) for i in range(5)]
            save_augmentation_chain(params_list, str(filepath))

            # Load using streaming with chunk_size=1 to get individual items
            loaded_params = list(load_augmentation_stream(str(filepath), chunk_size=1))

            assert loaded_params == params_list

    def test_load_augmentation_stream_chunked(self):
        """Test chunked streaming load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "chunked_load_test.json"

            # Save test data
            params_list = [gen_augmentation_params(i) for i in range(7)]
            save_augmentation_chain(params_list, str(filepath))

            # Load in chunks
            chunks = list(load_augmentation_stream(str(filepath), chunk_size=3))

            # Should have 3 chunks: [3, 3, 1]
            assert len(chunks) == 3
            assert len(chunks[0]) == 3
            assert len(chunks[1]) == 3
            assert len(chunks[2]) == 1

            # Verify content
            flattened = [item for chunk in chunks for item in chunk]
            assert flattened == params_list

    def test_load_augmentation_stream_single_items(self):
        """Test loading individual items (chunk_size=1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "single_load_test.json"

            # Save test data
            params_list = [gen_augmentation_params(i) for i in range(3)]
            save_augmentation_chain(params_list, str(filepath))

            # Load individual items
            loaded_items = list(load_augmentation_stream(str(filepath), chunk_size=1))

            assert loaded_items == params_list

    def test_load_augmentation_stream_nonexistent_file(self):
        """Test loading from nonexistent file raises appropriate error."""
        with pytest.raises(FileNotFoundError, match="Augmentation file not found"):
            list(load_augmentation_stream("nonexistent_file.json"))

    def test_load_augmentation_stream_invalid_json(self):
        """Test loading invalid JSON raises appropriate error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "invalid.json"

            # Write invalid JSON
            with open(filepath, "w") as f:
                f.write("{ invalid json")

            with pytest.raises(StreamingIOError, match="Invalid JSON"):
                list(load_augmentation_stream(str(filepath)))

    def test_load_augmentation_stream_invalid_format(self):
        """Test loading file with invalid format raises appropriate error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "invalid_format.json"

            # Write valid JSON but wrong format
            with open(filepath, "w") as f:
                json.dump({"wrong": "format"}, f)

            with pytest.raises(StreamingIOError, match="missing 'augmentations' key"):
                list(load_augmentation_stream(str(filepath)))

    def test_streaming_io_round_trip(self):
        """Test complete round-trip: stream save -> stream load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "round_trip.json"

            # Original data
            original_params = [gen_augmentation_params(i) for i in range(8)]

            # Save using streaming (without stats to test pure streaming)
            def param_generator():
                for params in original_params:
                    yield from [params]

            save_augmentation_stream(param_generator(), str(filepath), include_stats=False)

            # Load using streaming with chunk_size=1 to get individual items
            loaded_params = list(load_augmentation_stream(str(filepath), chunk_size=1))

            assert loaded_params == original_params

    def test_streaming_io_round_trip_with_stats(self):
        """Test round-trip with statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "round_trip_stats.json"

            # Create generator
            generator = stream_augmentation_chain(6)

            # Save with statistics
            save_augmentation_stream(generator, str(filepath), include_stats=True)

            # Load back with chunk_size=1 to get individual items
            loaded_params = list(load_augmentation_stream(str(filepath), chunk_size=1))

            # Compare with batch generation
            expected_params = generate_augmentation_chain(6)
            assert loaded_params == expected_params

    def test_streaming_io_with_config_round_trip(self):
        """Test round-trip with custom configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config_round_trip.json"
            config = AugmentationConfig(rotation_range=(-5, 5), brightness_range=(0.95, 1.05))

            # Save with config
            generator = stream_augmentation_chain(4, config=config)
            save_augmentation_stream(generator, str(filepath), config=config, include_stats=True)

            # Load and verify with chunk_size=1 to get individual items
            loaded_params = list(load_augmentation_stream(str(filepath), chunk_size=1))

            # Verify parameters are within config ranges
            for params in loaded_params:
                assert -5 <= params["rotation"] <= 5
                assert 0.95 <= params["brightness"] <= 1.05

    def test_streaming_save_error_handling(self):
        """Test error handling in streaming save operations."""

        # Test with a generator that raises an error
        def error_generator():
            yield gen_augmentation_params(0)
            raise RuntimeError("Test error during generation")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "error_test.json"

            # This should raise a PartialWriteError due to the generator error
            with pytest.raises(PartialWriteError) as exc_info:
                save_augmentation_stream(error_generator(), str(filepath), include_stats=True)

            # Verify the error contains information about samples written
            assert (
                exc_info.value.samples_written == 0
            )  # No samples should be written due to stats computation failure

    def test_streaming_statistics_computation(self):
        """Test that streaming statistics match batch statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "stats_comparison.json"

            # Generate data both ways
            batch_params = generate_augmentation_chain(20)
            batch_stats = compute_statistics(batch_params)

            # Save using streaming with stats
            generator = stream_augmentation_chain(20)
            save_augmentation_stream(generator, str(filepath), include_stats=True)

            # Load and compare statistics
            with open(filepath) as f:
                data = json.load(f)

            streaming_stats = data["statistics"]

            # Compare statistics (allowing for small floating point differences)
            for key in ["rotation", "brightness", "noise", "scale", "contrast"]:
                assert abs(streaming_stats[key]["mean"] - batch_stats[key]["mean"]) < 1e-10
                assert abs(streaming_stats[key]["stdev"] - batch_stats[key]["stdev"]) < 1e-10
                assert abs(streaming_stats[key]["min"] - batch_stats[key]["min"]) < 1e-10
                assert abs(streaming_stats[key]["max"] - batch_stats[key]["max"]) < 1e-10
                assert streaming_stats[key]["count"] == batch_stats[key]["count"]

    def test_streaming_memory_efficiency_large_dataset(self):
        """Test that streaming I/O can handle large datasets efficiently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "large_dataset.json"

            # Create a large generator (but don't consume it all at once)
            large_generator = stream_augmentation_chain(10000)

            # Save without statistics (pure streaming)
            save_augmentation_stream(large_generator, str(filepath), include_stats=False)

            # Load in chunks to verify it works
            chunk_count = 0
            total_items = 0

            for chunk in load_augmentation_stream(str(filepath), chunk_size=500):
                chunk_count += 1
                total_items += len(chunk)

                # Verify chunk content (spot check)
                if chunk_count == 1:  # Check first chunk
                    expected_first_chunk = [
                        gen_augmentation_params(i) for i in range(min(500, len(chunk)))
                    ]
                    assert chunk[: len(expected_first_chunk)] == expected_first_chunk  # pyright: ignore[reportArgumentType]

                # Don't load everything to keep test fast
                if chunk_count >= 5:  # Just test first few chunks
                    break

            assert chunk_count >= 5
            assert total_items >= 2500  # At least 5 chunks of 500 items each

    def test_streaming_exceptions_hierarchy(self):
        """Test that streaming exceptions have correct hierarchy."""
        # Test exception hierarchy
        assert issubclass(PartialWriteError, StreamingError)
        assert issubclass(StreamingIOError, StreamingError)
        assert issubclass(GeneratorExhaustionError, StreamingError)
        assert issubclass(ResourceCleanupError, StreamingError)
        assert issubclass(StreamingError, Exception)

        # Test PartialWriteError functionality
        error = PartialWriteError("Test error", 42)
        assert error.samples_written == 42
        assert str(error) == "Test error"

    def test_error_handling_resource_cleanup(self):
        """Test resource cleanup functions."""
        import os
        import tempfile

        # Test safe_close_file with valid file handle
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            filepath = f.name
            f.write("test")

        # Open and close file safely
        file_handle = open(filepath)
        safe_close_file(file_handle, filepath)

        # Verify file is closed
        assert file_handle.closed

        # Clean up
        os.unlink(filepath)

        # Test safe_close_file with None handle
        safe_close_file(None, "nonexistent")

        # Test safe_cleanup_generator
        def test_generator():
            yield 1
            yield 2

        gen = test_generator()
        safe_cleanup_generator(gen)

        # Test with None generator
        safe_cleanup_generator(None)

    def test_error_handling_partial_write_recovery(self):
        """Test partial write recovery functionality."""
        import json
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

            # Create a valid complete file
            data = {
                "metadata": {"num_samples": 2, "config": None, "streaming": True},
                "augmentations": [
                    {
                        "rotation": 1.0,
                        "brightness": 1.0,
                        "noise": 0.0,
                        "scale": 1.0,
                        "contrast": 1.0,
                        "hash": "abc",
                    },
                    {
                        "rotation": 2.0,
                        "brightness": 1.1,
                        "noise": 0.1,
                        "scale": 1.1,
                        "contrast": 1.1,
                        "hash": "def",
                    },
                ],
            }
            json.dump(data, f)

        try:
            # Test recovery of complete file
            recovery_info = recover_partial_write(filepath, verbose=True)
            assert recovery_info["recoverable"] is True
            assert recovery_info["samples_found"] == 2
            assert recovery_info["metadata"] is not None
            assert recovery_info["error"] is None

            # Test recovery of nonexistent file
            recovery_info = recover_partial_write("nonexistent.json")
            assert recovery_info["recoverable"] is False
            assert recovery_info["error"] == "File does not exist"

        finally:
            os.unlink(filepath)

    def test_error_handling_streaming_context(self):
        """Test StreamingContext for resource management."""
        import os
        import tempfile

        # Test successful context usage
        with StreamingContext(verbose=True) as context:
            # Create a test generator
            def test_gen():
                yield 1
                yield 2

            gen = test_gen()
            context.register_generator(gen)

            # Create a test file
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                filepath = f.name
                context.register_file_handle(f, filepath)
                f.write("test")

        # Verify cleanup happened
        assert gen.gi_frame is None  # Generator was closed

        # Clean up file
        if os.path.exists(filepath):
            os.unlink(filepath)

    def test_error_handling_resume_streaming_save(self):
        """Test resume streaming save functionality."""
        import os
        import tempfile

        # Create a generator with known data
        def test_generator():
            for i in range(10):
                yield gen_augmentation_params(i)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            # Test resume from sample 5
            resume_streaming_save(
                test_generator(), filepath, resume_from_sample=5, include_stats=False, verbose=True
            )

            # Verify file was created and contains expected samples
            assert os.path.exists(filepath)

            # Load and verify content
            with open(filepath) as f:
                data = json.load(f)

            # Should have 5 samples (from sample 5 to 9)
            assert len(data["augmentations"]) == 5

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_error_handling_with_streaming_context(self):
        """Test with_streaming_context wrapper function."""

        def test_operation(value, streaming_context=None):
            if streaming_context:
                # Register a dummy generator for cleanup testing
                def dummy_gen():
                    yield value

                streaming_context.register_generator(dummy_gen())
            return value * 2

        # Test successful operation
        result = with_streaming_context(test_operation, 5, verbose=True)
        assert result == 10

        # Test operation without streaming_context parameter
        def simple_operation(value):
            return value + 1

        result = with_streaming_context(simple_operation, 5)
        assert result == 6


class TestUtilityFunctions:
    """Test utility and conversion functions."""

    def test_generator_to_list_basic(self):
        """Test basic generator to list conversion."""
        from src.dpa import stream_augmentation_chain

        generator = stream_augmentation_chain(5)
        result = generator_to_list(generator)

        assert len(result) == 5
        assert all(isinstance(item, dict) for item in result)
        assert all("rotation" in item for item in result)

    def test_generator_to_list_with_limit(self):
        """Test generator to list conversion with size limit."""
        from src.dpa import stream_augmentation_chain

        generator = stream_augmentation_chain(10)
        result = generator_to_list(generator, max_items=3)

        assert len(result) == 3

    def test_generator_to_list_limit_larger_than_generator(self):
        """Test generator to list when limit is larger than generator size."""
        from src.dpa import stream_augmentation_chain

        generator = stream_augmentation_chain(3)
        result = generator_to_list(generator, max_items=10)

        assert len(result) == 3

    def test_generator_to_list_negative_limit(self):
        """Test generator to list with negative limit raises error."""
        from src.dpa import stream_augmentation_chain

        generator = stream_augmentation_chain(5)

        with pytest.raises(ValueError, match="max_items must be non-negative"):
            generator_to_list(generator, max_items=-1)

    def test_list_to_generator_basic(self):
        """Test basic list to generator conversion."""
        from src.dpa import generate_augmentation_chain

        params_list = generate_augmentation_chain(5)
        generator = list_to_generator(params_list)

        result = list(generator)
        assert result == params_list

    def test_list_to_generator_empty_list(self):
        """Test list to generator with empty list."""

        generator = list_to_generator([])
        result = list(generator)
        assert result == []

    def test_create_streaming_metadata_basic(self):
        """Test creating streaming metadata with basic parameters."""
        from src.dpa import AugmentationConfig

        config = AugmentationConfig()
        metadata = create_streaming_metadata(
            total_samples=100, config=config, start_id=5, chunk_size=10, buffer_size=500
        )

        assert metadata.total_samples == 100
        assert metadata.config == config
        assert metadata.start_id == 5
        assert metadata.chunk_size == 10
        assert metadata.buffer_size == 500

    def test_create_streaming_metadata_defaults(self):
        """Test creating streaming metadata with default values."""

        metadata = create_streaming_metadata()

        assert metadata.total_samples is None
        assert metadata.config is None
        assert metadata.start_id == 0
        assert metadata.chunk_size is None
        assert metadata.buffer_size == 1000

    def test_create_streaming_metadata_invalid_params(self):
        """Test creating streaming metadata with invalid parameters."""

        with pytest.raises(ValueError, match="start_id must be non-negative"):
            create_streaming_metadata(start_id=-1)

        with pytest.raises(ValueError, match="total_samples must be non-negative"):
            create_streaming_metadata(total_samples=-1)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            create_streaming_metadata(chunk_size=0)

        with pytest.raises(ValueError, match="buffer_size must be positive"):
            create_streaming_metadata(buffer_size=0)

    def test_create_streaming_progress_basic(self):
        """Test creating streaming progress tracker."""
        import time

        start_time = time.time()
        progress = create_streaming_progress(estimated_total=100, start_id=5)

        assert progress.samples_processed == 0
        assert progress.current_sample_id == 5
        assert progress.estimated_total == 100
        assert progress.start_time >= start_time

    def test_create_streaming_progress_invalid_params(self):
        """Test creating streaming progress with invalid parameters."""

        with pytest.raises(ValueError, match="estimated_total must be non-negative"):
            create_streaming_progress(estimated_total=-1)

        with pytest.raises(ValueError, match="start_id must be non-negative"):
            create_streaming_progress(start_id=-1)

    def test_update_streaming_progress(self):
        """Test updating streaming progress tracker."""

        progress = create_streaming_progress(estimated_total=100)
        update_streaming_progress(progress, samples_processed=25, current_sample_id=30)

        assert progress.samples_processed == 25
        assert progress.current_sample_id == 30

    def test_update_streaming_progress_invalid_params(self):
        """Test updating streaming progress with invalid parameters."""

        progress = create_streaming_progress()

        with pytest.raises(ValueError, match="samples_processed must be non-negative"):
            update_streaming_progress(progress, samples_processed=-1, current_sample_id=0)

        with pytest.raises(ValueError, match="current_sample_id must be non-negative"):
            update_streaming_progress(progress, samples_processed=0, current_sample_id=-1)

    def test_streaming_progress_percentage_calculation(self):
        """Test progress percentage calculation."""

        # Test with known total
        progress = create_streaming_progress(estimated_total=100)
        update_streaming_progress(progress, samples_processed=25, current_sample_id=25)

        percentage = progress.progress_percentage()
        assert percentage == 25.0

        # Test with unknown total
        progress_unknown = create_streaming_progress(estimated_total=None)
        percentage_unknown = progress_unknown.progress_percentage()
        assert percentage_unknown is None

    def test_streaming_progress_time_calculations(self):
        """Test elapsed time and ETA calculations."""
        import time

        progress = create_streaming_progress(estimated_total=100)

        # Wait a small amount to ensure elapsed time > 0
        time.sleep(0.01)

        update_streaming_progress(progress, samples_processed=25, current_sample_id=25)

        elapsed = progress.elapsed_time()
        assert elapsed > 0

        eta = progress.estimated_time_remaining()
        assert eta is not None and eta > 0

    def test_estimate_memory_usage_basic(self):
        """Test basic memory usage estimation."""

        usage = estimate_memory_usage(1000)

        assert "parameters" in usage
        assert "statistics" in usage
        assert "json_serialization" in usage
        assert "total" in usage

        assert usage["parameters"] == 1000 * 200  # 200 bytes per param
        assert usage["statistics"] == 1024  # Fixed overhead
        assert usage["total"] > usage["parameters"]

    def test_estimate_memory_usage_without_stats(self):
        """Test memory usage estimation without statistics."""

        usage = estimate_memory_usage(1000, include_statistics=False)

        assert usage["statistics"] == 0

    def test_estimate_memory_usage_invalid_params(self):
        """Test memory usage estimation with invalid parameters."""

        with pytest.raises(ValueError, match="num_samples must be non-negative"):
            estimate_memory_usage(-1)

    def test_get_current_memory_usage(self):
        """Test getting current memory usage."""

        usage = get_current_memory_usage()

        # Should return dict (empty if psutil not available)
        assert isinstance(usage, dict)

        # If psutil is available, should have expected keys
        if usage:  # Only test if psutil is available
            assert "rss" in usage
            assert "vms" in usage
            assert "percent" in usage

    def test_format_memory_size(self):
        """Test memory size formatting."""

        assert format_memory_size(512) == "512 B"
        assert format_memory_size(1536) == "1.5 KB"
        assert format_memory_size(1536 * 1024) == "1.5 MB"
        assert format_memory_size(1536 * 1024 * 1024) == "1.5 GB"

    def test_format_progress_info(self):
        """Test progress information formatting."""
        import time

        # Test with known total
        progress = create_streaming_progress(estimated_total=100)
        time.sleep(0.01)  # Ensure some elapsed time
        update_streaming_progress(progress, samples_processed=25, current_sample_id=25)

        info = format_progress_info(progress)
        assert "25/100" in info
        assert "25.0%" in info
        assert "elapsed:" in info

        # Test with unknown total
        progress_unknown = create_streaming_progress(estimated_total=None)
        update_streaming_progress(progress_unknown, samples_processed=25, current_sample_id=25)

        info_unknown = format_progress_info(progress_unknown)
        assert "25" in info_unknown
        assert "%" not in info_unknown  # No percentage for unknown total


class TestUtilityIntegration:
    """Test integration of utility functions with existing functionality."""

    def test_generator_list_conversion_round_trip(self):
        """Test round-trip conversion between generator and list."""
        from src.dpa import generate_augmentation_chain

        # Start with batch generation
        original_params = generate_augmentation_chain(10)

        # Convert to generator and back to list
        generator = list_to_generator(original_params)
        converted_params = generator_to_list(generator)

        assert converted_params == original_params

    def test_streaming_metadata_with_actual_streaming(self):
        """Test streaming metadata with actual streaming operations."""
        from src.dpa import (
            AugmentationConfig,
            stream_augmentation_chain,
        )

        config = AugmentationConfig(rotation_range=(-10, 10))
        metadata = create_streaming_metadata(total_samples=5, config=config, chunk_size=2)

        # Use metadata configuration with actual streaming
        generator = stream_augmentation_chain(
            num_samples=metadata.total_samples,  # pyright: ignore[reportArgumentType]
            config=metadata.config,
            chunk_size=metadata.chunk_size,
        )

        # Convert to list to verify
        result = generator_to_list(generator)

        # Should have 3 chunks: [2, 2, 1]
        assert len(result) == 3
        assert len(result[0]) == 2
        assert len(result[1]) == 2
        assert len(result[2]) == 1

    def test_progress_tracking_with_streaming(self):
        """Test progress tracking with actual streaming operations."""

        from src.dpa import (
            stream_augmentation_chain,
        )

        progress = create_streaming_progress(estimated_total=10)
        generator = stream_augmentation_chain(10)

        processed_count = 0
        for i, _params in enumerate(generator):
            processed_count += 1
            update_streaming_progress(progress, processed_count, i)

            # Test progress info formatting
            info = format_progress_info(progress)
            assert f"{processed_count}/10" in info

            # Only process a few to keep test fast
            if processed_count >= 3:
                break

        assert progress.samples_processed == 3
        assert progress.current_sample_id == 2


class TestStreamingComprehensive:
    """Comprehensive tests for streaming functionality covering all requirements."""

    def test_streaming_deterministic_behavior_comprehensive(self):
        """Comprehensive test of deterministic behavior across streaming operations."""

        # Test multiple runs of the same streaming operation
        runs = []
        for _ in range(3):
            generator = stream_augmentation_chain(50)
            run_results = list(generator)
            runs.append(run_results)

        # All runs should be identical
        for i in range(1, len(runs)):
            assert runs[i] == runs[0], f"Run {i} differs from run 0"

        # Test chunked streaming determinism
        chunked_runs = []
        for chunk_size in [1, 5, 10, 25]:
            generator = stream_augmentation_chain(50, chunk_size=chunk_size)
            chunks = list(generator)
            # Flatten chunks for comparison
            flattened = [item for chunk in chunks for item in chunk]
            chunked_runs.append(flattened)

        # All chunked runs should produce the same flattened results
        for i in range(1, len(chunked_runs)):
            assert chunked_runs[i] == chunked_runs[0], "Chunked run with chunk_size differs"

        # Chunked results should match non-chunked results
        assert chunked_runs[0] == runs[0], (
            "Chunked streaming produces different results than non-chunked"
        )

    def test_streaming_vs_batch_equivalence_comprehensive(self):
        """Comprehensive test of streaming vs batch equivalence."""

        test_configs = [
            AugmentationConfig(),  # Default
            AugmentationConfig(rotation_range=(-45, 45)),  # Custom rotation
            AugmentationConfig(augmentation_depth=5),  # Different depth
            AugmentationConfig(
                rotation_range=(-90, 90), brightness_range=(0.5, 1.5), augmentation_depth=25
            ),  # Complex config
        ]

        test_sizes = [1, 5, 10, 50, 100]

        for config in test_configs:
            for size in test_sizes:
                # Generate using batch method
                batch_results = generate_augmentation_chain(size, config=config)

                # Generate using streaming method
                stream_generator = stream_augmentation_chain(size, config=config)
                stream_results = list(stream_generator)

                # Results should be identical
                assert len(batch_results) == len(stream_results), f"Length mismatch for size {size}"
                assert batch_results == stream_results, (
                    f"Content mismatch for size {size} with config {config}"
                )

                # Test statistics equivalence
                batch_stats = compute_statistics(batch_results)
                stream_stats = compute_streaming_statistics(list_to_generator(batch_results))

                # Statistics should be nearly identical (allowing for floating-point precision)
                for key in ["rotation", "brightness", "noise", "scale", "contrast"]:
                    assert abs(batch_stats[key]["mean"] - stream_stats[key]["mean"]) < 1e-12
                    assert abs(batch_stats[key]["stdev"] - stream_stats[key]["stdev"]) < 1e-12
                    assert batch_stats[key]["min"] == stream_stats[key]["min"]
                    assert batch_stats[key]["max"] == stream_stats[key]["max"]
                    assert batch_stats[key]["count"] == stream_stats[key]["count"]

    def test_streaming_io_comprehensive_round_trip(self):
        """Comprehensive test of streaming I/O round-trip operations."""

        import os
        import tempfile

        test_cases = [
            {"size": 10, "config": None, "include_stats": True, "buffer_size": 5},
            {
                "size": 50,
                "config": AugmentationConfig(rotation_range=(-45, 45)),
                "include_stats": False,
                "buffer_size": 10,
            },
            {
                "size": 100,
                "config": AugmentationConfig(augmentation_depth=20),
                "include_stats": True,
                "buffer_size": 25,
            },
            {"size": 1, "config": None, "include_stats": True, "buffer_size": 1},  # Single item
        ]

        for case in test_cases:
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, f"test_{case['size']}.json")

                # Generate original data
                original_generator = stream_augmentation_chain(case["size"], config=case["config"])
                original_data = list(original_generator)

                # Save using streaming
                save_generator = list_to_generator(original_data)
                save_augmentation_stream(
                    save_generator,
                    filepath,
                    config=case["config"],
                    include_stats=case["include_stats"],
                    buffer_size=case["buffer_size"],
                )

                # Verify file exists
                assert os.path.exists(filepath)

                # Load using streaming (individual items)
                loaded_individual = list(load_augmentation_stream(filepath, chunk_size=1))

                # Load using streaming (chunks)
                chunk_size = (
                    max(2, case["size"] // 3) or 2
                )  # Ensure chunk_size > 1 to get actual chunks
                loaded_chunks = list(load_augmentation_stream(filepath, chunk_size=chunk_size))
                loaded_from_chunks = [item for chunk in loaded_chunks for item in chunk]

                # All loading methods should produce the same results
                assert loaded_individual == original_data
                assert loaded_from_chunks == original_data

                # Verify file structure
                with open(filepath) as f:
                    file_data = json.load(f)

                assert "metadata" in file_data
                assert "augmentations" in file_data
                assert file_data["metadata"]["streaming"] is True
                assert len(file_data["augmentations"]) == case["size"]

                if case["include_stats"]:
                    assert "statistics" in file_data
                    # Verify statistics are reasonable
                    stats = file_data["statistics"]
                    for key in ["rotation", "brightness", "noise", "scale", "contrast"]:
                        assert key in stats
                        assert "mean" in stats[key]
                        assert "count" in stats[key]
                        assert stats[key]["count"] == case["size"]

    def test_streaming_performance_constant_memory(self):
        """Test that streaming operations maintain constant memory usage."""

        # This test verifies that streaming doesn't accumulate memory
        # by processing increasingly large datasets and checking that
        # memory usage doesn't grow linearly

        try:
            import importlib.util

            psutil_available = importlib.util.find_spec("psutil") is not None
        except ImportError:
            psutil_available = False

        if not psutil_available:
            # Skip this test if psutil is not available
            return

        import gc

        # Force garbage collection before starting
        gc.collect()

        initial_memory = get_current_memory_usage()
        if not initial_memory:
            return  # Skip if we can't get memory info

        memory_readings = []
        dataset_sizes = [100, 500, 1000, 2000]  # Increasing sizes

        for size in dataset_sizes:
            # Force garbage collection before each test
            gc.collect()

            # Process the dataset using streaming
            generator = stream_augmentation_chain(size)
            processed_count = 0

            for params in generator:
                processed_count += 1
                # Process the parameters (simulate work)
                _ = params["rotation"] + params["brightness"]

            assert processed_count == size

            # Measure memory after processing
            current_memory = get_current_memory_usage()
            if current_memory:
                memory_readings.append(current_memory["rss"])

        # Verify that memory usage doesn't grow linearly with dataset size
        if len(memory_readings) >= 2:
            # Memory growth should be bounded
            max_memory = max(memory_readings)
            min_memory = min(memory_readings)

            # Allow for some variation but not linear growth
            growth_ratio = max_memory / min_memory if min_memory > 0 else 1
            assert growth_ratio < 1.5, (
                f"Memory grew by {growth_ratio:.2f}x across dataset sizes, indicating potential memory leak"
            )

    def test_streaming_early_termination_memory_cleanup(self):
        """Test that early termination of streaming operations cleans up properly."""

        # Create a large generator but only consume part of it
        large_generator = stream_augmentation_chain(10000)

        # Consume only first 100 items
        consumed = []
        for i, params in enumerate(large_generator):
            consumed.append(params)
            if i >= 99:  # Stop after 100 items
                break

        assert len(consumed) == 100

        # Verify that the generator can be properly cleaned up
        try:
            safe_cleanup_generator(large_generator)
        except ResourceCleanupError:
            # This is acceptable - cleanup might fail but shouldn't crash
            pass

        # Verify that we can create new generators without issues
        new_generator = stream_augmentation_chain(50)
        new_results = list(new_generator)
        assert len(new_results) == 50

    def test_streaming_statistics_numerical_stability(self):
        """Test numerical stability of streaming statistics computation."""

        # Test with values that might cause numerical instability
        def create_test_generator(values):
            for i, value in enumerate(values):
                yield {
                    "rotation": value,
                    "brightness": 1.0,
                    "noise": 0.0,
                    "scale": 1.0,
                    "contrast": 1.0,
                    "hash": f"test{i}",
                }

        # Test case 1: Very large values
        large_values = [1e6, 1e6 + 1, 1e6 + 2, 1e6 + 3, 1e6 + 4]
        large_generator = create_test_generator(large_values)
        large_stats = compute_streaming_statistics(large_generator)

        # Compare with batch computation
        large_params = [
            {
                "rotation": v,
                "brightness": 1.0,
                "noise": 0.0,
                "scale": 1.0,
                "contrast": 1.0,
                "hash": f"test{i}",
            }
            for i, v in enumerate(large_values)
        ]
        large_batch_stats = compute_statistics(large_params)

        # Should be nearly identical despite large values
        assert abs(large_stats["rotation"]["mean"] - large_batch_stats["rotation"]["mean"]) < 1e-6
        assert abs(large_stats["rotation"]["stdev"] - large_batch_stats["rotation"]["stdev"]) < 1e-6

        # Test case 2: Very small values
        small_values = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6]
        small_generator = create_test_generator(small_values)
        small_stats = compute_streaming_statistics(small_generator)

        small_params = [
            {
                "rotation": v,
                "brightness": 1.0,
                "noise": 0.0,
                "scale": 1.0,
                "contrast": 1.0,
                "hash": f"test{i}",
            }
            for i, v in enumerate(small_values)
        ]
        small_batch_stats = compute_statistics(small_params)

        # Should be nearly identical despite small values
        assert abs(small_stats["rotation"]["mean"] - small_batch_stats["rotation"]["mean"]) < 1e-12
        assert (
            abs(small_stats["rotation"]["stdev"] - small_batch_stats["rotation"]["stdev"]) < 1e-12
        )

    def test_streaming_range_equivalence(self):
        """Test equivalence between stream_augmentation_chain and stream_augmentation_range."""

        # Test various ranges
        test_cases = [
            (0, 10),  # Start from 0
            (5, 15),  # Start from middle
            (100, 150),  # Large start
            (0, 1),  # Single item
            (42, 42),  # Empty range
        ]

        for start_id, end_id in test_cases:
            num_samples = end_id - start_id

            if num_samples <= 0:
                # Test empty ranges
                chain_results = list(stream_augmentation_chain(0, start_id=start_id))
                range_results = list(stream_augmentation_range(start_id, end_id))
                assert chain_results == range_results == []
            else:
                # Test non-empty ranges
                chain_results = list(stream_augmentation_chain(num_samples, start_id=start_id))
                range_results = list(stream_augmentation_range(start_id, end_id))

                assert len(chain_results) == len(range_results) == num_samples
                assert chain_results == range_results

                # Verify that the sample IDs are correct by checking hash consistency
                for i, params in enumerate(chain_results):
                    expected_sample_id = start_id + i
                    expected_params = gen_augmentation_params(expected_sample_id)
                    assert params == expected_params

    def test_streaming_with_all_presets(self):
        """Test streaming functionality with all preset configurations."""

        import os
        import tempfile

        for preset_name in ["mild", "moderate", "aggressive"]:
            config = get_preset(preset_name)

            # Test streaming generation
            stream_results = list(stream_augmentation_chain(20, config=config))
            batch_results = generate_augmentation_chain(20, config=config)

            # Results should be identical
            assert stream_results == batch_results

            # Test streaming I/O
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, f"{preset_name}_test.json")

                # Save using streaming
                generator = stream_augmentation_chain(15, config=config)
                save_augmentation_stream(generator, filepath, config=config, include_stats=True)

                # Load using streaming
                loaded = list(load_augmentation_stream(filepath, chunk_size=1))

                # Verify results match expected
                expected = generate_augmentation_chain(15, config=config)
                assert loaded == expected

                # Verify statistics are computed correctly
                with open(filepath) as f:
                    file_data = json.load(f)

                assert "statistics" in file_data
                stats = file_data["statistics"]

                # Verify statistics are within preset ranges
                if preset_name == "mild":
                    assert -15 <= stats["rotation"]["min"] <= stats["rotation"]["max"] <= 15
                elif preset_name == "moderate":
                    assert -30 <= stats["rotation"]["min"] <= stats["rotation"]["max"] <= 30
                elif preset_name == "aggressive":
                    assert -45 <= stats["rotation"]["min"] <= stats["rotation"]["max"] <= 45

    def test_streaming_large_dataset_simulation(self):
        """Test streaming with simulated large dataset scenarios."""

        # Test various large dataset sizes without actually consuming excessive memory
        large_sizes = [10000, 50000, 100000]

        for size in large_sizes:
            # Create generator but don't consume all at once
            generator = stream_augmentation_chain(size)

            # Sample from the generator to verify it works
            sample_size = min(100, size)
            sample = []

            for i, params in enumerate(generator):
                sample.append(params)
                if i >= sample_size - 1:
                    break

            assert len(sample) == sample_size

            # Verify the samples are correct
            expected_sample = [gen_augmentation_params(i) for i in range(sample_size)]
            assert sample == expected_sample

            # Test chunked streaming with large datasets
            chunked_generator = stream_augmentation_chain(size, chunk_size=1000)

            # Take first few chunks
            chunks_taken = 0
            total_items = 0

            for chunk in chunked_generator:
                chunks_taken += 1
                total_items += len(chunk)

                # Verify chunk size (except possibly the last chunk)
                if chunks_taken == 1:  # First chunk should be full size
                    assert len(chunk) == 1000

                if chunks_taken >= 5:  # Take only first 5 chunks
                    break

            assert chunks_taken == 5
            assert total_items == 5000  # 5 chunks * 1000 items each

    def test_streaming_comprehensive_integration(self):
        """Comprehensive integration test combining all streaming features."""

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test scenario: Generate large dataset, save with streaming, load with streaming,
            # compute statistics, and verify everything matches batch operations

            dataset_size = 200
            config = AugmentationConfig(
                rotation_range=(-60, 60), brightness_range=(0.6, 1.4), augmentation_depth=15
            )

            # Step 1: Generate using batch method for reference
            batch_reference = generate_augmentation_chain(dataset_size, config=config)
            batch_stats = compute_statistics(batch_reference)

            # Step 2: Generate using streaming method
            stream_generator = stream_augmentation_chain(dataset_size, config=config)
            stream_results = list(stream_generator)

            # Verify streaming matches batch
            assert stream_results == batch_reference

            # Step 3: Save using streaming I/O
            filepath = os.path.join(tmpdir, "comprehensive_test.json")
            save_generator = list_to_generator(stream_results)
            save_augmentation_stream(
                save_generator, filepath, config=config, include_stats=True, buffer_size=50
            )

            # Step 4: Load using streaming I/O (individual items)
            loaded_individual = list(load_augmentation_stream(filepath, chunk_size=1))
            assert loaded_individual == batch_reference

            # Step 5: Load using streaming I/O (chunks)
            loaded_chunks = list(load_augmentation_stream(filepath, chunk_size=25))
            loaded_from_chunks = [item for chunk in loaded_chunks for item in chunk]
            assert loaded_from_chunks == batch_reference

            # Step 6: Compute streaming statistics
            stats_generator = list_to_generator(stream_results)
            streaming_stats = compute_streaming_statistics(stats_generator)

            # Verify statistics match
            for key in ["rotation", "brightness", "noise", "scale", "contrast"]:
                assert abs(streaming_stats[key]["mean"] - batch_stats[key]["mean"]) < 1e-12
                assert abs(streaming_stats[key]["stdev"] - batch_stats[key]["stdev"]) < 1e-12
                assert streaming_stats[key]["min"] == batch_stats[key]["min"]
                assert streaming_stats[key]["max"] == batch_stats[key]["max"]
                assert streaming_stats[key]["count"] == batch_stats[key]["count"]

            # Step 7: Verify file statistics match computed statistics
            with open(filepath) as f:
                file_data = json.load(f)

            file_stats = file_data["statistics"]
            for key in ["rotation", "brightness", "noise", "scale", "contrast"]:
                assert abs(file_stats[key]["mean"] - batch_stats[key]["mean"]) < 1e-12
                assert abs(file_stats[key]["stdev"] - batch_stats[key]["stdev"]) < 1e-12
                assert file_stats[key]["min"] == batch_stats[key]["min"]
                assert file_stats[key]["max"] == batch_stats[key]["max"]
                assert file_stats[key]["count"] == batch_stats[key]["count"]

            # Step 8: Test range streaming equivalence
            range_results = list(stream_augmentation_range(0, dataset_size, config=config))
            assert range_results == batch_reference

            # Step 9: Test chunked streaming equivalence
            chunked_generator = stream_augmentation_chain(
                dataset_size, config=config, chunk_size=40
            )
            chunked_results = [item for chunk in chunked_generator for item in chunk]
            assert chunked_results == batch_reference

            # Step 10: Test utility functions
            generator_copy = list_to_generator(batch_reference)
            converted_back = generator_to_list(generator_copy)
            assert converted_back == batch_reference
