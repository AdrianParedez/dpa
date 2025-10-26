"""
Integration and compatibility verification tests for streaming API.

This module tests that:
1. All existing functions remain unchanged and functional
2. Streaming API works with all existing preset configurations
3. Backward compatibility with existing save/load file formats is maintained
4. Conversion utilities between streaming and batch formats work correctly
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.dpa import (
    PRESETS,
    AugmentationConfig,
    compute_statistics,
    compute_streaming_statistics,
    generate_augmentation_chain,
    generator_to_list,
    get_preset,
    list_to_generator,
    load_augmentation_chain,
    load_augmentation_stream,
    save_augmentation_chain,
    save_augmentation_stream,
    stream_augmentation_chain,
    stream_augmentation_range,
)


class TestExistingFunctionCompatibility:
    """Test that all existing functions remain unchanged and functional."""

    def test_existing_batch_functions_unchanged(self):
        """Test that existing batch functions work exactly as before."""
        # Test generate_augmentation_chain
        batch_results = generate_augmentation_chain(10)
        assert len(batch_results) == 10
        assert all(isinstance(params, dict) for params in batch_results)
        assert all("rotation" in params for params in batch_results)

        # Test deterministic behavior
        batch_results2 = generate_augmentation_chain(10)
        assert batch_results == batch_results2

    def test_existing_save_load_functions_unchanged(self):
        """Test that existing save/load functions work exactly as before."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_existing.json"

            # Generate test data
            params_list = generate_augmentation_chain(15)

            # Test save function
            save_augmentation_chain(params_list, str(filepath), include_stats=True)
            assert filepath.exists()

            # Test load function
            loaded_params = load_augmentation_chain(str(filepath))
            assert loaded_params == params_list

            # Verify file structure is unchanged
            with open(filepath) as f:
                data = json.load(f)

            assert "metadata" in data
            assert "augmentations" in data
            assert "statistics" in data
            assert data["metadata"]["num_samples"] == 15

    def test_existing_statistics_functions_unchanged(self):
        """Test that existing statistics functions work exactly as before."""
        params_list = generate_augmentation_chain(20)
        stats = compute_statistics(params_list)

        # Verify structure
        expected_keys = {"rotation", "brightness", "noise", "scale", "contrast"}
        assert set(stats.keys()) == expected_keys

        for key in expected_keys:
            assert "mean" in stats[key]
            assert "stdev" in stats[key]
            assert "min" in stats[key]
            assert "max" in stats[key]
            assert "count" in stats[key]
            assert stats[key]["count"] == 20

    def test_existing_preset_functions_unchanged(self):
        """Test that existing preset functions work exactly as before."""
        # Test all presets exist
        assert "mild" in PRESETS
        assert "moderate" in PRESETS
        assert "aggressive" in PRESETS

        # Test get_preset function
        mild_config = get_preset("mild")
        assert mild_config.rotation_range == (-15, 15)
        assert mild_config.brightness_range == (0.9, 1.1)

        moderate_config = get_preset("moderate")
        assert moderate_config.rotation_range == (-30, 30)

        aggressive_config = get_preset("aggressive")
        assert aggressive_config.rotation_range == (-45, 45)

        # Test invalid preset
        with pytest.raises(ValueError):
            get_preset("nonexistent")

    def test_existing_config_validation_unchanged(self):
        """Test that AugmentationConfig validation works exactly as before."""
        # Test valid config
        config = AugmentationConfig(rotation_range=(-45, 45))
        assert config.rotation_range == (-45, 45)

        # Test invalid range
        with pytest.raises(ValueError):
            AugmentationConfig(rotation_range=(30, 10))

        # Test invalid depth
        with pytest.raises(ValueError):
            AugmentationConfig(augmentation_depth=0)


class TestStreamingWithPresets:
    """Test streaming API with all existing preset configurations."""

    def test_streaming_with_mild_preset(self):
        """Test streaming API with mild preset."""
        config = get_preset("mild")

        # Test streaming generation
        stream_results = list(stream_augmentation_chain(25, config=config))
        batch_results = generate_augmentation_chain(25, config=config)

        # Results should be identical
        assert stream_results == batch_results

        # Verify parameters are within mild preset ranges
        for params in stream_results:
            assert -15 <= params["rotation"] <= 15
            assert 0.9 <= params["brightness"] <= 1.1

    def test_streaming_with_moderate_preset(self):
        """Test streaming API with moderate preset."""
        config = get_preset("moderate")

        # Test streaming generation
        stream_results = list(stream_augmentation_chain(30, config=config))
        batch_results = generate_augmentation_chain(30, config=config)

        # Results should be identical
        assert stream_results == batch_results

        # Verify parameters are within moderate preset ranges
        for params in stream_results:
            assert -30 <= params["rotation"] <= 30
            assert 0.8 <= params["brightness"] <= 1.2

    def test_streaming_with_aggressive_preset(self):
        """Test streaming API with aggressive preset."""
        config = get_preset("aggressive")

        # Test streaming generation
        stream_results = list(stream_augmentation_chain(20, config=config))
        batch_results = generate_augmentation_chain(20, config=config)

        # Results should be identical
        assert stream_results == batch_results

        # Verify parameters are within aggressive preset ranges
        for params in stream_results:
            assert -45 <= params["rotation"] <= 45
            assert 0.7 <= params["brightness"] <= 1.3

    def test_streaming_chunked_with_all_presets(self):
        """Test chunked streaming with all presets."""
        for preset_name in ["mild", "moderate", "aggressive"]:
            config = get_preset(preset_name)

            # Test chunked streaming
            chunks = list(stream_augmentation_chain(15, config=config, chunk_size=5))
            assert len(chunks) == 3
            assert all(len(chunk) == 5 for chunk in chunks)

            # Flatten and compare with batch
            flattened = [item for chunk in chunks for item in chunk]
            batch_results = generate_augmentation_chain(15, config=config)
            assert flattened == batch_results

    def test_streaming_range_with_all_presets(self):
        """Test range streaming with all presets."""
        for preset_name in ["mild", "moderate", "aggressive"]:
            config = get_preset(preset_name)

            # Test range streaming
            range_results = list(stream_augmentation_range(5, 20, config=config))

            # Generate expected results manually since generate_augmentation_chain
            # doesn't have start_id parameter
            expected_results = []
            for i in range(5, 20):
                from src.dpa import gen_augmentation_params

                expected_results.append(gen_augmentation_params(i, config))

            assert range_results == expected_results

    def test_streaming_statistics_with_all_presets(self):
        """Test streaming statistics computation with all presets."""
        for preset_name in ["mild", "moderate", "aggressive"]:
            config = get_preset(preset_name)

            # Generate data both ways
            batch_params = generate_augmentation_chain(50, config=config)
            batch_stats = compute_statistics(batch_params)

            # Compute streaming statistics
            generator = list_to_generator(batch_params)
            streaming_stats = compute_streaming_statistics(generator)

            # Statistics should match exactly
            for key in ["rotation", "brightness", "noise", "scale", "contrast"]:
                assert abs(streaming_stats[key]["mean"] - batch_stats[key]["mean"]) < 1e-12
                assert abs(streaming_stats[key]["stdev"] - batch_stats[key]["stdev"]) < 1e-12
                assert streaming_stats[key]["min"] == batch_stats[key]["min"]
                assert streaming_stats[key]["max"] == batch_stats[key]["max"]
                assert streaming_stats[key]["count"] == batch_stats[key]["count"]


class TestBackwardCompatibility:
    """Test backward compatibility with existing save/load file formats."""

    def test_streaming_load_existing_batch_files(self):
        """Test that streaming load can read files created by batch save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "batch_created.json"

            # Create file using existing batch save
            params_list = generate_augmentation_chain(12)
            save_augmentation_chain(params_list, str(filepath), include_stats=True)

            # Load using streaming load (individual items)
            loaded_streaming = list(load_augmentation_stream(str(filepath), chunk_size=1))

            # Should match original data
            assert loaded_streaming == params_list

            # Load using streaming load (chunks)
            loaded_chunks = list(load_augmentation_stream(str(filepath), chunk_size=4))
            flattened_chunks = [item for chunk in loaded_chunks for item in chunk]
            assert flattened_chunks == params_list

    def test_batch_load_streaming_created_files(self):
        """Test that batch load can read files created by streaming save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "streaming_created.json"

            # Create file using streaming save
            generator = stream_augmentation_chain(18)
            save_augmentation_stream(generator, str(filepath), include_stats=True)

            # Load using existing batch load
            loaded_batch = load_augmentation_chain(str(filepath))

            # Should match expected data
            expected_data = generate_augmentation_chain(18)
            assert loaded_batch == expected_data

    def test_file_format_compatibility_with_config(self):
        """Test file format compatibility when config is included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_preset("moderate")

            # Test batch save -> streaming load
            batch_filepath = Path(tmpdir) / "batch_with_config.json"
            params_list = generate_augmentation_chain(10, config=config)
            save_augmentation_chain(
                params_list, str(batch_filepath), config=config, include_stats=True
            )

            loaded_streaming = list(load_augmentation_stream(str(batch_filepath), chunk_size=1))
            assert loaded_streaming == params_list

            # Test streaming save -> batch load
            streaming_filepath = Path(tmpdir) / "streaming_with_config.json"
            generator = stream_augmentation_chain(10, config=config)
            save_augmentation_stream(
                generator, str(streaming_filepath), config=config, include_stats=True
            )

            loaded_batch = load_augmentation_chain(str(streaming_filepath))
            assert loaded_batch == params_list

    def test_file_format_compatibility_without_stats(self):
        """Test file format compatibility when statistics are not included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test batch save without stats -> streaming load
            batch_filepath = Path(tmpdir) / "batch_no_stats.json"
            params_list = generate_augmentation_chain(8)
            save_augmentation_chain(params_list, str(batch_filepath), include_stats=False)

            loaded_streaming = list(load_augmentation_stream(str(batch_filepath), chunk_size=1))
            assert loaded_streaming == params_list

            # Test streaming save without stats -> batch load
            streaming_filepath = Path(tmpdir) / "streaming_no_stats.json"
            generator = stream_augmentation_chain(8)
            save_augmentation_stream(generator, str(streaming_filepath), include_stats=False)

            loaded_batch = load_augmentation_chain(str(streaming_filepath))
            assert loaded_batch == params_list

    def test_metadata_compatibility(self):
        """Test that metadata format is compatible between batch and streaming."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AugmentationConfig(rotation_range=(-20, 20))

            # Create files using both methods
            batch_filepath = Path(tmpdir) / "batch_metadata.json"
            streaming_filepath = Path(tmpdir) / "streaming_metadata.json"

            params_list = generate_augmentation_chain(5, config=config)
            save_augmentation_chain(
                params_list, str(batch_filepath), config=config, include_stats=True
            )

            generator = stream_augmentation_chain(5, config=config)
            save_augmentation_stream(
                generator, str(streaming_filepath), config=config, include_stats=True
            )

            # Compare metadata structures
            with open(batch_filepath) as f:
                batch_data = json.load(f)

            with open(streaming_filepath) as f:
                streaming_data = json.load(f)

            # Both should have same basic metadata structure
            assert "metadata" in batch_data
            assert "metadata" in streaming_data
            assert (
                batch_data["metadata"]["num_samples"] == streaming_data["metadata"]["num_samples"]
            )
            assert batch_data["metadata"]["config"] == streaming_data["metadata"]["config"]

            # Streaming files should have additional streaming flag
            assert streaming_data["metadata"]["streaming"] is True
            assert (
                "streaming" not in batch_data["metadata"]
                or batch_data["metadata"]["streaming"] is None
            )

    def test_statistics_format_compatibility(self):
        """Test that statistics format is compatible between batch and streaming."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with statistics using both methods
            batch_filepath = Path(tmpdir) / "batch_stats.json"
            streaming_filepath = Path(tmpdir) / "streaming_stats.json"

            params_list = generate_augmentation_chain(25)
            save_augmentation_chain(params_list, str(batch_filepath), include_stats=True)

            generator = stream_augmentation_chain(25)
            save_augmentation_stream(generator, str(streaming_filepath), include_stats=True)

            # Load and compare statistics
            with open(batch_filepath) as f:
                batch_data = json.load(f)

            with open(streaming_filepath) as f:
                streaming_data = json.load(f)

            batch_stats = batch_data["statistics"]
            streaming_stats = streaming_data["statistics"]

            # Statistics should be nearly identical
            for key in ["rotation", "brightness", "noise", "scale", "contrast"]:
                assert abs(batch_stats[key]["mean"] - streaming_stats[key]["mean"]) < 1e-10
                assert abs(batch_stats[key]["stdev"] - streaming_stats[key]["stdev"]) < 1e-10
                assert batch_stats[key]["min"] == streaming_stats[key]["min"]
                assert batch_stats[key]["max"] == streaming_stats[key]["max"]
                assert batch_stats[key]["count"] == streaming_stats[key]["count"]


class TestConversionUtilities:
    """Test conversion utilities between streaming and batch formats."""

    def test_generator_to_list_basic_conversion(self):
        """Test basic generator to list conversion."""
        # Create a generator
        generator = stream_augmentation_chain(10)

        # Convert to list
        result_list = generator_to_list(generator)

        # Should match batch generation
        expected_list = generate_augmentation_chain(10)
        assert result_list == expected_list

    def test_generator_to_list_with_limit(self):
        """Test generator to list conversion with size limit."""
        # Create a large generator
        generator = stream_augmentation_chain(100)

        # Convert with limit
        result_list = generator_to_list(generator, max_items=15)

        # Should have only 15 items
        assert len(result_list) == 15

        # Should match first 15 items from batch generation
        expected_list = generate_augmentation_chain(15)
        assert result_list == expected_list

    def test_generator_to_list_limit_larger_than_generator(self):
        """Test generator to list when limit is larger than generator."""
        generator = stream_augmentation_chain(5)
        result_list = generator_to_list(generator, max_items=20)

        assert len(result_list) == 5
        expected_list = generate_augmentation_chain(5)
        assert result_list == expected_list

    def test_generator_to_list_error_handling(self):
        """Test generator to list error handling."""
        generator = stream_augmentation_chain(5)

        # Test negative limit
        with pytest.raises(ValueError, match="max_items must be non-negative"):
            generator_to_list(generator, max_items=-1)

    def test_list_to_generator_basic_conversion(self):
        """Test basic list to generator conversion."""
        # Create a list
        params_list = generate_augmentation_chain(12)

        # Convert to generator
        generator = list_to_generator(params_list)

        # Convert back to list
        result_list = list(generator)

        # Should match original
        assert result_list == params_list

    def test_list_to_generator_empty_list(self):
        """Test list to generator with empty list."""
        generator = list_to_generator([])
        result_list = list(generator)
        assert result_list == []

    def test_round_trip_conversion(self):
        """Test round-trip conversion: list -> generator -> list."""
        # Start with batch data
        original_list = generate_augmentation_chain(20)

        # Convert to generator
        generator = list_to_generator(original_list)

        # Convert back to list
        converted_list = generator_to_list(generator)

        # Should match original
        assert converted_list == original_list

    def test_conversion_with_streaming_operations(self):
        """Test conversions work with streaming operations."""
        # Generate streaming data
        stream_generator = stream_augmentation_chain(15)

        # Convert to list
        stream_as_list = generator_to_list(stream_generator)

        # Use with batch statistics
        batch_stats = compute_statistics(stream_as_list)

        # Convert back to generator and use with streaming statistics
        list_as_generator = list_to_generator(stream_as_list)
        streaming_stats = compute_streaming_statistics(list_as_generator)

        # Statistics should match
        for key in ["rotation", "brightness", "noise", "scale", "contrast"]:
            assert abs(batch_stats[key]["mean"] - streaming_stats[key]["mean"]) < 1e-12
            assert abs(batch_stats[key]["stdev"] - streaming_stats[key]["stdev"]) < 1e-12
            assert batch_stats[key]["min"] == streaming_stats[key]["min"]
            assert batch_stats[key]["max"] == streaming_stats[key]["max"]
            assert batch_stats[key]["count"] == streaming_stats[key]["count"]

    def test_conversion_with_chunked_streaming(self):
        """Test conversions work with chunked streaming."""
        # Create chunked generator
        chunked_generator = stream_augmentation_chain(20, chunk_size=5)

        # Convert chunks to flat list
        all_chunks = list(chunked_generator)
        flattened = [item for chunk in all_chunks for item in chunk]

        # Convert to generator and back
        generator = list_to_generator(flattened)
        converted_back = generator_to_list(generator)

        # Should match expected batch data
        expected = generate_augmentation_chain(20)
        assert converted_back == expected

    def test_conversion_with_presets(self):
        """Test conversions work with preset configurations."""
        for preset_name in ["mild", "moderate", "aggressive"]:
            config = get_preset(preset_name)

            # Generate with streaming
            stream_generator = stream_augmentation_chain(10, config=config)

            # Convert to list
            stream_as_list = generator_to_list(stream_generator)

            # Should match batch generation with same preset
            batch_with_preset = generate_augmentation_chain(10, config=config)
            assert stream_as_list == batch_with_preset

            # Convert back to generator
            list_as_generator = list_to_generator(stream_as_list)
            converted_back = generator_to_list(list_as_generator)

            # Should still match
            assert converted_back == batch_with_preset

    def test_conversion_utilities_with_io_operations(self):
        """Test conversion utilities work with I/O operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "conversion_test.json"

            # Generate streaming data
            stream_generator = stream_augmentation_chain(18)

            # Convert to list and save using batch save
            stream_as_list = generator_to_list(stream_generator)
            save_augmentation_chain(stream_as_list, str(filepath), include_stats=True)

            # Load using batch load
            loaded_batch = load_augmentation_chain(str(filepath))

            # Convert to generator and save using streaming save
            filepath2 = Path(tmpdir) / "conversion_test2.json"
            loaded_as_generator = list_to_generator(loaded_batch)
            save_augmentation_stream(loaded_as_generator, str(filepath2), include_stats=True)

            # Load using streaming load
            loaded_streaming = list(load_augmentation_stream(str(filepath2), chunk_size=1))

            # All should match
            expected = generate_augmentation_chain(18)
            assert stream_as_list == expected
            assert loaded_batch == expected
            assert loaded_streaming == expected


class TestIntegrationScenarios:
    """Test comprehensive integration scenarios combining all features."""

    def test_mixed_api_workflow(self):
        """Test workflow mixing batch and streaming APIs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_preset("moderate")

            # Step 1: Generate using batch API
            batch_data = generate_augmentation_chain(30, config=config)

            # Step 2: Convert to streaming and save
            generator = list_to_generator(batch_data)
            filepath1 = Path(tmpdir) / "mixed_workflow1.json"
            save_augmentation_stream(generator, str(filepath1), config=config, include_stats=True)

            # Step 3: Load using streaming API
            loaded_streaming = list(load_augmentation_stream(str(filepath1), chunk_size=1))

            # Step 4: Save using batch API
            filepath2 = Path(tmpdir) / "mixed_workflow2.json"
            save_augmentation_chain(
                loaded_streaming, str(filepath2), config=config, include_stats=True
            )

            # Step 5: Load using batch API
            final_result = load_augmentation_chain(str(filepath2))

            # All steps should preserve data integrity
            assert final_result == batch_data

    def test_preset_compatibility_comprehensive(self):
        """Test comprehensive preset compatibility across all APIs."""
        for preset_name in ["mild", "moderate", "aggressive"]:
            config = get_preset(preset_name)

            with tempfile.TemporaryDirectory() as tmpdir:
                # Generate using all methods
                batch_result = generate_augmentation_chain(25, config=config)
                stream_result = list(stream_augmentation_chain(25, config=config))
                range_result = list(stream_augmentation_range(0, 25, config=config))

                # All should be identical
                assert stream_result == batch_result
                assert range_result == batch_result

                # Test I/O compatibility
                batch_file = Path(tmpdir) / f"{preset_name}_batch.json"
                stream_file = Path(tmpdir) / f"{preset_name}_stream.json"

                # Save using both methods
                save_augmentation_chain(
                    batch_result, str(batch_file), config=config, include_stats=True
                )

                generator = list_to_generator(batch_result)
                save_augmentation_stream(
                    generator, str(stream_file), config=config, include_stats=True
                )

                # Cross-load and verify
                batch_loaded_from_stream = load_augmentation_chain(str(stream_file))
                stream_loaded_from_batch = list(
                    load_augmentation_stream(str(batch_file), chunk_size=1)
                )

                assert batch_loaded_from_stream == batch_result
                assert stream_loaded_from_batch == batch_result

    def test_large_dataset_compatibility(self):
        """Test compatibility with larger datasets."""
        # Test with a moderately large dataset
        dataset_size = 500

        # Generate using both APIs
        batch_data = generate_augmentation_chain(dataset_size)

        # Convert to streaming and process in chunks
        generator = list_to_generator(batch_data)

        # Process in chunks and reassemble
        chunk_size = 50
        processed_chunks = []
        current_chunk = []

        for params in generator:
            current_chunk.append(params)
            if len(current_chunk) >= chunk_size:
                processed_chunks.append(current_chunk)
                current_chunk = []

        if current_chunk:  # Add remaining items
            processed_chunks.append(current_chunk)

        # Reassemble
        reassembled = [item for chunk in processed_chunks for item in chunk]

        # Should match original
        assert reassembled == batch_data
        assert len(reassembled) == dataset_size

    def test_error_handling_compatibility(self):
        """Test that error handling is compatible between APIs."""
        # Test invalid configurations work the same way
        with pytest.raises(ValueError):
            AugmentationConfig(rotation_range=(30, 10))

        # Test that both APIs handle invalid parameters the same way
        with pytest.raises(ValueError):
            generate_augmentation_chain(-1)  # Negative samples

        with pytest.raises(ValueError):
            list(stream_augmentation_chain(-1))  # Negative samples

        # Test invalid start_id for streaming
        with pytest.raises(ValueError):
            list(stream_augmentation_chain(5, start_id=-1))

        # Test invalid chunk_size for streaming
        with pytest.raises(ValueError):
            list(stream_augmentation_chain(5, chunk_size=0))

    def test_memory_efficiency_vs_functionality_tradeoff(self):
        """Test that streaming provides memory efficiency without losing functionality."""
        # Generate same data using both methods
        batch_data = generate_augmentation_chain(100)
        stream_data = list(stream_augmentation_chain(100))

        # Functionality should be identical
        assert stream_data == batch_data

        # Statistics should be identical
        batch_stats = compute_statistics(batch_data)
        stream_stats = compute_streaming_statistics(list_to_generator(batch_data))

        for key in ["rotation", "brightness", "noise", "scale", "contrast"]:
            assert abs(batch_stats[key]["mean"] - stream_stats[key]["mean"]) < 1e-12
            assert abs(batch_stats[key]["stdev"] - stream_stats[key]["stdev"]) < 1e-12

        # Test that streaming can handle larger datasets that batch might struggle with
        # (This is more of a conceptual test since we can't easily measure memory in unit tests)
        large_generator = stream_augmentation_chain(10000)

        # Should be able to process first few items without loading everything
        first_few = []
        for i, params in enumerate(large_generator):
            first_few.append(params)
            if i >= 9:  # Take first 10 items
                break

        assert len(first_few) == 10
        expected_first_few = generate_augmentation_chain(10)
        assert first_few == expected_first_few
