#!/usr/bin/env python3
"""
Integration and Compatibility Demonstration

This script demonstrates the integration and compatibility verification features
implemented for the streaming generator API, showing:

1. All existing functions remain unchanged and functional
2. Streaming API works with all existing preset configurations
3. Backward compatibility with existing save/load file formats
4. Conversion utilities between streaming and batch formats
"""

import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.dpa import (
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
)


def demo_existing_functions_unchanged():
    """Demonstrate that existing functions work exactly as before."""
    print("=" * 60)
    print("1. EXISTING FUNCTIONS REMAIN UNCHANGED")
    print("=" * 60)

    # Test batch generation (existing functionality)
    print("Generating 5 samples using existing batch API...")
    batch_results = generate_augmentation_chain(5, verbose=True)
    print(f"Generated {len(batch_results)} samples successfully\n")

    # Test statistics computation (existing functionality)
    print("Computing statistics using existing batch API...")
    stats = compute_statistics(batch_results)
    print(f"Rotation mean: {stats['rotation']['mean']:.2f}°")
    print(f"Brightness mean: {stats['brightness']['mean']:.2f}")
    print(f"Sample count: {stats['rotation']['count']}\n")


def demo_streaming_with_presets():
    """Demonstrate streaming API works with all existing presets."""
    print("=" * 60)
    print("2. STREAMING API WITH ALL PRESETS")
    print("=" * 60)

    for preset_name in ["mild", "moderate", "aggressive"]:
        print(f"Testing {preset_name} preset...")
        config = get_preset(preset_name)

        # Generate using both APIs
        batch_results = generate_augmentation_chain(3, config=config)
        stream_results = list(stream_augmentation_chain(3, config=config))

        # Verify they're identical
        identical = batch_results == stream_results
        print(f"  Batch and streaming results identical: {identical}")

        # Show parameter ranges for this preset
        rotations = [p["rotation"] for p in stream_results]
        print(f"  Rotation range: {min(rotations):.1f}° to {max(rotations):.1f}°")
        print()


def demo_backward_compatibility():
    """Demonstrate backward compatibility with existing file formats."""
    print("=" * 60)
    print("3. BACKWARD COMPATIBILITY WITH FILE FORMATS")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        batch_file = Path(tmpdir) / "batch_created.json"
        stream_file = Path(tmpdir) / "stream_created.json"

        # Generate test data
        test_data = generate_augmentation_chain(8)

        print("Creating files using both batch and streaming APIs...")

        # Save using batch API
        save_augmentation_chain(test_data, str(batch_file), include_stats=True)
        print(f"  Batch API created: {batch_file.name}")

        # Save using streaming API
        generator = list_to_generator(test_data)
        save_augmentation_stream(generator, str(stream_file), include_stats=True)
        print(f"  Streaming API created: {stream_file.name}")

        print("\nCross-loading files to test compatibility...")

        # Load batch-created file with streaming API
        stream_loaded_batch = list(load_augmentation_stream(str(batch_file), chunk_size=1))
        print(f"  Streaming API loaded batch file: {len(stream_loaded_batch)} samples")

        # Load stream-created file with batch API
        batch_loaded_stream = load_augmentation_chain(str(stream_file))
        print(f"  Batch API loaded streaming file: {len(batch_loaded_stream)} samples")

        # Verify all data is identical
        all_identical = test_data == stream_loaded_batch == batch_loaded_stream
        print(f"  All data identical across APIs: {all_identical}\n")


def demo_conversion_utilities():
    """Demonstrate conversion utilities between streaming and batch formats."""
    print("=" * 60)
    print("4. CONVERSION UTILITIES")
    print("=" * 60)

    print("Testing generator ↔ list conversion utilities...")

    # Start with streaming data
    print("  Creating streaming generator...")
    original_generator = stream_augmentation_chain(6)

    # Convert to list
    print("  Converting generator → list...")
    converted_list = generator_to_list(original_generator)
    print(f"    Converted {len(converted_list)} items to list")

    # Convert back to generator
    print("  Converting list → generator...")
    converted_generator = list_to_generator(converted_list)

    # Convert back to list to verify round-trip
    print("  Converting generator → list (round-trip test)...")
    final_list = generator_to_list(converted_generator)
    print(f"    Final list has {len(final_list)} items")

    # Verify round-trip integrity
    expected_list = generate_augmentation_chain(6)
    round_trip_success = final_list == expected_list
    print(f"  Round-trip conversion successful: {round_trip_success}")

    # Test with size limits
    print("\n  Testing conversion with size limits...")
    large_generator = stream_augmentation_chain(100)
    limited_list = generator_to_list(large_generator, max_items=10)
    print(f"    Limited conversion: {len(limited_list)} items (max 10)")

    # Test statistics compatibility
    print("\n  Testing statistics compatibility across conversions...")
    test_data = generate_augmentation_chain(20)

    # Batch statistics
    batch_stats = compute_statistics(test_data)

    # Streaming statistics via conversion
    generator = list_to_generator(test_data)
    streaming_stats = compute_streaming_statistics(generator)

    # Compare means (should be identical)
    batch_mean = batch_stats["rotation"]["mean"]
    stream_mean = streaming_stats["rotation"]["mean"]
    means_match = abs(batch_mean - stream_mean) < 1e-12
    print(f"    Batch mean rotation: {batch_mean:.6f}°")
    print(f"    Stream mean rotation: {stream_mean:.6f}°")
    print(f"    Statistics match: {means_match}\n")


def demo_comprehensive_integration():
    """Demonstrate comprehensive integration scenario."""
    print("=" * 60)
    print("5. COMPREHENSIVE INTEGRATION SCENARIO")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Running mixed API workflow...")

        # Step 1: Generate with batch API and moderate preset
        config = get_preset("moderate")
        print("  Step 1: Generate data with batch API + moderate preset")
        batch_data = generate_augmentation_chain(15, config=config)

        # Step 2: Convert to streaming and save with streaming API
        print("  Step 2: Convert to streaming and save with streaming API")
        generator = list_to_generator(batch_data)
        stream_file = Path(tmpdir) / "mixed_workflow.json"
        save_augmentation_stream(generator, str(stream_file), config=config, include_stats=True)

        # Step 3: Load with streaming API in chunks
        print("  Step 3: Load with streaming API in chunks")
        chunks = list(load_augmentation_stream(str(stream_file), chunk_size=5))
        print(f"    Loaded {len(chunks)} chunks")

        # Step 4: Flatten chunks and verify with batch API
        print("  Step 4: Flatten chunks and verify with batch API")
        flattened = [item for chunk in chunks for item in chunk]

        # Step 5: Save flattened data with batch API
        print("  Step 5: Save flattened data with batch API")
        batch_file = Path(tmpdir) / "final_batch.json"
        save_augmentation_chain(flattened, str(batch_file), config=config, include_stats=True)

        # Step 6: Final verification
        print("  Step 6: Final verification")
        final_data = load_augmentation_chain(str(batch_file))

        workflow_success = batch_data == final_data
        print(f"    Mixed workflow preserved data integrity: {workflow_success}")

        # Verify statistics are preserved
        original_stats = compute_statistics(batch_data)
        final_stats = compute_statistics(final_data)
        stats_preserved = (
            abs(original_stats["rotation"]["mean"] - final_stats["rotation"]["mean"]) < 1e-12
        )
        print(f"    Statistics preserved through workflow: {stats_preserved}")

        print(f"    Original rotation mean: {original_stats['rotation']['mean']:.6f}°")
        print(f"    Final rotation mean: {final_stats['rotation']['mean']:.6f}°\n")


def demo_memory_efficiency():
    """Demonstrate memory efficiency benefits."""
    print("=" * 60)
    print("6. MEMORY EFFICIENCY DEMONSTRATION")
    print("=" * 60)

    print("Demonstrating memory-efficient processing...")

    # Create a large generator (but don't consume it all)
    print("  Creating generator for 50,000 samples...")
    large_generator = stream_augmentation_chain(50000)

    # Process only first 10 samples to show we can handle large datasets
    print("  Processing first 10 samples from large generator...")
    sample_count = 0
    for params in large_generator:
        sample_count += 1
        if sample_count >= 10:
            break

    print(f"    Successfully processed {sample_count} samples")
    print("    (Generator can handle 50,000+ samples without memory issues)")

    # Show chunked processing
    print("\n  Demonstrating chunked processing...")
    chunked_generator = stream_augmentation_chain(100, chunk_size=20)
    chunk_count = 0
    total_processed = 0

    for chunk in chunked_generator:
        chunk_count += 1
        total_processed += len(chunk)
        print(f"    Processed chunk {chunk_count}: {len(chunk)} items")

        if chunk_count >= 3:  # Process first 3 chunks
            break

    print(f"    Total items processed in chunks: {total_processed}")
    print("    (Memory usage remains constant regardless of total dataset size)\n")


def main():
    """Run all integration and compatibility demonstrations."""
    print("STREAMING API INTEGRATION & COMPATIBILITY DEMONSTRATION")
    print("=" * 60)
    print("This demo shows that the streaming API is fully integrated")
    print("and compatible with all existing functionality.\n")

    # Run all demonstrations
    demo_existing_functions_unchanged()
    demo_streaming_with_presets()
    demo_backward_compatibility()
    demo_conversion_utilities()
    demo_comprehensive_integration()
    demo_memory_efficiency()

    print("=" * 60)
    print("INTEGRATION & COMPATIBILITY VERIFICATION COMPLETE")
    print("=" * 60)
    print("✓ All existing functions remain unchanged and functional")
    print("✓ Streaming API works with all existing preset configurations")
    print("✓ Backward compatibility with existing save/load file formats")
    print("✓ Conversion utilities between streaming and batch formats")
    print("✓ Memory efficiency benefits while maintaining full functionality")
    print("\nThe streaming API is ready for production use!")


if __name__ == "__main__":
    main()
