#!/usr/bin/env python3
"""
Distributed Training Example: Range Splitting

This example demonstrates how to split sample ranges across multiple ranks
in distributed training to ensure no overlap and proper data distribution.

Requirements addressed: 4.1 (range splitting utilities)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distributed import DistributedRangeSplitter, stream_distributed_augmentation_chain
from src.dpa import get_preset


def demo_basic_range_splitting():
    """Demonstrate basic range splitting across ranks."""
    print("=== Basic Range Splitting Demo ===\n")

    total_samples = 100
    world_size = 4

    splitter = DistributedRangeSplitter(total_samples, world_size)

    print(f"Splitting {total_samples} samples across {world_size} ranks:")
    print(f"Samples per rank: {total_samples // world_size} (with remainder handled)\n")

    total_assigned = 0
    for rank in range(world_size):
        start, end = splitter.get_rank_range(rank)
        samples_for_rank = end - start
        total_assigned += samples_for_rank

        print(f"Rank {rank}: samples {start:3d} to {end - 1:3d} ({samples_for_rank:2d} samples)")

    print(f"\nTotal samples assigned: {total_assigned}")
    print(f"Validation passed: {splitter.validate_ranges()}")


def demo_uneven_distribution():
    """Demonstrate handling of uneven sample distribution."""
    print("\n=== Uneven Distribution Demo ===\n")

    # Test cases where samples don't divide evenly
    test_cases = [
        (23, 4),  # 23 samples, 4 ranks
        (100, 7),  # 100 samples, 7 ranks
        (5, 8),  # Fewer samples than ranks
    ]

    for total_samples, world_size in test_cases:
        print(f"Case: {total_samples} samples, {world_size} ranks")

        if total_samples < world_size:
            print(f"  Warning: More ranks ({world_size}) than samples ({total_samples})")

        splitter = DistributedRangeSplitter(total_samples, world_size)

        for rank in range(world_size):
            start, end = splitter.get_rank_range(rank)
            samples_for_rank = end - start

            if samples_for_rank > 0:
                print(
                    f"  Rank {rank}: samples {start:2d} to {end - 1:2d} ({samples_for_rank} samples)"
                )
            else:
                print(f"  Rank {rank}: no samples assigned")

        print(f"  Validation: {splitter.validate_ranges()}\n")


def demo_range_verification():
    """Demonstrate range validation and overlap checking."""
    print("=== Range Verification Demo ===\n")

    total_samples = 50
    world_size = 3

    splitter = DistributedRangeSplitter(total_samples, world_size)
    all_ranges = splitter.get_all_ranges()

    print(f"All ranges for {total_samples} samples across {world_size} ranks:")

    assigned_samples = set()
    for rank, (start, end) in enumerate(all_ranges):
        rank_samples = set(range(start, end))

        # Check for overlaps
        overlap = assigned_samples.intersection(rank_samples)
        if overlap:
            print(f"  Rank {rank}: {start}-{end - 1} (OVERLAP DETECTED: {overlap})")
        else:
            print(f"  Rank {rank}: {start}-{end - 1} (✓ no overlap)")

        assigned_samples.update(rank_samples)

    # Verify all samples are covered
    expected_samples = set(range(total_samples))
    missing = expected_samples - assigned_samples
    extra = assigned_samples - expected_samples

    print("\nVerification:")
    print(f"  Expected samples: 0-{total_samples - 1}")
    print(f"  Assigned samples: {len(assigned_samples)}")
    print(f"  Missing samples: {missing if missing else 'None'}")
    print(f"  Extra samples: {extra if extra else 'None'}")
    print(f"  Validation result: {splitter.validate_ranges()}")


def demo_streaming_with_ranges():
    """Demonstrate streaming augmentation parameters with range splitting."""
    print("\n=== Streaming with Range Splitting Demo ===\n")

    total_samples = 20
    world_size = 3
    config = get_preset("mild")

    print(f"Streaming {total_samples} samples across {world_size} ranks")
    print("Each rank processes only its assigned sample range:\n")

    for rank in range(world_size):
        print(f"--- Rank {rank} Stream ---")

        # Get the parameter stream for this rank
        param_stream = stream_distributed_augmentation_chain(
            num_samples=total_samples, rank=rank, world_size=world_size, config=config
        )

        # Process first few parameters to demonstrate
        params_processed = 0
        for params in param_stream:
            if params_processed < 3:  # Show first 3 for demo
                sample_id = params.get("sample_id", "unknown")
                print(f"  Sample {sample_id}: rotation={params['rotation']:.2f}")
            params_processed += 1

        print(f"  Total parameters processed: {params_processed}")
        print()


def demo_edge_cases():
    """Demonstrate edge cases in range splitting."""
    print("=== Edge Cases Demo ===\n")

    edge_cases = [
        (1, 1),  # Single sample, single rank
        (10, 1),  # Multiple samples, single rank
        (0, 4),  # No samples
        (3, 10),  # More ranks than samples
    ]

    for total_samples, world_size in edge_cases:
        print(f"Edge case: {total_samples} samples, {world_size} ranks")

        try:
            splitter = DistributedRangeSplitter(total_samples, world_size)

            active_ranks = 0
            for rank in range(world_size):
                start, end = splitter.get_rank_range(rank)
                if end > start:
                    active_ranks += 1
                    print(f"  Rank {rank}: {start}-{end - 1}")
                else:
                    print(f"  Rank {rank}: inactive (no samples)")

            print(f"  Active ranks: {active_ranks}/{world_size}")
            print(f"  Validation: {splitter.validate_ranges()}")

        except Exception as e:
            print(f"  Error: {e}")

        print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Distributed Training: Range Splitting")
    print("=" * 70)

    demo_basic_range_splitting()
    demo_uneven_distribution()
    demo_range_verification()
    demo_streaming_with_ranges()
    demo_edge_cases()

    print("\n" + "=" * 70)
    print("Demo completed! Range splitting ensures no overlap across ranks.")
    print("=" * 70 + "\n")
