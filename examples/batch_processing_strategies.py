#!/usr/bin/env python3
"""
Batch Processing Example: Different Batching Strategies

This example demonstrates various batch processing strategies available in the
DPA library, including sequential, round-robin, memory-optimized, and adaptive
batching approaches.

Requirements addressed: 2.1, 2.4 (batch processing strategies)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.batch import BatchConfig, BatchProcessor, BatchStrategy
from src.distributed import stream_distributed_augmentation_chain
from src.dpa import get_preset, stream_augmentation_chain


def demo_sequential_batching():
    """Demonstrate sequential batching strategy."""
    print("=== Sequential Batching Demo ===\n")

    num_samples = 20
    batch_size = 5
    config = get_preset("mild")

    # Create batch processor with sequential strategy
    batch_config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=batch_size)
    processor = BatchProcessor(BatchStrategy.SEQUENTIAL, batch_config)

    print(f"Processing {num_samples} samples in batches of {batch_size}")
    print("Sequential strategy: samples are grouped in order (0-4, 5-9, 10-14, etc.)\n")

    # Create parameter stream
    param_stream = stream_augmentation_chain(num_samples, config)

    # Process in batches
    batch_num = 0
    for batch in processor.process_stream(param_stream):
        batch_num += 1
        sample_ids = [params.get("sample_id", i) for i, params in enumerate(batch)]
        rotations = [params["rotation"] for params in batch]

        print(f"Batch {batch_num}: {len(batch)} samples")
        print(f"  Sample IDs: {sample_ids}")
        print(f"  Rotations: {[f'{r:.2f}' for r in rotations]}")
        print()


def demo_round_robin_batching():
    """Demonstrate round-robin batching strategy."""
    print("=== Round-Robin Batching Demo ===\n")

    num_samples = 15
    batch_size = 4
    config = get_preset("moderate")

    # Create batch processor with round-robin strategy
    batch_config = BatchConfig(strategy=BatchStrategy.ROUND_ROBIN, batch_size=batch_size)
    processor = BatchProcessor(BatchStrategy.ROUND_ROBIN, batch_config)

    print(f"Processing {num_samples} samples in batches of {batch_size}")
    print("Round-robin strategy: samples are distributed evenly across batches\n")

    # Create parameter stream
    param_stream = stream_augmentation_chain(num_samples, config)

    # Process in batches
    batch_num = 0
    for batch in processor.process_stream(param_stream):
        batch_num += 1
        sample_ids = [params.get("sample_id", i) for i, params in enumerate(batch)]
        brightness_values = [params["brightness"] for params in batch]

        print(f"Batch {batch_num}: {len(batch)} samples")
        print(f"  Sample IDs: {sample_ids}")
        print(f"  Brightness: {[f'{b:.3f}' for b in brightness_values]}")
        print()


def demo_memory_optimized_batching():
    """Demonstrate memory-optimized batching strategy."""
    print("=== Memory-Optimized Batching Demo ===\n")

    num_samples = 25
    config = get_preset("aggressive")

    # Create batch processor with memory-optimized strategy
    batch_config = BatchConfig(
        strategy=BatchStrategy.MEMORY_OPTIMIZED,
        max_memory_mb=50,  # Low limit for demonstration
        min_batch_size=2,
    )
    processor = BatchProcessor(BatchStrategy.MEMORY_OPTIMIZED, batch_config)

    print(f"Processing {num_samples} samples with memory optimization")
    print(f"Memory limit: {batch_config.max_memory_mb}MB")
    print("Memory-optimized strategy: batch size adjusts based on available memory\n")

    # Create parameter stream
    param_stream = stream_augmentation_chain(num_samples, config)

    # Process in batches with memory monitoring
    batch_num = 0
    total_processed = 0

    for batch in processor.process_with_memory_monitoring(param_stream):
        batch_num += 1
        total_processed += len(batch)

        # Get memory info from the processor
        memory_info = processor._memory_batcher.monitor_memory_usage()

        print(f"Batch {batch_num}: {len(batch)} samples")
        print(
            f"  Memory usage: {memory_info['used_mb']:.1f}MB / {memory_info['available_mb']:.1f}MB available"
        )
        print(f"  Memory percentage: {memory_info['percent_used']:.1f}%")

        # Show sample parameters
        if len(batch) > 0:
            first_sample = batch[0]
            print(
                f"  First sample: rotation={first_sample['rotation']:.2f}, "
                f"scale={first_sample['scale']:.3f}"
            )
        print()

    print(f"Total samples processed: {total_processed}/{num_samples}")


def demo_adaptive_batching():
    """Demonstrate adaptive batching strategy."""
    print("=== Adaptive Batching Demo ===\n")

    num_samples = 30
    config = get_preset("moderate")

    # Create batch processor with adaptive strategy
    batch_config = BatchConfig(
        strategy=BatchStrategy.ADAPTIVE,
        batch_size=8,  # Starting batch size
        adaptive_sizing=True,
    )
    processor = BatchProcessor(BatchStrategy.ADAPTIVE, batch_config)

    print(f"Processing {num_samples} samples with adaptive batching")
    print(f"Starting batch size: {batch_config.batch_size}")
    print("Adaptive strategy: batch size adjusts based on processing performance\n")

    # Create parameter stream
    param_stream = stream_augmentation_chain(num_samples, config)

    # Process in batches
    batch_num = 0
    processing_times = []

    for batch in processor.process_stream(param_stream):
        batch_num += 1

        # Simulate processing time (varies by batch size)
        start_time = time.time()
        time.sleep(0.01 * len(batch))  # Simulate work proportional to batch size
        processing_time = time.time() - start_time
        processing_times.append(processing_time)

        print(f"Batch {batch_num}: {len(batch)} samples in {processing_time:.3f}s")
        print(f"  Throughput: {len(batch) / processing_time:.1f} samples/sec")

        # Show adaptation
        if batch_num > 1:
            if processing_time < processing_times[-2]:
                print("  → Performance improved, may increase batch size")
            else:
                print("  → Performance degraded, may decrease batch size")
        print()


def demo_batch_strategy_comparison():
    """Compare different batching strategies side by side."""
    print("=== Batch Strategy Comparison ===\n")

    num_samples = 16
    batch_size = 4
    config = get_preset("mild")

    strategies = [
        BatchStrategy.SEQUENTIAL,
        BatchStrategy.ROUND_ROBIN,
        BatchStrategy.MEMORY_OPTIMIZED,
    ]

    print(f"Comparing strategies with {num_samples} samples, batch size {batch_size}:\n")

    for strategy in strategies:
        print(f"--- {strategy.value.upper()} STRATEGY ---")

        batch_config = BatchConfig(strategy=strategy, batch_size=batch_size, max_memory_mb=100)
        processor = BatchProcessor(strategy, batch_config)

        # Create fresh parameter stream for each strategy
        param_stream = stream_augmentation_chain(num_samples, config)

        batch_num = 0
        for batch in processor.process_stream(param_stream):
            batch_num += 1
            sample_ids = [params.get("sample_id", i) for i, params in enumerate(batch)]
            print(f"  Batch {batch_num}: samples {sample_ids}")

        print()


def demo_distributed_batch_processing():
    """Demonstrate batch processing with distributed training."""
    print("=== Distributed Batch Processing Demo ===\n")

    total_samples = 20
    world_size = 2
    rank = 0  # Simulate rank 0
    batch_size = 3
    config = get_preset("moderate")

    print(f"Distributed setup: {total_samples} total samples, world_size={world_size}, rank={rank}")
    print(f"Batch processing rank {rank}'s assigned samples in batches of {batch_size}\n")

    # Create batch processor
    batch_config = BatchConfig(strategy=BatchStrategy.SEQUENTIAL, batch_size=batch_size)
    processor = BatchProcessor(BatchStrategy.SEQUENTIAL, batch_config)

    # Create distributed parameter stream for this rank
    param_stream = stream_distributed_augmentation_chain(
        num_samples=total_samples, rank=rank, world_size=world_size, config=config
    )

    # Process in batches
    batch_num = 0
    total_processed = 0

    for batch in processor.process_stream(param_stream):
        batch_num += 1
        total_processed += len(batch)

        sample_ids = [params.get("sample_id", "unknown") for params in batch]
        rotations = [params["rotation"] for params in batch]

        print(f"Rank {rank} Batch {batch_num}: {len(batch)} samples")
        print(f"  Sample IDs: {sample_ids}")
        print(f"  Rotations: {[f'{r:.2f}' for r in rotations]}")
        print()

    print(f"Rank {rank} processed {total_processed} samples total")

    # Show what rank 1 would process
    print("\n--- What Rank 1 would process ---")
    rank1_stream = stream_distributed_augmentation_chain(
        num_samples=total_samples, rank=1, world_size=world_size, config=config
    )

    rank1_samples = list(rank1_stream)
    print(f"Rank 1 would process {len(rank1_samples)} samples:")
    for _i, params in enumerate(rank1_samples[:5]):  # Show first 5
        sample_id = params.get("sample_id", "unknown")
        print(f"  Sample {sample_id}: rotation={params['rotation']:.2f}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Batch Processing: Different Batching Strategies")
    print("=" * 70)

    demo_sequential_batching()
    demo_round_robin_batching()
    demo_memory_optimized_batching()
    demo_adaptive_batching()
    demo_batch_strategy_comparison()
    demo_distributed_batch_processing()

    print("\n" + "=" * 70)
    print("Demo completed! Different strategies optimize for different use cases.")
    print("=" * 70 + "\n")
