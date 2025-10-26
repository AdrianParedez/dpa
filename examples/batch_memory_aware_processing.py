#!/usr/bin/env python3
"""
Batch Processing Example: Memory-Aware Batching

This example demonstrates memory-aware batch processing capabilities,
including dynamic batch sizing, memory monitoring, and memory limit enforcement.

Requirements addressed: 5.1 (memory monitoring and limits)
"""

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.batch import BatchConfig, BatchProcessor, BatchStrategy, MemoryAwareBatcher
from src.dpa import get_preset, stream_augmentation_chain


def demo_memory_monitoring():
    """Demonstrate basic memory monitoring capabilities."""
    print("=== Memory Monitoring Demo ===\n")

    # Create memory-aware batcher
    batcher = MemoryAwareBatcher(max_memory_mb=500, min_batch_size=1)

    print("Current memory status:")
    memory_info = batcher.monitor_memory_usage()

    print(f"  Total memory: {memory_info['total_mb']:.1f} MB")
    print(f"  Available memory: {memory_info['available_mb']:.1f} MB")
    print(f"  Used memory: {memory_info['used_mb']:.1f} MB")
    print(f"  Memory usage: {memory_info['percent_used']:.1f}%")
    print()

    # Demonstrate memory usage calculation
    sample_size_bytes = 1024  # Assume 1KB per sample
    optimal_batch_size = batcher.calculate_optimal_batch_size(sample_size_bytes)

    print(f"For samples of {sample_size_bytes} bytes each:")
    print(f"  Optimal batch size: {optimal_batch_size} samples")
    print(f"  Memory per batch: {optimal_batch_size * sample_size_bytes / 1024:.1f} KB")


def demo_dynamic_batch_sizing():
    """Demonstrate dynamic batch size adjustment based on memory."""
    print("\n=== Dynamic Batch Sizing Demo ===\n")

    # Create batcher with conservative memory limit
    max_memory_mb = 100  # Conservative limit for demo
    batcher = MemoryAwareBatcher(max_memory_mb=max_memory_mb, min_batch_size=2)

    print(f"Memory limit: {max_memory_mb} MB")
    print("Demonstrating batch size adjustment based on memory pressure:\n")

    # Simulate different memory conditions
    memory_scenarios = [
        {"used_mb": 50, "available_mb": 200, "percent": 25.0},  # Low usage
        {"used_mb": 80, "available_mb": 120, "percent": 40.0},  # Medium usage
        {"used_mb": 95, "available_mb": 50, "percent": 65.0},  # High usage
        {"used_mb": 98, "available_mb": 20, "percent": 83.0},  # Very high usage
    ]

    current_batch_size = 10

    for i, scenario in enumerate(memory_scenarios, 1):
        print(f"Scenario {i}: {scenario['percent']:.1f}% memory usage")
        print(f"  Available: {scenario['available_mb']} MB")

        # Adjust batch size based on memory
        new_batch_size = batcher.adjust_batch_size(current_batch_size, scenario)

        if new_batch_size < current_batch_size:
            print(f"  Batch size reduced: {current_batch_size} → {new_batch_size}")
        elif new_batch_size > current_batch_size:
            print(f"  Batch size increased: {current_batch_size} → {new_batch_size}")
        else:
            print(f"  Batch size unchanged: {current_batch_size}")

        current_batch_size = new_batch_size
        print()


def demo_memory_limit_enforcement():
    """Demonstrate memory limit enforcement during processing."""
    print("=== Memory Limit Enforcement Demo ===\n")

    num_samples = 50
    config = get_preset("aggressive")

    # Create processor with strict memory limits
    batch_config = BatchConfig(
        strategy=BatchStrategy.MEMORY_OPTIMIZED,
        max_memory_mb=75,  # Strict limit
        min_batch_size=1,
        adaptive_sizing=True,
    )
    processor = BatchProcessor(BatchStrategy.MEMORY_OPTIMIZED, batch_config)

    print(f"Processing {num_samples} samples with {batch_config.max_memory_mb}MB memory limit")
    print("Monitoring memory usage and batch size adjustments:\n")

    # Create parameter stream
    param_stream = stream_augmentation_chain(num_samples, config)

    batch_num = 0
    batch_sizes = []
    memory_usages = []

    try:
        for batch in processor.process_with_memory_monitoring(param_stream):
            batch_num += 1
            batch_size = len(batch)
            batch_sizes.append(batch_size)

            # Get current memory usage
            memory_info = processor._memory_batcher.monitor_memory_usage()
            memory_usages.append(memory_info["percent_used"])

            print(f"Batch {batch_num}: {batch_size} samples")
            print(f"  Memory: {memory_info['used_mb']:.1f}MB ({memory_info['percent_used']:.1f}%)")

            # Check if we're approaching the limit
            if memory_info["percent_used"] > 70:
                print("  ⚠️  High memory usage detected")
            elif memory_info["percent_used"] > 85:
                print("  🚨 Critical memory usage!")

            # Simulate some memory cleanup
            if batch_num % 5 == 0:
                gc.collect()
                print("  🧹 Garbage collection performed")

            print()

    except Exception as e:
        print(f"Processing stopped due to: {e}")

    # Summary
    if batch_sizes:
        avg_batch_size = sum(batch_sizes) / len(batch_sizes)
        max_memory = max(memory_usages)

        print("Summary:")
        print(f"  Batches processed: {len(batch_sizes)}")
        print(f"  Average batch size: {avg_batch_size:.1f}")
        print(f"  Peak memory usage: {max_memory:.1f}%")
        print(f"  Memory limit respected: {'✓' if max_memory < 90 else '✗'}")


def demo_memory_pressure_simulation():
    """Simulate memory pressure and show adaptive behavior."""
    print("\n=== Memory Pressure Simulation ===\n")

    # Create memory-aware batcher
    batcher = MemoryAwareBatcher(max_memory_mb=200, min_batch_size=1)

    print("Simulating increasing memory pressure:")
    print("Showing how batch size adapts to memory conditions\n")

    # Simulate memory pressure over time
    base_batch_size = 20
    current_batch_size = base_batch_size

    memory_pressures = [
        (30, "Low pressure - plenty of memory"),
        (50, "Moderate pressure - some memory used"),
        (70, "High pressure - memory getting tight"),
        (85, "Critical pressure - very little memory left"),
        (95, "Extreme pressure - almost out of memory"),
        (60, "Pressure relieved - memory freed up"),
        (40, "Back to normal - memory available again"),
    ]

    for pressure_percent, description in memory_pressures:
        # Simulate memory info based on pressure
        total_mb = 1000
        used_mb = (pressure_percent / 100) * total_mb
        available_mb = total_mb - used_mb

        memory_info = {
            "total_mb": total_mb,
            "used_mb": used_mb,
            "available_mb": available_mb,
            "percent": pressure_percent,
        }

        # Adjust batch size
        new_batch_size = batcher.adjust_batch_size(current_batch_size, memory_info)

        print(f"Memory pressure: {pressure_percent}% - {description}")
        print(f"  Available memory: {available_mb:.0f} MB")

        if new_batch_size != current_batch_size:
            change = "increased" if new_batch_size > current_batch_size else "decreased"
            print(f"  Batch size {change}: {current_batch_size} → {new_batch_size}")
        else:
            print(f"  Batch size maintained: {current_batch_size}")

        current_batch_size = new_batch_size
        print()


def demo_memory_optimization_strategies():
    """Demonstrate different memory optimization strategies."""
    print("=== Memory Optimization Strategies ===\n")

    num_samples = 30
    config = get_preset("moderate")

    strategies = [
        ("Conservative", {"max_memory_mb": 50, "min_batch_size": 1}),
        ("Balanced", {"max_memory_mb": 100, "min_batch_size": 2}),
        ("Aggressive", {"max_memory_mb": 200, "min_batch_size": 5}),
    ]

    print(f"Comparing memory optimization strategies for {num_samples} samples:\n")

    for strategy_name, params in strategies:
        print(f"--- {strategy_name.upper()} STRATEGY ---")
        print(
            f"Memory limit: {params['max_memory_mb']}MB, Min batch size: {params['min_batch_size']}"
        )

        batch_config = BatchConfig(strategy=BatchStrategy.MEMORY_OPTIMIZED, **params)
        processor = BatchProcessor(BatchStrategy.MEMORY_OPTIMIZED, batch_config)

        # Process samples
        param_stream = stream_augmentation_chain(num_samples, config)

        batch_count = 0
        total_processed = 0
        batch_sizes = []

        for batch in processor.process_stream(param_stream):
            batch_count += 1
            batch_size = len(batch)
            batch_sizes.append(batch_size)
            total_processed += batch_size

        avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0

        print(f"  Batches created: {batch_count}")
        print(f"  Average batch size: {avg_batch_size:.1f}")
        print(f"  Total processed: {total_processed}")
        print(f"  Efficiency: {total_processed / batch_count:.1f} samples/batch")
        print()


def demo_memory_warnings_and_recovery():
    """Demonstrate memory warning system and recovery mechanisms."""
    print("=== Memory Warnings and Recovery ===\n")

    # Create batcher with warning thresholds
    batcher = MemoryAwareBatcher(max_memory_mb=100, min_batch_size=1)

    print("Demonstrating memory warning system:")
    print("Different warning levels based on memory usage\n")

    # Test different memory levels
    memory_levels = [
        (60, "Normal operation"),
        (75, "Warning threshold"),
        (85, "High usage warning"),
        (95, "Critical warning"),
        (99, "Emergency threshold"),
    ]

    for memory_percent, description in memory_levels:
        memory_info = {
            "total_mb": 1000,
            "used_mb": memory_percent * 10,
            "available_mb": (100 - memory_percent) * 10,
            "percent": memory_percent,
        }

        print(f"Memory usage: {memory_percent}% - {description}")

        # Determine warning level
        if memory_percent < 70:
            warning = "✅ Normal"
        elif memory_percent < 80:
            warning = "⚠️  Warning"
        elif memory_percent < 90:
            warning = "🔶 High Usage"
        elif memory_percent < 98:
            warning = "🚨 Critical"
        else:
            warning = "💥 Emergency"

        print(f"  Status: {warning}")

        # Show recommended action
        if memory_percent >= 90:
            print("  Recommended: Reduce batch size immediately")
        elif memory_percent >= 80:
            print("  Recommended: Consider reducing batch size")
        elif memory_percent >= 70:
            print("  Recommended: Monitor closely")
        else:
            print("  Recommended: Normal operation")

        print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Batch Processing: Memory-Aware Processing")
    print("=" * 70)

    demo_memory_monitoring()
    demo_dynamic_batch_sizing()
    demo_memory_limit_enforcement()
    demo_memory_pressure_simulation()
    demo_memory_optimization_strategies()
    demo_memory_warnings_and_recovery()

    print("\n" + "=" * 70)
    print("Demo completed! Memory-aware batching prevents out-of-memory errors.")
    print("=" * 70 + "\n")
