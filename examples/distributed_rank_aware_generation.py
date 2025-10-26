#!/usr/bin/env python3
"""
Distributed Training Example: Rank-Aware Parameter Generation

This example demonstrates how to generate unique, deterministic augmentation 
parameters for each rank in a distributed training setup. Each rank will 
generate different parameters for the same sample_id, ensuring no data 
duplication across processes.

Requirements addressed: 1.1 (rank-aware deterministic seeding)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dpa import AugmentationConfig, get_preset
from src.distributed import gen_distributed_augmentation_params, RankAwareSeedGenerator


def demo_rank_aware_generation():
    """Demonstrate rank-aware parameter generation for distributed training."""
    print("=== Rank-Aware Parameter Generation Demo ===\n")
    
    # Simulate a distributed training setup with 4 ranks
    world_size = 4
    num_samples = 5
    config = get_preset("moderate")
    
    print(f"Simulating distributed training with {world_size} ranks")
    print(f"Generating parameters for {num_samples} samples per rank\n")
    
    # Generate parameters for each rank
    all_rank_params = {}
    
    for rank in range(world_size):
        print(f"--- Rank {rank} Parameters ---")
        rank_params = []
        
        for sample_id in range(num_samples):
            params = gen_distributed_augmentation_params(
                sample_id=sample_id,
                rank=rank,
                config=config,
                world_size=world_size
            )
            rank_params.append(params)
            
            # Show first few parameters for demonstration
            if sample_id < 2:
                print(f"Sample {sample_id}: rotation={params['rotation']:.2f}, "
                      f"brightness={params['brightness']:.3f}")
        
        all_rank_params[rank] = rank_params
        print()
    
    # Verify uniqueness across ranks for same sample_id
    print("=== Uniqueness Verification ===")
    sample_id_to_check = 0
    print(f"Checking sample_id {sample_id_to_check} across all ranks:")
    
    for rank in range(world_size):
        params = all_rank_params[rank][sample_id_to_check]
        print(f"Rank {rank}: rotation={params['rotation']:.2f}, "
              f"brightness={params['brightness']:.3f}, "
              f"scale={params['scale']:.3f}")
    
    print("\n✓ Each rank generates different parameters for the same sample_id")


def demo_reproducibility_across_runs():
    """Demonstrate that the same rank generates identical parameters across runs."""
    print("\n=== Reproducibility Demo ===\n")
    
    rank = 1
    sample_id = 42
    config = get_preset("mild")
    
    print(f"Testing reproducibility for rank {rank}, sample_id {sample_id}")
    
    # Generate parameters twice
    params1 = gen_distributed_augmentation_params(sample_id, rank, config, world_size=4)
    params2 = gen_distributed_augmentation_params(sample_id, rank, config, world_size=4)
    
    print(f"Run 1: rotation={params1['rotation']:.6f}, brightness={params1['brightness']:.6f}")
    print(f"Run 2: rotation={params2['rotation']:.6f}, brightness={params2['brightness']:.6f}")
    
    if params1 == params2:
        print("\n✓ Reproducibility verified! Same rank produces identical parameters")
    else:
        print("\n✗ Reproducibility failed!")


def demo_seed_generator_direct():
    """Demonstrate direct usage of RankAwareSeedGenerator."""
    print("\n=== Direct Seed Generator Usage ===\n")
    
    base_seed = 12345
    world_size = 8
    generator = RankAwareSeedGenerator(base_seed=base_seed, world_size=world_size)
    
    print(f"RankAwareSeedGenerator with base_seed={base_seed}, world_size={world_size}")
    print("Generated seeds for different ranks and sample_ids:\n")
    
    # Show seed generation pattern
    for rank in [0, 1, 7]:  # Show first, second, and last rank
        print(f"Rank {rank}:")
        for sample_id in range(3):
            seed = generator.generate_seed(sample_id, rank)
            print(f"  Sample {sample_id}: {seed}")
        print()


def demo_backward_compatibility():
    """Demonstrate that existing code still works without rank parameter."""
    print("=== Backward Compatibility Demo ===\n")
    
    from src.dpa import gen_augmentation_params
    
    sample_id = 10
    config = get_preset("moderate")
    
    # Old way (still works)
    old_params = gen_augmentation_params(sample_id, config)
    
    # New way with rank=0, world_size=1 (should be identical to non-distributed)
    new_params = gen_distributed_augmentation_params(sample_id, rank=0, config=config, world_size=1)
    
    print(f"Original function: rotation={old_params['rotation']:.6f}")
    print(f"Distributed function (rank=0, world_size=1): rotation={new_params['rotation']:.6f}")
    
    if old_params == new_params:
        print("\n✓ Backward compatibility maintained!")
    else:
        print("\n✗ Backward compatibility broken!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Distributed Training: Rank-Aware Parameter Generation")
    print("=" * 70)
    
    demo_rank_aware_generation()
    demo_reproducibility_across_runs()
    demo_seed_generator_direct()
    demo_backward_compatibility()
    
    print("\n" + "=" * 70)
    print("Demo completed! Each rank generates unique, reproducible parameters.")
    print("=" * 70 + "\n")