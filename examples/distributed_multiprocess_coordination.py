#!/usr/bin/env python3
"""
Distributed Training Example: Multi-Process Coordination

This example demonstrates how to coordinate multiple processes in distributed
training, showing how each process handles its assigned data range and 
generates unique augmentation parameters.

Requirements addressed: 1.1, 4.1 (distributed coordination and range splitting)
"""

import sys
import multiprocessing as mp
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distributed import (
    DistributedRangeSplitter, 
    stream_distributed_augmentation_chain,
    gen_distributed_augmentation_params
)
from src.dpa import get_preset, AugmentationConfig


def worker_process(rank, world_size, total_samples, config_dict, results_queue):
    """
    Worker process that simulates a distributed training rank.
    
    Args:
        rank: Process rank (0-based)
        world_size: Total number of processes
        total_samples: Total samples in the dataset
        config_dict: Augmentation configuration as dictionary
        results_queue: Queue to send results back to main process
    """
    try:
        # Reconstruct config from dictionary
        config = AugmentationConfig(**config_dict)
        
        # Get the range of samples this rank should process
        splitter = DistributedRangeSplitter(total_samples, world_size)
        start_id, end_id = splitter.get_rank_range(rank)
        samples_to_process = end_id - start_id
        
        print(f"Rank {rank}: Processing samples {start_id} to {end_id-1} ({samples_to_process} samples)")
        
        # Process samples using streaming API
        processed_samples = []
        param_stream = stream_distributed_augmentation_chain(
            num_samples=total_samples,
            rank=rank,
            world_size=world_size,
            config=config
        )
        
        start_time = time.time()
        for params in param_stream:
            # Simulate some processing time
            time.sleep(0.01)  # 10ms per sample
            processed_samples.append({
                'sample_id': params.get('sample_id', len(processed_samples) + start_id),
                'rotation': params['rotation'],
                'brightness': params['brightness'],
                'scale': params['scale']
            })
        
        processing_time = time.time() - start_time
        
        # Send results back to main process
        results_queue.put({
            'rank': rank,
            'samples_processed': len(processed_samples),
            'processing_time': processing_time,
            'sample_range': (start_id, end_id),
            'first_few_samples': processed_samples[:3],  # First 3 for verification
            'status': 'completed'
        })
        
    except Exception as e:
        results_queue.put({
            'rank': rank,
            'status': 'error',
            'error': str(e)
        })


def demo_multiprocess_coordination():
    """Demonstrate coordination between multiple processes."""
    print("=== Multi-Process Coordination Demo ===\n")
    
    # Configuration
    world_size = 4
    total_samples = 50
    config = get_preset("moderate")
    
    print(f"Launching {world_size} worker processes")
    print(f"Total samples to process: {total_samples}")
    print(f"Expected samples per rank: ~{total_samples // world_size}\n")
    
    # Create a queue for collecting results
    results_queue = mp.Queue()
    
    # Convert config to dictionary for multiprocessing
    config_dict = {
        'rotation_range': config.rotation_range,
        'brightness_range': config.brightness_range,
        'noise_range': config.noise_range,
        'scale_range': config.scale_range,
        'contrast_range': config.contrast_range,
        'augmentation_depth': config.augmentation_depth
    }
    
    # Launch worker processes
    processes = []
    start_time = time.time()
    
    for rank in range(world_size):
        p = mp.Process(
            target=worker_process,
            args=(rank, world_size, total_samples, config_dict, results_queue)
        )
        p.start()
        processes.append(p)
    
    # Collect results from all processes
    results = []
    for _ in range(world_size):
        result = results_queue.get()
        results.append(result)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    total_time = time.time() - start_time
    
    # Analyze results
    print("=== Results Analysis ===")
    total_processed = 0
    successful_ranks = 0
    
    # Sort results by rank for consistent output
    results.sort(key=lambda x: x.get('rank', -1))
    
    for result in results:
        if result['status'] == 'completed':
            rank = result['rank']
            samples = result['samples_processed']
            proc_time = result['processing_time']
            start_id, end_id = result['sample_range']
            
            print(f"Rank {rank}: {samples} samples ({start_id}-{end_id-1}) in {proc_time:.2f}s")
            
            # Show first few samples for verification
            print(f"  First samples:")
            for sample in result['first_few_samples']:
                print(f"    Sample {sample['sample_id']}: rot={sample['rotation']:.2f}, "
                      f"bright={sample['brightness']:.3f}")
            
            total_processed += samples
            successful_ranks += 1
        else:
            print(f"Rank {result['rank']}: ERROR - {result['error']}")
        print()
    
    print(f"Summary:")
    print(f"  Successful ranks: {successful_ranks}/{world_size}")
    print(f"  Total samples processed: {total_processed}/{total_samples}")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Average time per rank: {total_time:.2f}s (parallel execution)")


def demo_parameter_uniqueness_verification():
    """Verify that different ranks generate different parameters for same sample_id."""
    print("\n=== Parameter Uniqueness Verification ===\n")
    
    world_size = 3
    sample_id = 42
    config = get_preset("mild")
    
    print(f"Verifying parameter uniqueness across {world_size} ranks for sample_id {sample_id}")
    
    rank_parameters = {}
    for rank in range(world_size):
        params = gen_distributed_augmentation_params(sample_id, rank, config, world_size=world_size)
        rank_parameters[rank] = params
        
        print(f"Rank {rank}: rotation={params['rotation']:.4f}, "
              f"brightness={params['brightness']:.4f}, "
              f"scale={params['scale']:.4f}")
    
    # Check for uniqueness
    unique_params = set()
    for rank, params in rank_parameters.items():
        param_signature = (
            round(params['rotation'], 6),
            round(params['brightness'], 6),
            round(params['scale'], 6)
        )
        unique_params.add(param_signature)
    
    print(f"\nUnique parameter sets: {len(unique_params)}")
    if len(unique_params) == world_size:
        print("✓ All ranks generate unique parameters for the same sample_id")
    else:
        print("✗ Some ranks generated identical parameters (unexpected)")


def demo_reproducibility_across_runs():
    """Demonstrate that the same distributed setup produces identical results."""
    print("\n=== Reproducibility Across Runs ===\n")
    
    world_size = 2
    total_samples = 10
    config = get_preset("mild")
    
    print(f"Testing reproducibility with {world_size} ranks, {total_samples} samples")
    
    # Run 1
    print("\nRun 1:")
    run1_results = {}
    for rank in range(world_size):
        splitter = DistributedRangeSplitter(total_samples, world_size)
        start_id, end_id = splitter.get_rank_range(rank)
        
        rank_params = []
        for sample_id in range(start_id, end_id):
            params = gen_distributed_augmentation_params(sample_id, rank, config, world_size=world_size)
            rank_params.append(params['rotation'])  # Just track rotation for simplicity
        
        run1_results[rank] = rank_params
        print(f"  Rank {rank}: {len(rank_params)} samples, first rotation: {rank_params[0]:.6f}")
    
    # Run 2
    print("\nRun 2:")
    run2_results = {}
    for rank in range(world_size):
        splitter = DistributedRangeSplitter(total_samples, world_size)
        start_id, end_id = splitter.get_rank_range(rank)
        
        rank_params = []
        for sample_id in range(start_id, end_id):
            params = gen_distributed_augmentation_params(sample_id, rank, config, world_size=world_size)
            rank_params.append(params['rotation'])
        
        run2_results[rank] = rank_params
        print(f"  Rank {rank}: {len(rank_params)} samples, first rotation: {rank_params[0]:.6f}")
    
    # Compare results
    print("\nComparison:")
    all_match = True
    for rank in range(world_size):
        if run1_results[rank] == run2_results[rank]:
            print(f"  Rank {rank}: ✓ Identical results")
        else:
            print(f"  Rank {rank}: ✗ Different results")
            all_match = False
    
    if all_match:
        print("\n✓ Full reproducibility verified across distributed runs!")
    else:
        print("\n✗ Reproducibility failed!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Distributed Training: Multi-Process Coordination")
    print("=" * 70)
    
    # Note: Multiprocessing demo might not work in all environments
    print("Note: The multiprocessing demo requires a proper Python environment.")
    print("If running in a restricted environment, only the verification demos will run.\n")
    
    try:
        # Test if multiprocessing works
        mp.set_start_method('spawn', force=True)
        demo_multiprocess_coordination()
    except Exception as e:
        print(f"Multiprocessing demo skipped due to environment limitations: {e}\n")
        print("Running verification demos instead...\n")
    
    demo_parameter_uniqueness_verification()
    demo_reproducibility_across_runs()
    
    print("\n" + "=" * 70)
    print("Demo completed! Distributed coordination ensures proper data handling.")
    print("=" * 70 + "\n")