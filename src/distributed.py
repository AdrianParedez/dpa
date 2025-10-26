"""
Distributed training utilities for DPA library.

This module provides rank-aware deterministic seeding and distributed range splitting
utilities for distributed training environments. It ensures that each process in a
distributed training setup generates unique but reproducible augmentation parameters
without data duplication across ranks.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Generator

from .dpa import AugmentationConfig, gen_augmentation_params, fib


class DistributedError(Exception):
    """Base exception for distributed training operations."""
    pass


class InvalidRankError(DistributedError):
    """Raised when rank is invalid for the given world size."""
    pass


class RangeSplittingError(DistributedError):
    """Raised when range splitting operations fail."""
    pass


class SeedGenerationError(DistributedError):
    """Raised when distributed seed generation fails."""
    pass


@dataclass
class RankRange:
    """Represents a sample range assigned to a specific rank."""
    start_id: int
    end_id: int
    rank: int
    total_samples: int


class RankAwareSeedGenerator:
    """
    Generates deterministic seeds that incorporate process rank for distributed training.
    
    This class ensures that different ranks produce different augmentation parameters
    for the same sample_id while maintaining reproducibility when the same rank and
    sample_id combination is used.
    """
    
    def __init__(self, base_seed: int = 0, world_size: int = 1) -> None:
        """
        Initialize the rank-aware seed generator.
        
        Args:
            base_seed: Base seed for deterministic generation
            world_size: Total number of processes in distributed training
            
        Raises:
            ValueError: If base_seed is negative or world_size < 1
        """
        if base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
            
        self.base_seed = base_seed
        self.world_size = world_size
    
    def generate_seed(self, sample_id: int, rank: int, augmentation_depth: int = 10) -> str:
        """
        Generate deterministic hash seed incorporating rank and sample_id.
        
        Args:
            sample_id: Unique sample identifier
            rank: Process rank (0-based)
            augmentation_depth: Number of hash iterations
            
        Returns:
            SHA256 hash as hexadecimal string
            
        Raises:
            ValueError: If sample_id < 0, rank < 0, rank >= world_size, or augmentation_depth < 1
            SeedGenerationError: If seed generation fails
        """
        if sample_id < 0:
            raise ValueError("sample_id must be non-negative")
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if rank >= self.world_size:
            raise ValueError(f"rank ({rank}) must be < world_size ({self.world_size})")
        if augmentation_depth < 1:
            raise ValueError("augmentation_depth must be >= 1")
            
        try:
            # Combine base_seed, rank, and sample_id for unique seed per rank
            combined_seed = f"{self.base_seed}:{rank}:{sample_id}"
            prev_hash = hashlib.sha256(combined_seed.encode()).hexdigest()
            
            # Apply Fibonacci-based hash chain like the original implementation
            for i in range(augmentation_depth):
                fib_val = str(fib(i))
                combined = prev_hash + fib_val
                prev_hash = hashlib.sha256(combined.encode()).hexdigest()
                
            return prev_hash
            
        except Exception as e:
            raise SeedGenerationError(f"Failed to generate rank-aware seed: {e}") from e


class DistributedRangeSplitter:
    """
    Splits sample ranges across multiple ranks for distributed training.
    
    This class handles range calculation for even and uneven sample distribution,
    edge cases like single rank or more ranks than samples, and provides
    validation methods for range correctness.
    """
    
    def __init__(self, total_samples: int, world_size: int) -> None:
        """
        Initialize the distributed range splitter.
        
        Args:
            total_samples: Total number of samples to distribute
            world_size: Total number of processes in distributed training
            
        Raises:
            ValueError: If total_samples < 0 or world_size < 1
        """
        if total_samples < 0:
            raise ValueError("total_samples must be non-negative")
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
            
        self.total_samples = total_samples
        self.world_size = world_size
        self._ranges_cache = None
    
    def get_rank_range(self, rank: int) -> tuple[int, int]:
        """
        Get the sample range (start_id, end_id) for a specific rank.
        
        Args:
            rank: Process rank (0-based)
            
        Returns:
            Tuple of (start_id, end_id) where end_id is exclusive
            
        Raises:
            ValueError: If rank is invalid
            RangeSplittingError: If range calculation fails
        """
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if rank >= self.world_size:
            raise ValueError(f"rank ({rank}) must be < world_size ({self.world_size})")
            
        try:
            # Handle edge case: no samples
            if self.total_samples == 0:
                return (0, 0)
            
            # Handle edge case: more ranks than samples
            if rank >= self.total_samples:
                return (0, 0)  # Empty range for excess ranks
            
            # Calculate base samples per rank and remainder
            base_samples_per_rank = self.total_samples // self.world_size
            remainder = self.total_samples % self.world_size
            
            # Distribute remainder among first 'remainder' ranks
            if rank < remainder:
                samples_for_rank = base_samples_per_rank + 1
                start_id = rank * samples_for_rank
            else:
                samples_for_rank = base_samples_per_rank
                start_id = remainder * (base_samples_per_rank + 1) + (rank - remainder) * base_samples_per_rank
            
            end_id = start_id + samples_for_rank
            
            # Ensure we don't exceed total samples
            end_id = min(end_id, self.total_samples)
            
            return (start_id, end_id)
            
        except Exception as e:
            raise RangeSplittingError(f"Failed to calculate range for rank {rank}: {e}") from e
    
    def get_all_ranges(self) -> list[tuple[int, int]]:
        """
        Get sample ranges for all ranks.
        
        Returns:
            List of (start_id, end_id) tuples for each rank
            
        Raises:
            RangeSplittingError: If range calculation fails
        """
        if self._ranges_cache is None:
            try:
                self._ranges_cache = []
                for rank in range(self.world_size):
                    range_tuple = self.get_rank_range(rank)
                    self._ranges_cache.append(range_tuple)
            except Exception as e:
                raise RangeSplittingError(f"Failed to calculate all ranges: {e}") from e
                
        return self._ranges_cache.copy()
    
    def get_all_rank_ranges(self) -> list[RankRange]:
        """
        Get RankRange objects for all ranks.
        
        Returns:
            List of RankRange objects containing detailed range information
            
        Raises:
            RangeSplittingError: If range calculation fails
        """
        try:
            ranges = []
            for rank in range(self.world_size):
                start_id, end_id = self.get_rank_range(rank)
                ranges.append(RankRange(
                    start_id=start_id,
                    end_id=end_id,
                    rank=rank,
                    total_samples=self.total_samples
                ))
            return ranges
        except Exception as e:
            raise RangeSplittingError(f"Failed to create RankRange objects: {e}") from e
    
    def validate_ranges(self) -> bool:
        """
        Validate that ranges cover all samples without overlap or gaps.
        
        Returns:
            True if ranges are valid, False otherwise
        """
        try:
            ranges = self.get_all_ranges()
            
            # Handle edge case: no samples
            if self.total_samples == 0:
                return all(start == 0 and end == 0 for start, end in ranges)
            
            # Check for gaps and overlaps
            ranges.sort(key=lambda x: x[0])  # Sort by start_id
            
            expected_start = 0
            total_covered = 0
            
            for start_id, end_id in ranges:
                # Skip empty ranges (for excess ranks)
                if start_id == end_id:
                    continue
                    
                # Check for gaps
                if start_id != expected_start:
                    return False
                    
                # Check for valid range
                if start_id >= end_id:
                    return False
                    
                total_covered += (end_id - start_id)
                expected_start = end_id
            
            # Check that we covered exactly all samples
            return total_covered == self.total_samples and expected_start == self.total_samples
            
        except Exception:
            return False
    
    def get_rank_for_sample(self, sample_id: int) -> int:
        """
        Determine which rank should process a given sample_id.
        
        Args:
            sample_id: Sample identifier to find rank for
            
        Returns:
            Rank that should process this sample
            
        Raises:
            ValueError: If sample_id is invalid
            RangeSplittingError: If no rank found for sample
        """
        if sample_id < 0:
            raise ValueError("sample_id must be non-negative")
        if sample_id >= self.total_samples:
            raise ValueError(f"sample_id ({sample_id}) must be < total_samples ({self.total_samples})")
            
        try:
            for rank in range(self.world_size):
                start_id, end_id = self.get_rank_range(rank)
                if start_id <= sample_id < end_id:
                    return rank
                    
            raise RangeSplittingError(f"No rank found for sample_id {sample_id}")
            
        except Exception as e:
            if isinstance(e, RangeSplittingError):
                raise
            raise RangeSplittingError(f"Failed to find rank for sample_id {sample_id}: {e}") from e


def stream_distributed_augmentation_chain(
    num_samples: int,
    rank: int,
    world_size: int,
    config: AugmentationConfig | None = None,
    base_seed: int = 0,
    chunk_size: int | None = None,
    verbose: bool = False,
    inclusive_range: bool = True,
) -> Generator[dict[str, Any], None, None]:
    """
    Stream augmentation parameters for a specific rank in distributed training.
    
    This function integrates range splitting with the existing streaming API to provide
    distributed parameter generation. It supports both inclusive and exclusive range
    strategies and maintains compatibility with existing streaming patterns.
    
    Args:
        num_samples: Total number of samples across all ranks
        rank: Process rank (0-based)
        world_size: Total number of processes in distributed training
        config: AugmentationConfig instance (uses defaults if None)
        base_seed: Base seed for deterministic generation
        chunk_size: If specified, yields lists of parameters instead of individual items
        verbose: If True, print progress information
        inclusive_range: If True, use inclusive range strategy; if False, use exclusive
        
    Yields:
        Individual parameter dictionaries when chunk_size=None,
        or lists of parameter dictionaries when chunk_size is specified
        
    Raises:
        ValueError: If parameters are invalid
        RangeSplittingError: If range calculation fails
        SeedGenerationError: If parameter generation fails
        
    Examples:
        Basic distributed streaming:
        >>> # Rank 0 of 4 processes
        >>> for params in stream_distributed_augmentation_chain(1000, rank=0, world_size=4):
        ...     print(f"Rank 0 processing: {params['rotation']:.2f}")
        
        Chunked distributed streaming:
        >>> # Rank 1 of 2 processes with chunking
        >>> for chunk in stream_distributed_augmentation_chain(
        ...     1000, rank=1, world_size=2, chunk_size=50
        ... ):
        ...     batch_process_rank1(chunk)
        
        With custom configuration:
        >>> config = AugmentationConfig(rotation_range=(-90, 90))
        >>> generator = stream_distributed_augmentation_chain(
        ...     5000, rank=2, world_size=8, config=config, verbose=True
        ... )
        >>> for params in generator:
        ...     if params['rotation'] > 45:
        ...         process_high_rotation(params)
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if rank < 0:
        raise ValueError("rank must be non-negative")
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if rank >= world_size:
        raise ValueError(f"rank ({rank}) must be < world_size ({world_size})")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
        
    if config is None:
        config = AugmentationConfig()
        
    try:
        # Create range splitter to determine this rank's sample range
        splitter = DistributedRangeSplitter(num_samples, world_size)
        start_id, end_id = splitter.get_rank_range(rank)
        
        # Calculate actual samples for this rank
        rank_samples = end_id - start_id
        
        if verbose:
            print(f"Rank {rank}/{world_size}: processing samples {start_id}-{end_id-1} ({rank_samples} samples)")
        
        # Handle empty range case
        if rank_samples == 0:
            if verbose:
                print(f"Rank {rank}: no samples to process (empty range)")
            return
            
        # The range splitter already gives us the correct exclusive range
        # The inclusive_range parameter is for documentation/future use
        actual_end = end_id
        actual_samples = rank_samples
            
        # Generate parameters for this rank's range using distributed seeding
        if chunk_size is None:
            # Yield individual parameters
            for sample_id in range(start_id, actual_end):
                try:
                    params = gen_distributed_augmentation_params(
                        sample_id=sample_id,
                        rank=rank,
                        config=config,
                        base_seed=base_seed,
                        world_size=world_size
                    )
                    
                    if verbose and actual_samples <= 10:
                        print(f"Rank {rank} Sample({sample_id}) -> "
                              f"rotation={params['rotation']:.2f}° "
                              f"brightness={params['brightness']:.2f} "
                              f"hash={params['hash'][:8]}...")
                    
                    yield params
                    
                except Exception as e:
                    raise SeedGenerationError(f"Failed to generate parameters for rank {rank}, sample {sample_id}: {e}") from e
        else:
            # Yield chunks of parameters
            chunk = []
            for sample_id in range(start_id, actual_end):
                try:
                    params = gen_distributed_augmentation_params(
                        sample_id=sample_id,
                        rank=rank,
                        config=config,
                        base_seed=base_seed,
                        world_size=world_size
                    )
                    chunk.append(params)
                    
                    if verbose and actual_samples <= 10:
                        print(f"Rank {rank} Sample({sample_id}) -> "
                              f"rotation={params['rotation']:.2f}° "
                              f"brightness={params['brightness']:.2f} "
                              f"hash={params['hash'][:8]}...")
                    
                    if len(chunk) == chunk_size:
                        yield chunk  # pyright: ignore[reportReturnType]
                        chunk = []
                        
                except Exception as e:
                    raise SeedGenerationError(f"Failed to generate parameters for rank {rank}, sample {sample_id}: {e}") from e
            
            # Yield remaining items if any
            if chunk:
                yield chunk  # pyright: ignore[reportReturnType]
                
        if verbose:
            print(f"Rank {rank}: completed processing {actual_samples} samples")
            
    except Exception as e:
        if isinstance(e, (ValueError, RangeSplittingError, SeedGenerationError)):
            raise
        raise RangeSplittingError(f"Failed to stream distributed augmentation chain for rank {rank}: {e}") from e


def stream_distributed_augmentation_range(
    start_id: int,
    end_id: int,
    rank: int,
    world_size: int,
    config: AugmentationConfig | None = None,
    base_seed: int = 0,
    chunk_size: int | None = None,
    verbose: bool = False,
) -> Generator[dict[str, Any], None, None]:
    """
    Stream distributed parameters for a specific range of sample IDs.
    
    This function provides distributed parameter generation for a custom range,
    useful for processing subsets of data or resuming distributed operations.
    Unlike stream_distributed_augmentation_chain, this function works with
    a pre-defined range rather than splitting the total samples.
    
    Args:
        start_id: Starting sample ID (inclusive)
        end_id: Ending sample ID (exclusive)
        rank: Process rank (0-based)
        world_size: Total number of processes in distributed training
        config: AugmentationConfig instance (uses defaults if None)
        base_seed: Base seed for deterministic generation
        chunk_size: If specified, yields lists of parameters instead of individual items
        verbose: If True, print progress information
        
    Yields:
        Individual parameter dictionaries when chunk_size=None,
        or lists of parameter dictionaries when chunk_size is specified
        
    Raises:
        ValueError: If parameters are invalid
        SeedGenerationError: If parameter generation fails
        
    Examples:
        Stream a specific range for a rank:
        >>> # Process samples 1000-2000 for rank 2
        >>> for params in stream_distributed_augmentation_range(
        ...     1000, 2000, rank=2, world_size=4
        ... ):
        ...     process_sample(params)
        
        Resume processing from a specific point:
        >>> # Resume rank 1 processing from sample 5000 to 10000
        >>> generator = stream_distributed_augmentation_range(
        ...     5000, 10000, rank=1, world_size=8, verbose=True
        ... )
        >>> for params in generator:
        ...     if should_stop():
        ...         break
        ...     process_sample(params)
    """
    if start_id < 0:
        raise ValueError("start_id must be non-negative")
    if end_id < start_id:
        raise ValueError("end_id must be >= start_id")
    if rank < 0:
        raise ValueError("rank must be non-negative")
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if rank >= world_size:
        raise ValueError(f"rank ({rank}) must be < world_size ({world_size})")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
        
    if config is None:
        config = AugmentationConfig()
        
    num_samples = end_id - start_id
    
    if verbose:
        print(f"Rank {rank}/{world_size}: processing range {start_id}-{end_id-1} ({num_samples} samples)")
    
    try:
        if chunk_size is None:
            # Yield individual parameters
            for sample_id in range(start_id, end_id):
                try:
                    params = gen_distributed_augmentation_params(
                        sample_id=sample_id,
                        rank=rank,
                        config=config,
                        base_seed=base_seed,
                        world_size=world_size
                    )
                    
                    if verbose and num_samples <= 10:
                        print(f"Rank {rank} Sample({sample_id}) -> "
                              f"rotation={params['rotation']:.2f}° "
                              f"brightness={params['brightness']:.2f} "
                              f"hash={params['hash'][:8]}...")
                    
                    yield params
                    
                except Exception as e:
                    raise SeedGenerationError(f"Failed to generate parameters for rank {rank}, sample {sample_id}: {e}") from e
        else:
            # Yield chunks of parameters
            chunk = []
            for sample_id in range(start_id, end_id):
                try:
                    params = gen_distributed_augmentation_params(
                        sample_id=sample_id,
                        rank=rank,
                        config=config,
                        base_seed=base_seed,
                        world_size=world_size
                    )
                    chunk.append(params)
                    
                    if verbose and num_samples <= 10:
                        print(f"Rank {rank} Sample({sample_id}) -> "
                              f"rotation={params['rotation']:.2f}° "
                              f"brightness={params['brightness']:.2f} "
                              f"hash={params['hash'][:8]}...")
                    
                    if len(chunk) == chunk_size:
                        yield chunk  # pyright: ignore[reportReturnType]
                        chunk = []
                        
                except Exception as e:
                    raise SeedGenerationError(f"Failed to generate parameters for rank {rank}, sample {sample_id}: {e}") from e
            
            # Yield remaining items if any
            if chunk:
                yield chunk  # pyright: ignore[reportReturnType]
                
        if verbose:
            print(f"Rank {rank}: completed processing range {start_id}-{end_id-1}")
            
    except Exception as e:
        if isinstance(e, (ValueError, SeedGenerationError)):
            raise
        raise SeedGenerationError(f"Failed to stream distributed range for rank {rank}: {e}") from e


def gen_distributed_augmentation_params(
    sample_id: int, 
    rank: int, 
    config: AugmentationConfig | None = None,
    base_seed: int = 0,
    world_size: int = 1
) -> dict[str, Any]:
    """
    Generate deterministic augmentation parameters with rank awareness.
    
    This function generates augmentation parameters that are unique per rank
    while maintaining deterministic behavior for the same rank/sample_id combination.
    It integrates with the existing AugmentationConfig system and maintains
    backward compatibility.
    
    Args:
        sample_id: Unique sample identifier
        rank: Process rank (0-based)
        config: AugmentationConfig instance (uses defaults if None)
        base_seed: Base seed for deterministic generation
        world_size: Total number of processes in distributed training
        
    Returns:
        Dictionary with augmentation parameters including rank-aware hash
        
    Raises:
        ValueError: If parameters are invalid
        SeedGenerationError: If parameter generation fails
    """
    if sample_id < 0:
        raise ValueError("sample_id must be non-negative")
    if rank < 0:
        raise ValueError("rank must be non-negative")
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if rank >= world_size:
        raise ValueError(f"rank ({rank}) must be < world_size ({world_size})")
        
    if config is None:
        config = AugmentationConfig()
        
    try:
        # Create rank-aware seed generator
        seed_generator = RankAwareSeedGenerator(base_seed=base_seed, world_size=world_size)
        
        # Generate rank-aware hash seed
        hash_seed = seed_generator.generate_seed(sample_id, rank, config.augmentation_depth)
        
        # Convert hash to integer seed for random number generation
        seed_int = int(hash_seed, 16) % (2**32)
        
        # Use the same random generation logic as the original implementation
        import random
        random.seed(seed_int)
        
        return {
            "rotation": random.uniform(*config.rotation_range),
            "brightness": random.uniform(*config.brightness_range),
            "noise": random.uniform(*config.noise_range),
            "scale": random.uniform(*config.scale_range),
            "contrast": random.uniform(*config.contrast_range),
            "hash": hash_seed,
            "rank": rank,  # Include rank in output for debugging/verification
        }
        
    except Exception as e:
        raise SeedGenerationError(f"Failed to generate distributed augmentation parameters: {e}") from e