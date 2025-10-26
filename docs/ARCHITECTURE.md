# Architecture

## Overview

DPA generates deterministic augmentation parameters through a multi-stage pipeline that combines cryptographic hashing with Fibonacci sequences to ensure reproducibility while maintaining statistical diversity.

## Core Pipeline

### Stage 1: Seed ID Hashing

Each sample receives a unique seed_id (0, 1, 2, ...). This ID is converted to a SHA256 hash:

```
seed_id: 42
    |
    v
encode to string "42"
    |
    v
SHA256 hash
    |
    v
f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1
```

### Stage 2: Fibonacci Chain

The hash is iteratively combined with Fibonacci numbers. For each iteration i from 0 to augmentation_depth:

```
current_hash + str(fib(i)) -> SHA256 -> new_hash
```

Example with depth=3:

```
Iteration 0: hash + "0" -> SHA256 -> hash1
Iteration 1: hash1 + "1" -> SHA256 -> hash2
Iteration 2: hash2 + "1" -> SHA256 -> hash3
Iteration 3: hash3 + "2" -> SHA256 -> hash4
```

The Fibonacci sequence (0, 1, 1, 2, 3, 5, 8, 13...) adds mathematical structure while the SHA256 hashing provides cryptographic mixing.

### Stage 3: Random Seed Generation

Convert the final hash to a 32-bit integer for use as a random seed:

```
final_hash (hex string)
    |
    v
convert to integer
    |
    v
modulo 2^32 (keep 32-bit range)
    |
    v
seed_int (0 to 4,294,967,295)
```

### Stage 4: Parameter Generation

Use the seed_int to initialize Python's random module, then generate parameters uniformly within configured ranges:

```
random.seed(seed_int)
    |
    +-- rotation = random.uniform(rotation_range[0], rotation_range[1])
    +-- brightness = random.uniform(brightness_range[0], brightness_range[1])
    +-- noise = random.uniform(noise_range[0], noise_range[1])
    +-- scale = random.uniform(scale_range[0], scale_range[1])
    +-- contrast = random.uniform(contrast_range[0], contrast_range[1])
    |
    v
parameters dict
```

## Key Design Decisions

### Why Fibonacci?

Fibonacci numbers are deterministic, bounded, and mathematically rich. They grow slowly (good for deep chains) and have interesting mathematical properties. They're also memorable and make the algorithm distinctive.

### Why SHA256?

SHA256 provides:
- Deterministic output (same input always produces same output)
- Cryptographic mixing (small input changes cause large output changes)
- Fixed output size (64 hex characters)
- No collisions in practice

### Why Chain Multiple Times?

The augmentation_depth parameter allows tuning the "mixing" of the hash. Higher depths create more complex relationships between seed_id and output, reducing predictability while maintaining determinism.

### Why Modulo 2^32?

Python's random module uses a Mersenne Twister with 32-bit seeds. Using modulo 2^32 ensures the seed fits the expected range and distributes evenly.

## Data Flow

```
seed_id (int)
    |
    v
[SHA256 Hash]
    |
    v
Fibonacci Chain Loop (augmentation_depth times)
    |
    +-> [Combine with fib(i)]
    +-> [SHA256 Hash]
    |
    v
final_hash (hex string)
    |
    v
[Convert to 32-bit int]
    |
    v
[Initialize random.seed()]
    |
    v
[Generate 5 parameters]
    |
    v
parameters dict + hash
```

## Reproducibility Guarantees

### Determinism

Given the same input (seed_id, config), DPA always produces identical output:

```python
params1 = gen_augmentation_params(42, config)
params2 = gen_augmentation_params(42, config)
assert params1 == params2  # Always true
```

This holds because:
- SHA256 is deterministic
- Fibonacci sequence is deterministic
- Python's random module is deterministic with fixed seed

### Independence

Different seed_ids produce different parameters due to SHA256's avalanche effect (small input changes cause large output changes).

### Consistency

The same augmentation chain can be reproduced years later because the algorithm never changes - it's purely mathematical with no dependencies on external state.

## Performance Characteristics

### Time Complexity

For n samples with augmentation_depth d:
- O(n * d) for generation (d hash operations per sample)
- Each hash operation is O(1) for fixed-size input

### Space Complexity

- O(1) for single sample generation
- O(n) for storing n samples in memory
- Fibonacci cache grows with augmentation_depth (typically d ≤ 20)

### Optimization Techniques

1. **LRU Caching** - Fibonacci numbers are memoized to avoid recomputation
2. **Batch Output** - String concatenation happens once at the end, not per sample
3. **In-Memory Storage** - No disk I/O during generation

## Configuration

### Presets

Three presets balance augmentation intensity:

**Mild**
- Subtle changes
- Narrow ranges (e.g., rotation ±15°)
- Best for: Fine-tuning, small datasets

**Moderate**
- Balanced augmentation
- Medium ranges (e.g., rotation ±30°)
- Best for: General use, training

**Aggressive**
- Heavy augmentation
- Wide ranges (e.g., rotation ±45°)
- Best for: Large datasets, robust models

### Augmentation Depth

Higher depth increases hash mixing but has minimal performance impact:

- depth=5: Light mixing, fast
- depth=10: Balanced (default)
- depth=15+: Heavy mixing, still fast

## Output Format

Augmentations are saved as JSON with three sections:

**Metadata**: Configuration used, number of samples

**Augmentations**: Array of parameter dictionaries, each containing 5 parameters + the underlying hash

**Statistics**: Mean, stdev, min, max for each parameter (optional)