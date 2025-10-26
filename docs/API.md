# API Reference

## AugmentationConfig

Dataclass for configuring augmentation parameter ranges.

```python
@dataclass
class AugmentationConfig:
    rotation_range: Tuple[float, float] = (-30, 30)
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    noise_range: Tuple[float, float] = (0, 0.1)
    scale_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    augmentation_depth: int = 10
```

## Functions

### generate_augmentation_chain

```python
generate_augmentation_chain(
    num_samples: int,
    config: Optional[AugmentationConfig] = None,
    verbose: bool = False,
    save_path: Optional[str] = None
) -> List[Dict[str, Any]]
```

Generate augmentation parameters for multiple samples.

**Parameters:**
- `num_samples`: Number of samples to generate
- `config`: Custom AugmentationConfig (uses defaults if None)
- `verbose`: Print results to stdout
- `save_path`: Path to save JSON file

**Returns:** List of augmentation parameter dictionaries

### gen_augmentation_params

```python
gen_augmentation_params(
    seed_id: int,
    config: Optional[AugmentationConfig] = None
) -> Dict[str, Any]
```

Generate augmentation parameters for a single sample.

**Parameters:**
- `seed_id`: Unique sample identifier
- `config`: Custom AugmentationConfig

**Returns:** Dictionary with augmentation parameters

### gen_augmentation_seed

```python
gen_augmentation_seed(seed_id: int, augmentation_depth: int = 10) -> str
```

Generate deterministic hash seed using Fibonacci chain.

**Parameters:**
- `seed_id`: Unique sample identifier
- `augmentation_depth`: Number of hash iterations

**Returns:** SHA256 hash as hexadecimal string

### save_augmentation_chain

```python
save_augmentation_chain(
    params_list: List[Dict[str, Any]],
    filepath: str,
    config: Optional[AugmentationConfig] = None,
    include_stats: bool = True
) -> None
```

Save augmentation chain to JSON file with optional statistics.

**Parameters:**
- `params_list`: Augmentation parameters to save
- `filepath`: Output file path
- `config`: Configuration to include in metadata
- `include_stats`: Include statistics in output

### load_augmentation_chain

```python
load_augmentation_chain(filepath: str) -> List[Dict[str, Any]]
```

Load augmentation chain from JSON file.

**Parameters:**
- `filepath`: Path to JSON file

**Returns:** List of augmentation parameter dictionaries

### compute_statistics

```python
compute_statistics(params_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]
```

Compute statistics for augmentation parameters.

**Parameters:**
- `params_list`: Augmentation parameters

**Returns:** Dictionary with mean, stdev, min, max for each parameter

### get_preset

```python
get_preset(preset_name: str) -> AugmentationConfig
```

Get a preset configuration.

**Parameters:**
- `preset_name`: "mild", "moderate", or "aggressive"

**Returns:** AugmentationConfig instance