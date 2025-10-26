import hashlib
import json
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters."""

    rotation_range: Tuple[float, float] = (-30, 30)
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    noise_range: Tuple[float, float] = (0, 0.1)
    scale_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    augmentation_depth: int = 10

    def __post_init__(self) -> None:
        """Validate ranges after initialization."""
        for name, (low, high) in [
            ("rotation", self.rotation_range),
            ("brightness", self.brightness_range),
            ("noise", self.noise_range),
            ("scale", self.scale_range),
            ("contrast", self.contrast_range),
        ]:
            if low >= high:
                raise ValueError(f"{name}_range: lower bound must be < upper bound")

        if self.augmentation_depth < 1:
            raise ValueError("augmentation_depth must be >= 1")


PRESETS = {
    "mild": AugmentationConfig(
        rotation_range=(-15, 15),
        brightness_range=(0.9, 1.1),
        noise_range=(0, 0.05),
        scale_range=(0.9, 1.1),
        contrast_range=(0.85, 1.15),
    ),
    "moderate": AugmentationConfig(
        rotation_range=(-30, 30),
        brightness_range=(0.8, 1.2),
        noise_range=(0, 0.1),
        scale_range=(0.8, 1.2),
        contrast_range=(0.7, 1.3),
    ),
    "aggressive": AugmentationConfig(
        rotation_range=(-45, 45),
        brightness_range=(0.7, 1.3),
        noise_range=(0, 0.15),
        scale_range=(0.7, 1.3),
        contrast_range=(0.6, 1.4),
    ),
}


@lru_cache(maxsize=None)
def fib(n: int) -> int:
    """Compute nth Fibonacci number with memoization."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fib(n - 1) + fib(n - 2)


def gen_augmentation_seed(seed_id: int, augmentation_depth: int = 10) -> str:
    """
    Generate deterministic hash seed using Fibonacci chain.

    Args:
        seed_id: Unique sample identifier
        augmentation_depth: Number of hash iterations

    Returns:
        SHA256 hash as hexadecimal string

    Raises:
        ValueError: If seed_id is negative or augmentation_depth < 1
    """
    if seed_id < 0:
        raise ValueError("seed_id must be non-negative")
    if augmentation_depth < 1:
        raise ValueError("augmentation_depth must be >= 1")

    prev_hash = hashlib.sha256(str(seed_id).encode()).hexdigest()

    for i in range(augmentation_depth):
        try:
            fib_val = str(fib(i))
            combined = prev_hash + fib_val
            prev_hash = hashlib.sha256(combined.encode()).hexdigest()
        except Exception as e:
            raise RuntimeError(f"Error computing Fibonacci seed at depth {i}: {e}") from e

    return prev_hash


def gen_augmentation_params(
    seed_id: int, config: Optional[AugmentationConfig] = None
) -> Dict[str, Any]:
    """
    Generate deterministic augmentation parameters.

    Args:
        seed_id: Unique sample identifier
        config: AugmentationConfig instance (uses defaults if None)

    Returns:
        Dictionary with augmentation parameters

    Raises:
        ValueError: If seed_id is invalid
    """
    if seed_id < 0:
        raise ValueError("seed_id must be non-negative")

    if config is None:
        config = AugmentationConfig()

    hash_seed = gen_augmentation_seed(seed_id, config.augmentation_depth)
    seed_int = int(hash_seed, 16) % (2**32)
    random.seed(seed_int)

    return {
        "rotation": random.uniform(*config.rotation_range),
        "brightness": random.uniform(*config.brightness_range),
        "noise": random.uniform(*config.noise_range),
        "scale": random.uniform(*config.scale_range),
        "contrast": random.uniform(*config.contrast_range),
        "hash": hash_seed,
    }


def compute_statistics(
    params_list: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics (mean, std, min, max) for augmentation parameters.

    Args:
        params_list: List of augmentation parameter dictionaries

    Returns:
        Dictionary with statistics for each parameter
    """
    stats = {}
    param_keys = ["rotation", "brightness", "noise", "scale", "contrast"]

    for key in param_keys:
        values = [p[key] for p in params_list]
        stats[key] = {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    return stats


def save_augmentation_chain(
    params_list: List[Dict[str, Any]],
    filepath: str,
    config: Optional[AugmentationConfig] = None,
    include_stats: bool = True,
) -> None:
    """
    Save augmentation chain to JSON file with optional statistics.

    Args:
        params_list: List of augmentation parameter dictionaries
        filepath: Path to save JSON file
        config: AugmentationConfig used (included in metadata)
        include_stats: If True, include statistics in output

    Raises:
        OSError: If file cannot be written
    """
    try:
        output = {
            "metadata": {
                "num_samples": len(params_list),
                "config": asdict(config) if config else None,
            },
            "augmentations": params_list,
        }

        if include_stats:
            output["statistics"] = compute_statistics(params_list)

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        sys.stdout.write(f"Saved {len(params_list)} augmentations to {filepath}\n")

    except OSError as e:
        raise OSError(f"Failed to save augmentation chain: {e}") from e


def load_augmentation_chain(filepath: str) -> List[Dict[str, Any]]:
    """
    Load augmentation chain from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        List of augmentation parameter dictionaries

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is not valid JSON
    """
    try:
        with open(filepath) as f:
            data = json.load(f)

        sys.stdout.write(
            f"Loaded {data['metadata']['num_samples']} augmentations from {filepath}\n"
        )
        return data["augmentations"]

    except FileNotFoundError:
        raise FileNotFoundError(f"Augmentation file not found: {filepath}") from None
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {filepath}: {e}", "", 0) from e


def get_preset(preset_name: str) -> AugmentationConfig:
    """
    Get a preset configuration.

    Args:
        preset_name: Name of preset ("mild", "moderate", "aggressive")

    Returns:
        AugmentationConfig instance

    Raises:
        ValueError: If preset_name is not recognized
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(PRESETS.keys())}")

    return PRESETS[preset_name]


def generate_augmentation_chain(
    num_samples: int,
    config: Optional[AugmentationConfig] = None,
    verbose: bool = False,
    save_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate augmentation parameters for multiple samples.

    Args:
        num_samples: Number of samples to augment
        config: AugmentationConfig instance (uses defaults if None)
        verbose: If True, print results to stdout
        save_path: If provided, save results to JSON file

    Returns:
        List of augmentation parameter dictionaries

    Raises:
        ValueError: If num_samples < 0
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")

    if config is None:
        config = AugmentationConfig()

    results = []
    output = []

    for sample_id in range(num_samples):
        params = gen_augmentation_params(sample_id, config)
        results.append(params)

        if verbose:
            output.append(
                f"Sample({sample_id}) -> rotation={params['rotation']:.2f}° "
                f"brightness={params['brightness']:.2f} "
                f"noise={params['noise']:.3f} "
                f"scale={params['scale']:.2f} "
                f"contrast={params['contrast']:.2f}\n"
            )

    if verbose:
        sys.stdout.write("".join(output))

    if save_path:
        save_augmentation_chain(results, save_path, config, include_stats=True)

    return results
