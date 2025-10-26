import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dpa import (
    AugmentationConfig,
    compute_statistics,
    generate_augmentation_chain,
    get_preset,
    load_augmentation_chain,
)


def demo_basic_usage():
    print("=== Demo 1: Basic Usage ===\n")

    generate_augmentation_chain(5, verbose=True)
    print("\nGenerated 5 augmentation parameters\n")


def demo_with_presets():
    print("=== Demo 2: Using Presets ===\n")

    for preset_name in ["mild", "moderate", "aggressive"]:
        print(f"\n--- {preset_name.upper()} Preset ---")
        config = get_preset(preset_name)
        generate_augmentation_chain(3, config, verbose=True)


def demo_save_and_load():
    print("\n=== Demo 3: Save and Load ===\n")

    config = get_preset("moderate")
    filepath = Path(__file__).parent / "demo_augmentations.json"

    print(f"Generating 20 samples and saving to {filepath.name}...\n")
    generate_augmentation_chain(20, config, verbose=False, save_path=str(filepath))

    print(f"\nLoading augmentations from {filepath.name}...\n")
    loaded = load_augmentation_chain(str(filepath))

    print(f"Successfully loaded {len(loaded)} samples")


def demo_statistics():
    print("\n=== Demo 4: Statistics ===\n")

    filepath = Path(__file__).parent / "demo_augmentations.json"
    loaded = load_augmentation_chain(str(filepath))
    stats = compute_statistics(loaded)

    print("Parameter Statistics:")
    print("-" * 75)
    print(f"{'Parameter':<12} | {'Mean':>8} | {'StDev':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 75)

    for param, stat_dict in stats.items():
        print(
            f"{param:<12} | {stat_dict['mean']:>8.3f} | "
            f"{stat_dict['stdev']:>8.3f} | {stat_dict['min']:>8.3f} | "
            f"{stat_dict['max']:>8.3f}"
        )
    print("-" * 75)


def demo_reproducibility():
    print("\n=== Demo 5: Reproducibility ===\n")

    config = get_preset("mild")

    print("Generating augmentations with 'mild' preset...")
    results1 = generate_augmentation_chain(10, config, verbose=False)

    print("Generating again with same config...")
    results2 = generate_augmentation_chain(10, config, verbose=False)

    if results1 == results2:
        print("\nReproducibility verified!")
        print("Identical results across runs - deterministic augmentation working!\n")
    else:
        print("\nReproducibility failed!")


def demo_custom_config():
    print("=== Demo 6: Custom Configuration ===\n")

    custom_config = AugmentationConfig(
        rotation_range=(-60, 60),
        brightness_range=(0.6, 1.4),
        noise_range=(0, 0.2),
        scale_range=(0.6, 1.4),
        contrast_range=(0.5, 1.5),
        augmentation_depth=15,
    )

    print("Custom Config:")
    print(f"  Rotation: {custom_config.rotation_range}")
    print(f"  Brightness: {custom_config.brightness_range}")
    print(f"  Noise: {custom_config.noise_range}")
    print(f"  Scale: {custom_config.scale_range}")
    print(f"  Contrast: {custom_config.contrast_range}")
    print(f"  Depth: {custom_config.augmentation_depth}\n")

    generate_augmentation_chain(5, custom_config, verbose=True)


if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("DPA (Deterministic Procedural Augmentation) - Example Usage")
    print("=" * 75 + "\n")

    demo_basic_usage()
    demo_with_presets()
    demo_save_and_load()
    demo_statistics()
    demo_reproducibility()
    demo_custom_config()

    print("\n" + "=" * 75)
    print("All demos completed!")
    print("=" * 75 + "\n")
