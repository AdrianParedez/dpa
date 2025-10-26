import json
import tempfile
from pathlib import Path

import pytest

from src.dpa import (
    PRESETS,
    AugmentationConfig,
    compute_statistics,
    fib,
    gen_augmentation_params,
    gen_augmentation_seed,
    generate_augmentation_chain,
    get_preset,
    load_augmentation_chain,
    save_augmentation_chain,
)


class TestAugmentationConfig:
    def test_default_config(self):
        config = AugmentationConfig()
        assert config.rotation_range == (-30, 30)
        assert config.brightness_range == (0.8, 1.2)
        assert config.augmentation_depth == 10

    def test_invalid_range_lower_greater_than_upper(self):
        with pytest.raises(ValueError):
            AugmentationConfig(rotation_range=(30, 10))

    def test_invalid_augmentation_depth(self):
        with pytest.raises(ValueError):
            AugmentationConfig(augmentation_depth=0)

    def test_custom_config(self):
        config = AugmentationConfig(
            rotation_range=(-45, 45),
            brightness_range=(0.7, 1.3),
        )
        assert config.rotation_range == (-45, 45)
        assert config.brightness_range == (0.7, 1.3)


class TestFibonacci:
    def test_fib_zero(self):
        assert fib(0) == 0

    def test_fib_one(self):
        assert fib(1) == 1

    def test_fib_sequence(self):
        assert fib(2) == 1
        assert fib(3) == 2
        assert fib(4) == 3
        assert fib(5) == 5
        assert fib(10) == 55

    def test_fib_negative(self):
        with pytest.raises(ValueError):
            fib(-1)

    def test_fib_memoization(self):
        fib.cache_clear()
        fib(5)
        info = fib.cache_info()
        assert info.hits > 0


class TestAugmentationSeed:
    def test_gen_seed_deterministic(self):
        seed1 = gen_augmentation_seed(0)
        seed2 = gen_augmentation_seed(0)
        assert seed1 == seed2

    def test_gen_seed_different_ids(self):
        seed0 = gen_augmentation_seed(0)
        seed1 = gen_augmentation_seed(1)
        assert seed0 != seed1

    def test_gen_seed_negative_id(self):
        with pytest.raises(ValueError):
            gen_augmentation_seed(-1)

    def test_gen_seed_invalid_depth(self):
        with pytest.raises(ValueError):
            gen_augmentation_seed(0, augmentation_depth=0)

    def test_gen_seed_hex_format(self):
        seed = gen_augmentation_seed(0)
        assert isinstance(seed, str)
        assert len(seed) == 64
        assert all(c in "0123456789abcdef" for c in seed)


class TestAugmentationParams:
    def test_gen_params_deterministic(self):
        params1 = gen_augmentation_params(0)
        params2 = gen_augmentation_params(0)
        assert params1 == params2

    def test_gen_params_different_ids(self):
        params0 = gen_augmentation_params(0)
        params1 = gen_augmentation_params(1)
        assert params0 != params1

    def test_gen_params_negative_id(self):
        with pytest.raises(ValueError):
            gen_augmentation_params(-1)

    def test_gen_params_keys(self):
        params = gen_augmentation_params(0)
        expected_keys = {"rotation", "brightness", "noise", "scale", "contrast", "hash"}
        assert set(params.keys()) == expected_keys

    def test_gen_params_within_range(self):
        config = AugmentationConfig()
        params = gen_augmentation_params(0, config)

        assert config.rotation_range[0] <= params["rotation"] <= config.rotation_range[1]
        assert config.brightness_range[0] <= params["brightness"] <= config.brightness_range[1]
        assert config.noise_range[0] <= params["noise"] <= config.noise_range[1]
        assert config.scale_range[0] <= params["scale"] <= config.scale_range[1]
        assert config.contrast_range[0] <= params["contrast"] <= config.contrast_range[1]

    def test_gen_params_custom_config(self):
        config = AugmentationConfig(rotation_range=(-10, 10))
        for _ in range(100):
            params = gen_augmentation_params(0, config)
            assert -10 <= params["rotation"] <= 10


class TestStatistics:
    def test_compute_statistics_single_value(self):
        params_list = [
            {"rotation": 5, "brightness": 1.0, "noise": 0.05, "scale": 1.0, "contrast": 1.0}
        ]
        stats = compute_statistics(params_list)

        assert stats["rotation"]["mean"] == 5
        assert stats["rotation"]["min"] == 5
        assert stats["rotation"]["max"] == 5
        assert stats["rotation"]["count"] == 1

    def test_compute_statistics_multiple_values(self):
        params_list = [
            {"rotation": 0, "brightness": 1.0, "noise": 0.0, "scale": 1.0, "contrast": 1.0},
            {"rotation": 10, "brightness": 1.0, "noise": 0.0, "scale": 1.0, "contrast": 1.0},
        ]
        stats = compute_statistics(params_list)

        assert stats["rotation"]["mean"] == 5
        assert stats["rotation"]["min"] == 0
        assert stats["rotation"]["max"] == 10
        assert stats["rotation"]["count"] == 2

    def test_compute_statistics_keys(self):
        params_list = [gen_augmentation_params(i) for i in range(5)]
        stats = compute_statistics(params_list)

        expected_keys = {"rotation", "brightness", "noise", "scale", "contrast"}
        assert set(stats.keys()) == expected_keys

    def test_compute_statistics_stdev(self):
        params_list = [
            {"rotation": 0, "brightness": 1.0, "noise": 0.0, "scale": 1.0, "contrast": 1.0},
        ]
        stats = compute_statistics(params_list)
        assert stats["rotation"]["stdev"] == 0.0


class TestPersistence:
    def test_save_and_load_augmentation_chain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_augmentations.json"
            params_list = [gen_augmentation_params(i) for i in range(5)]

            save_augmentation_chain(params_list, str(filepath))
            loaded = load_augmentation_chain(str(filepath))

            assert loaded == params_list

    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "test.json"
            params_list = [gen_augmentation_params(0)]

            save_augmentation_chain(params_list, str(filepath))
            assert filepath.exists()

    def test_save_with_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            params_list = [gen_augmentation_params(i) for i in range(5)]

            save_augmentation_chain(params_list, str(filepath), include_stats=True)

            with open(filepath) as f:
                data = json.load(f)

            assert "statistics" in data
            assert "rotation" in data["statistics"]

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_augmentation_chain("nonexistent_file.json")

    def test_save_includes_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            config = AugmentationConfig(rotation_range=(-45, 45))
            params_list = [gen_augmentation_params(0, config)]

            save_augmentation_chain(params_list, str(filepath), config=config)

            with open(filepath) as f:
                data = json.load(f)

            assert data["metadata"]["config"] is not None
            assert data["metadata"]["config"]["rotation_range"] == [-45, 45]


class TestPresets:
    def test_get_preset_mild(self):
        config = get_preset("mild")
        assert config.rotation_range == (-15, 15)
        assert config.brightness_range == (0.9, 1.1)

    def test_get_preset_moderate(self):
        config = get_preset("moderate")
        assert config.rotation_range == (-30, 30)

    def test_get_preset_aggressive(self):
        config = get_preset("aggressive")
        assert config.rotation_range == (-45, 45)

    def test_get_preset_invalid(self):
        with pytest.raises(ValueError):
            get_preset("nonexistent")

    def test_presets_exist(self):
        assert "mild" in PRESETS
        assert "moderate" in PRESETS
        assert "aggressive" in PRESETS


class TestGenerateChain:
    def test_generate_chain_deterministic(self):
        results1 = generate_augmentation_chain(5)
        results2 = generate_augmentation_chain(5)
        assert results1 == results2

    def test_generate_chain_length(self):
        results = generate_augmentation_chain(10)
        assert len(results) == 10

    def test_generate_chain_negative_samples(self):
        with pytest.raises(ValueError):
            generate_augmentation_chain(-1)

    def test_generate_chain_zero_samples(self):
        results = generate_augmentation_chain(0)
        assert results == []

    def test_generate_chain_with_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            results = generate_augmentation_chain(5, save_path=str(filepath))

            assert filepath.exists()
            assert len(results) == 5

    def test_generate_chain_with_preset(self):
        config = get_preset("aggressive")
        results = generate_augmentation_chain(3, config=config)

        for params in results:
            assert -45 <= params["rotation"] <= 45


class TestIntegration:
    def test_full_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "augmentations.json"
            config = get_preset("moderate")

            results = generate_augmentation_chain(10, config, save_path=str(filepath))
            loaded = load_augmentation_chain(str(filepath))
            stats = compute_statistics(loaded)

            assert len(results) == 10
            assert len(loaded) == 10
            assert len(stats) == 5
            assert all(stat["count"] == 10 for stat in stats.values())

    def test_reproducibility_across_runs(self):
        config = get_preset("mild")
        results1 = generate_augmentation_chain(20, config)
        results2 = generate_augmentation_chain(20, config)

        assert results1 == results2
