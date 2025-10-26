# DPA - Deterministic Procedural Augmentation

A Python library for generating deterministic, reproducible data augmentation parameters using Fibonacci-based seeding.

## Installation

```bash
git clone https://github.com/AdrianParedez/dpa.git
cd dpa
```

No external dependencies required (Python 3.8+).

## Quick Start

```python
from src.dpa import generate_augmentation_chain, get_preset

config = get_preset("moderate")
results = generate_augmentation_chain(100, config, save_path="augmentations.json")
```

## Key Features

- Deterministic and reproducible augmentation generation
- Three presets: mild, moderate, aggressive
- Save and load augmentation chains as JSON
- Automatic statistics computation
- 42 comprehensive tests

## Usage

See `example_usage.py` for full examples or check the [documentation](docs/).

## Testing

```bash
pytest test_dpa.py -v
```

## License

MIT License - see LICENSE file for details