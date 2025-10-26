# DPA - Deterministic Procedural Augmentation

A Python library for generating deterministic, reproducible data augmentation parameters using Fibonacci-based seeding.

## Installation

```bash
git clone https://github.com/AdrianParedez/dpa.git
cd dpa
```

**Requirements:** Python 3.12+

**Runtime:** No external dependencies

**Development:** Install dev dependencies (optional)

```bash
pip install -r requirements-dev.txt
```

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

See `examples/example_usage.py` for full examples or check the [documentation](docs/).

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Lint with Ruff:

```bash
ruff check .
```

## Documentation

- [API Reference](docs/API.md) - Complete function documentation
- [Usage Guide](docs/GUIDE.md) - Examples and best practices
- [Architecture](docs/ARCHITECTURE.md) - Deep dive into how DPA works

## License

MIT License - see LICENSE file for details