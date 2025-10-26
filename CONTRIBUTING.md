# Contributing to DPA

Thank you for your interest in contributing to DPA (Deterministic Procedural Augmentation)!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AdrianParedez/dpa.git
   cd dpa
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements-dev.txt
   pip install -e .
   ```

3. **Run tests**
   ```bash
   pytest tests/ -v
   ```

## Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation if needed

3. **Test your changes**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Run linting
   ruff check .
   
   # Test examples
   python examples/example_usage.py
   ```

4. **Submit a pull request**
   - Use the pull request template
   - Provide a clear description of changes
   - Reference any related issues

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write clear, descriptive docstrings
- Keep functions focused and small

## Testing

- Write tests for new functionality
- Ensure all tests pass before submitting
- Aim for good test coverage of core functionality
- Test examples to ensure they work correctly

## Documentation

- Update docstrings for new/changed functions
- Update relevant documentation files
- Add examples for new features
- Keep the README.md up to date

## Questions?

Feel free to open an issue if you have questions about contributing!