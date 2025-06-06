# Contributing to SmoLoRA

Thank you for your interest in contributing to SmoLoRA! This document outlines the guidelines and requirements for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Testing Requirements](#testing-requirements)
- [Code Style and Linting](#code-style-and-linting)
- [Contribution Workflow](#contribution-workflow)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you are expected to uphold our code of conduct. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the coding standards
4. Run tests and ensure they pass
5. Add relevant documentation for your changes
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.13 or higher
- Virtual environment tool (venv, conda, etc.)

### Installation

#### Automatic
```bash
# Clone the repository
git clone https://github.com/thatrandomfrenchdude/SmoLoRA.git
cd SmoLoRA

# Run the setup script to create a virtual environment and install dependencies
chmod +x scripts/setup-dev.sh
./setup-dev.sh
```

#### Manual Setup

1. Create and activate a virtual environment:
```bash
python -m venv smolora-dev-venv
source smolora-dev-venv/bin/activate  # On macOS/Linux
# or
smolora-dev-venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install development dependencies:
```bash
pip install -r dev-requirements.txt
```

4. # Install the package in editable mode
```bash
pip install -e .  # Install the package in editable mode
```

5. Install and set up pre-commit hooks:
```bash
pip install pre-commit
pre-commit install  # Set up pre-commit hooks
```

## Testing Requirements

**All new functionality must include comprehensive tests.** We use pytest for testing with extensive mocking for ML components.

### Test Structure

Our tests follow these patterns:

1. **Unit Tests**: Test individual functions and methods in isolation
2. **Integration Tests**: Test component interactions
3. **Mock-Heavy Testing**: Use `unittest.mock` to avoid loading actual ML models

### Required Test Patterns

#### 1. Mock External Dependencies
```python
from unittest.mock import patch, MagicMock

def test_your_feature():
    with patch("your_module.AutoModelForCausalLM") as mock_model_cls, \
         patch("your_module.AutoTokenizer") as mock_tokenizer_cls:
        # Your test code here
        pass
```

#### 2. Test File Structure
- Create tests in `test_*.py` files
- Use descriptive test function names: `test_feature_name_scenario`
- Include docstrings explaining what each test validates

#### 3. Temporary Directory Usage
```python
def test_with_files(tmp_path):
    # Use tmp_path fixture for file operations
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    # Your test logic here
```

#### 4. Comprehensive Assertions
- Test both success and error conditions
- Verify mock calls with expected parameters
- Assert return values and side effects

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest test_smolora.py

# Run specific test function
pytest test_smolora.py::test_function_name
```

### Test Coverage Requirements

- New code must have at least 80% test coverage
- Critical paths (model loading, training, inference) must have 95%+ coverage
- Include tests for error conditions and edge cases

## Code Style and Linting

We maintain consistent code quality through automated tools:

### Formatting with Black
```bash
# Format all Python files
black .

# Check formatting without making changes
black --check .
```

### Import Sorting with isort
```bash
# Sort imports
isort .

# Check import sorting
isort --check-only .
```

### Linting with flake8
```bash
# Run linting
flake8 .
```

### Type Checking with mypy
```bash
# Run type checking
mypy smoLoRA.py prepare_dataset.py local_text.py
```

### Code Style Guidelines

1. **Line Length**: Maximum 88 characters (Black default)
2. **Import Organization**:
   - Standard library imports first
   - Third-party imports second
   - Local imports last
3. **Type Hints**: Use type hints for all public functions and methods
4. **Docstrings**: Use Google-style docstrings for all public functions

Example:
```python
def prepare_dataset(
    source: str,
    text_field: str = "text",
    chunk_size: int = 0
) -> Dataset:
    """Prepare a dataset from various file formats.

    Args:
        source: Path to data source (folder, .jsonl, or .csv file)
        text_field: Field name containing text data
        chunk_size: Split texts into chunks of this many words

    Returns:
        HuggingFace Dataset with standardized 'text' field

    Raises:
        ValueError: If file type cannot be inferred or is unsupported
    """
```

## Contribution Workflow

### Branch Naming
- Feature branches: `feature/description-of-feature`
- Bug fixes: `fix/description-of-bug`
- Documentation: `docs/description-of-changes`

### Commit Messages
Use conventional commit format:
```
type(scope): description

Examples:
feat(core): add support for custom LoRA configurations
fix(dataset): handle empty files in prepare_dataset
docs(api): update SmoLoRA class documentation
test(integration): add tests for end-to-end training workflow
```

### Pre-commit Checklist
Before submitting a pull request:

- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`black .`)
- [ ] Imports are sorted (`isort .`)
- [ ] No linting errors (`flake8 .`)
- [ ] Type checking passes (`mypy`)
- [ ] Documentation is updated if needed
- [ ] New tests are added for new functionality

## Pull Request Guidelines

### PR Title and Description
- Use clear, descriptive titles
- Include a detailed description of changes
- Reference any related issues
- List breaking changes if any

### PR Checklist Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Test coverage maintained/improved

## Documentation
- [ ] Code is documented with type hints and docstrings
- [ ] README updated if needed
- [ ] Comprehensive documentation updated if needed
```

### Review Process
1. Automated checks must pass (formatting, linting, tests)
2. At least one maintainer review required
3. Address all review feedback
4. Squash commits before merge if requested

## Documentation

### Code Documentation
- All public classes and functions must have docstrings
- Use type hints consistently
- Include examples in docstrings for complex functions

### API Documentation
If adding new public APIs:
1. Update relevant documentation in `docs/`
2. Add usage examples
3. Document any configuration options
4. Include common pitfalls or gotchas

### README Updates
Update the main README.md if your contribution:
- Adds new features
- Changes installation requirements
- Modifies the public API
- Adds new configuration options

## Questions and Support

- Open an issue for questions about contributing
- Tag maintainers for urgent questions
- Check existing issues and PRs before creating new ones

Thank you for contributing to SmoLoRA! 🚀
