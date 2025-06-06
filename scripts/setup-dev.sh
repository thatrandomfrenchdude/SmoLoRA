#!/bin/bash

# SmoLoRA Development Setup Script
# This script sets up a complete development environment for SmoLoRA

set -e  # Exit on any error

echo "🚀 Setting up SmoLoRA development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.13"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv smolora-dev
source smolora-dev/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Install development dependencies
echo "🔧 Installing development dependencies..."
pip install -r dev-requirements.txt

# install smolora package in editable mode
echo "🔍 Installing SmoLoRA package in editable mode..."
pip install -e .

# Install pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pip install pre-commit
pre-commit install

# Run initial code quality checks
echo "🔍 Running initial code quality checks..."

echo "  - Running Black formatter..."
black --check . || {
    echo "  ⚠️  Code formatting issues found. Run 'black .' to fix."
}

echo "  - Running isort import sorter..."
isort --check-only . || {
    echo "  ⚠️  Import sorting issues found. Run 'isort .' to fix."
}

echo "  - Running flake8 linter..."
flake8 . || {
    echo "  ⚠️  Linting issues found. Check output above."
}

echo "  - Running mypy type checker..."
mypy src/ || {
    echo "  ⚠️  Type checking issues found. Check output above."
}

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v || {
    echo "  ⚠️  Some tests failed. Check output above."
}

# Generate test coverage report
echo "📊 Generating test coverage report..."
pytest --cov=src --cov-report=html tests/

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source smolora-dev/bin/activate"
echo "2. Read the contributing guidelines: cat CONTRIBUTING.md"
echo "3. Check the documentation: ls docs/"
echo "4. View test coverage: open htmlcov/index.html"
echo ""
echo "Development commands:"
echo "- Format code: black ."
echo "- Sort imports: isort ."
echo "- Lint code: flake8 ."
echo "- Type check: mypy src/"
echo "- Run tests: pytest tests/ -v"
echo "- Run pre-commit: pre-commit run --all-files"
echo ""
echo "Happy coding! 🐍✨"
