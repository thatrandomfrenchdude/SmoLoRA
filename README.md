![SmoLoRA Logo](logo.png)

# SmoLoRA: Edge Language Model Fine-Tuning & Inference Toolkit

A lightweight, developer-friendly Python package for fine-tuning small language models using LoRA adapters and running on-device inference. Built for flexibility and rapid prototyping, SmoLoRA allows you to train, save, load, and generate text from language models with a clean, modular architecture.

## Table of Contents
- [🔧 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📂 Custom Datasets](#-custom-datasets)
- [🛠️ Knobs and Levers](#️-knobs-and-levers)
- [🧪 Testing](#-testing)
- [📁 Project Structure](#-project-structure)
- [📚 Documentation](#-documentation)

## 🔧 Installation

### Quick Install
```bash
pip install smolora
```

### Development Setup
For developers who want to contribute or modify the code. Please review the [Contributing](CONTRIBUTING.md#) section for guidelines, then follow these steps to set up your development environment:
```bash
# Clone the repository
git clone https://github.com/thatrandomfrenchdude/smolora.git
cd smolora

# Run the development setup script
chmod +x scripts/setup-dev.sh
./scripts/setup-dev.sh
```

This will create a virtual environment, install all dependencies, and set up pre-commit hooks.

## 🚀 Quick Start

```python
from smolora import SmoLoRA

# Initialize the trainer
trainer = SmoLoRA(
    base_model_name="microsoft/Phi-1.5", # or any HuggingFace model
    dataset_name="yelp_review_full", # HuggingFace dataset
    text_field="text", # Field containing text data
    output_dir="./output_model" # Directory to save the fine-tuned model
)

# Fine-tune the model
trainer.train()

# Save the adapter and merge with base model
trainer.save()

# Load the merged model for inference
model, tokenizer = trainer.load_model("./output_model/final_merged")

# Generate text
prompt = "Write a review about a great coffee shop."
result = trainer.inference(prompt)
print("Generated output:", result)
```

## 📂 Custom Datasets

SmoLoRA supports multiple data formats through the `smolora.dataset` module. You can use HuggingFace datasets, local text files, CSV, or JSONL files for training.

You can use the `prepare_dataset.py` tool to convert your raw text, CSV, or JSONL data into a HuggingFace `Dataset` ready for fine-tuning.

### Text Files
```python
from smolora.dataset import load_text_data

# Load all .txt files from a directory
dataset = load_text_data("./text_directory/")

# Use with SmoLoRA
trainer = SmoLoRA(
    base_model_name="microsoft/Phi-1.5",
    dataset_name=dataset,  # Use the prepared dataset directly
    output_dir="./custom_model"
)
```

### JSONL Files
```python
from smolora.dataset import prepare_dataset

# Prepare JSONL data
dataset = prepare_dataset(
    source="data.jsonl",
    text_field="text",  # Field containing the text data
    chunk_size=100      # Optional: words per chunk
)

# Use with SmoLoRA
```

### CSV Files
```python
from smolora.dataset import prepare_dataset

# Prepare CSV data
dataset = prepare_dataset(
    source="data.csv",
    text_field="content",
    file_type="csv"  # Explicitly specify format
)

# Use with SmoLoRA
```

## 🛠️ Knobs and Levers

### SmoLoRA Configuration

The `SmoLoRA` class accepts several parameters for customization:

```python
trainer = SmoLoRA(
    base_model_name="microsoft/Phi-1.5",  # Any HuggingFace model
    dataset_name="yelp_review_full",      # HF dataset or custom Dataset object
    text_field="text",                    # Field containing text data
    output_dir="./fine_tuned_model"       # Output directory
)
```

### LoRA Configuration

You can customize the LoRA adapter settings by modifying the `peft_config` after initialization:

```python
trainer = SmoLoRA(...)
trainer.peft_config.r = 16              # Rank
trainer.peft_config.lora_alpha = 32     # Alpha scaling
trainer.peft_config.lora_dropout = 0.1  # Dropout
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/smolora --cov-report=html

# Run specific test categories
pytest tests/ -m unit        # Unit tests only
pytest tests/ -m integration # Integration tests only
```

The test suite includes:
- Unit tests for core functionality
- Dataset loading and preparation tests
- Mock-based training pipeline tests
- Integration tests with sample data

## 📁 Project Structure

```
smolora/
├── src/smolora/           # Main package source
│   ├── __init__.py        # Package initialization
│   ├── core.py            # Main SmoLoRA class
│   └── dataset.py         # Dataset handling utilities
├── examples/              # Usage examples
│   └── usage.py           # Basic usage example
├── tests/                 # Test suite
│   └── test_smolora.py    # Comprehensive tests
├── scripts/               # Development scripts
│   └── setup-dev.sh       # Development environment setup
├── docs/                  # Documentation
│   ├── api-reference.md   # API documentation
│   ├── architecture.md    # Architecture overview
│   └── ...               # Additional documentation
├── pyproject.toml         # Project configuration
├── requirements.txt       # Production dependencies
├── dev-requirements.txt   # Development dependencies
└── README.md             # This file
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[API Reference](docs/api-reference.md)**: Complete API documentation
- **[Architecture](docs/architecture.md)**: System design and components
- **[Core Components](docs/core-components.md)**: Detailed explanation of main classes and modules
- **[Dataset Handling](docs/dataset-handling.md)**: Data preparation guide
- **[Model Management](docs/model-management.md)**: Model loading, saving, and inference workflows
- **[Testing Strategy](docs/testing-strategy.md)**: Testing approach and mock patterns
- **[Training Pipeline](docs/training-pipeline.md)**: Deep dive into the LoRA training process
- **[Development Guide](docs/development-guide.md)**: Contributing guidelines

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
