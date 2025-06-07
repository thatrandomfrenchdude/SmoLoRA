![SmoLoRA Logo](logo.png)

# SmoLoRA: Edge Language Model Fine-Tuning & Inference Toolkit

A lightweight, developer-friendly Python package for fine-tuning small language models using LoRA adapters and running on-device inference. Built for flexibility and rapid prototyping, SmoLoRA allows you to train, save, load, and generate text from language models with a clean, modular architecture.

## Table of Contents
- [📦 Features](#📦-features)
- [🔧 Installation](#🔧-installation)
- [📁 Project Structure](#📁-project-structure)
- [🚀 Quick Start](#🚀-quick-start)
- [📂 Custom Dataset Handling](#📂-custom-dataset-handling)
- [🛠️ Advanced Usage](#🛠️-advanced-usage)
- [🧠 Tips & Best Practices](#🧠-tips--best-practices)
- [🧪 Testing](#🧪-testing)
- [📚 Documentation](#📚-documentation)

## 📦 Features

- **Easy Installation**: Simple pip install with all dependencies managed
- **Modular Architecture**: Clean separation between core functionality and utilities
- **Multiple Data Sources**: Support for HuggingFace datasets, local text files, CSV, and JSONL
- **LoRA Fine-tuning**: Efficient fine-tuning using PEFT LoRA adapters
- **Model Management**: Save, load, and merge adapters with base models
- **Comprehensive Testing**: Full test suite with mocking for reliable development
- **Developer Tools**: Pre-commit hooks, formatting, and linting included

## 🔧 Installation

### Quick Install
```bash
pip install smolora
```

### Development Setup
For developers who want to contribute or modify the code. Please review the [Contributing](CONTRIBUTING.md#) section for guidelines, then follow these steps to set up your development environment:
```bash
# Clone the repository
git clone https://github.com/username/smolora.git
cd smolora

# Run the development setup script
chmod +x scripts/setup-dev.sh
./scripts/setup-dev.sh
```

This will create a virtual environment, install all dependencies, and set up pre-commit hooks.

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

## 🚀 Quick Start

```python
from smolora import SmoLoRA

# Initialize the trainer
trainer = SmoLoRA(
    base_model_name="microsoft/Phi-1.5",
    dataset_name="yelp_review_full", # HuggingFace dataset
    text_field="text",
    output_dir="./output_model"
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

## 🛠️ Advanced Usage

### Configuration Options

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

### Training Configuration

Customize training parameters through the trainer's configuration:

```python
# Access training configuration
training_config = trainer.training_args
training_config.num_train_epochs = 3
training_config.per_device_train_batch_size = 2
training_config.learning_rate = 2e-4
```

## 🧠 Tips & Best Practices

- **Start Small**: Begin with smaller models like Phi-1.5 for faster iteration
- **Memory Management**: Use gradient checkpointing for larger models
- **Data Quality**: Clean and consistent training data leads to better results
- **Evaluation**: Monitor training loss and validate on held-out data
- **Chunking**: Use appropriate chunk sizes for your specific use case
- **Device Selection**: The toolkit automatically uses MPS on Apple Silicon Macs

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
