# SmoLoRA Developer Guide

**Level 3: Advanced Development and Contribution**

This guide is for developers who want to contribute to SmoLoRA, understand its internal architecture, implement new features, and maintain the codebase. It assumes familiarity with both basic usage and customization.

## Architecture Deep Dive

### Core Design Principles

SmoLoRA follows these architectural principles:

1. **Simplicity First**: Single class interface with sensible defaults
2. **Modularity**: Separate concerns for data, training, and inference
3. **Memory Efficiency**: Aggressive memory management for resource-constrained environments
4. **Device Agnostic**: Automatic device detection with fallback strategies

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        SmoLoRA Core                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Data Layer    │  │  Training Layer │  │ Inference    │ │
│  │                 │  │                 │  │ Layer        │ │
│  │ • Dataset Prep  │  │ • LoRA Config   │  │ • Model Load │ │
│  │ • Text Loading  │  │ • SFT Training  │  │ • Text Gen   │ │
│  │ • Preprocessing │  │ • Checkpointing │  │ • Batching   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Key Classes and Responsibilities

#### SmoLoRA (`smolora/core.py`)
- **Primary Interface**: Single entry point for all functionality
- **State Management**: Handles model, tokenizer, and configuration state
- **Workflow Orchestration**: Coordinates training, saving, and inference phases
- **Device Management**: Automatic device selection and memory optimization

#### DatasetHandler (`smolora/dataset.py`)
- **Format Detection**: Automatic file type detection (TXT, CSV, JSONL)
- **Preprocessing**: Text cleaning, chunking, and field mapping
- **Validation**: Data quality checks and error handling
- **HuggingFace Integration**: Seamless dataset loading and transformation

## Development Environment Setup

### Complete Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/smolora.git
cd smolora

# Create isolated environment
python -m venv smolora-dev
source smolora-dev/bin/activate  # On Windows: smolora-dev\Scripts\activate

# Install development dependencies
pip install -e .
pip install -r dev-requirements.txt

# Setup pre-commit hooks
pre-commit install

# Verify installation
python -c "from smolora import SmoLoRA; print('Success')"
```

### Development Dependencies

```txt
# Core testing
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0

# Code quality
black>=22.0.0
flake8>=6.0.0
isort>=5.12.0
mypy>=1.0.0

# Development tools
pre-commit>=3.0.0
jupyterlab>=4.0.0
ipython>=8.0.0

# Documentation
sphinx>=5.0.0
sphinx-rtd-theme>=1.2.0
```

## Core Implementation Details

### Device Management Strategy

```python
class DeviceManager:
    """Centralized device management with fallback strategies."""

    @staticmethod
    def get_optimal_device():
        """Get the best available device with comprehensive fallback."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @staticmethod
    def optimize_for_device(model, device):
        """Apply device-specific optimizations."""
        if device.type == "mps":
            # MPS-specific optimizations
            model.config.use_cache = False
        elif device.type == "cuda":
            # CUDA-specific optimizations
            model = model.half()  # Use FP16 if available

        return model.to(device)
```

### Memory Management Implementation

```python
class MemoryManager:
    """Advanced memory management for different scenarios."""

    def __init__(self, device):
        self.device = device

    def clear_cache(self):
        """Device-specific cache clearing."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

        # Python garbage collection
        import gc
        gc.collect()

    def get_memory_usage(self):
        """Get current memory usage statistics."""
        if self.device.type == "cuda":
            return {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }
        return {'status': 'unsupported_device'}

    def optimize_batch_size(self, base_batch_size=1):
        """Dynamically adjust batch size based on available memory."""
        try:
            if self.device.type == "cuda":
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                # Heuristic: Use 80% of free memory
                suggested_batch_size = min(base_batch_size * 4, int(free_memory / (2 * 1024**3)))  # 2GB per batch
                return max(1, suggested_batch_size)
        except:
            pass
        return base_batch_size
```

### Configuration Management

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class LoRAConfig:
    """Enhanced LoRA configuration with validation."""
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.r <= 0:
            raise ValueError("r must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if not 0 <= self.lora_dropout <= 1:
            raise ValueError("lora_dropout must be between 0 and 1")

@dataclass
class TrainingConfig:
    """Enhanced training configuration with intelligent defaults."""
    output_dir: str = "./fine_tuned_model"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_steps: int = 500
    warmup_steps: int = 50
    logging_steps: int = 10
    save_steps: int = 250
    save_total_limit: int = 2

    def get_effective_batch_size(self):
        """Calculate effective batch size."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def optimize_for_dataset_size(self, dataset_size: int):
        """Adjust parameters based on dataset size."""
        if dataset_size < 1000:
            self.num_train_epochs = 3
            self.max_steps = min(500, dataset_size // 2)
        elif dataset_size > 10000:
            self.num_train_epochs = 1
            self.max_steps = min(2000, dataset_size // 10)
```

## Advanced Testing Patterns

### Comprehensive Mock Strategy

```python
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from smolora import SmoLoRA

class TestFixtures:
    """Centralized test fixtures and mocks."""

    @pytest.fixture
    def mock_model(self):
        """Comprehensive model mock with all required attributes."""
        mock = MagicMock()
        mock.config.use_cache = True
        mock.config.model_type = "phi"
        mock.to.return_value = mock
        mock.parameters.return_value = [MagicMock() for _ in range(10)]
        mock.named_parameters.return_value = [
            ("layer.0.weight", MagicMock()),
            ("layer.1.weight", MagicMock())
        ]
        return mock

    @pytest.fixture
    def mock_tokenizer(self):
        """Comprehensive tokenizer mock."""
        mock = MagicMock()
        mock.pad_token_id = None
        mock.eos_token_id = 50256
        mock.padding_side = "right"
        mock.encode.return_value = [1, 2, 3, 4, 5]
        mock.decode.return_value = "mocked output"
        return mock

    @pytest.fixture
    def mock_dataset(self):
        """Mock dataset with realistic behavior."""
        mock = MagicMock()
        mock.map.return_value = mock
        mock.__iter__.return_value = iter([
            {"text": "sample 1"},
            {"text": "sample 2"}
        ])
        mock.__len__.return_value = 2
        return mock
```

### Integration Test Patterns

```python
class TestIntegration:
    """Integration tests with comprehensive mocking."""

    @patch('smolora.core.AutoModelForCausalLM')
    @patch('smolora.core.AutoTokenizer')
    @patch('smolora.core.load_dataset')
    @patch('smolora.core.SFTTrainer')
    def test_complete_workflow(self, mock_trainer, mock_load_dataset,
                              mock_tokenizer_cls, mock_model_cls):
        """Test complete workflow with realistic mocks."""

        # Setup mocks
        mock_model = self.setup_model_mock(mock_model_cls)
        mock_tokenizer = self.setup_tokenizer_mock(mock_tokenizer_cls)
        mock_dataset = self.setup_dataset_mock(mock_load_dataset)
        mock_sft_trainer = self.setup_trainer_mock(mock_trainer)

        # Execute workflow
        trainer = SmoLoRA("test-model", "test-dataset")
        trainer.train()
        trainer.save()
        result = trainer.inference("test prompt")

        # Verify interactions
        assert mock_model_cls.from_pretrained.called
        assert mock_trainer.called
        assert result is not None

    def setup_model_mock(self, mock_cls):
        """Setup realistic model mock."""
        mock = MagicMock()
        mock.config.use_cache = True
        mock.to.return_value = mock
        mock_cls.from_pretrained.return_value = mock
        return mock
```

### Performance Testing

```python
import time
import psutil
import pytest

class TestPerformance:
    """Performance benchmarking and regression tests."""

    def test_memory_usage_bounds(self):
        """Ensure memory usage stays within acceptable bounds."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Create and train model
        trainer = SmoLoRA("microsoft/Phi-1.5", "tiny_dataset")
        trainer.train()

        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory

        # Assert memory increase is reasonable (adjust threshold as needed)
        assert memory_increase < 4 * 1024**3  # 4GB limit

    def test_training_speed_regression(self):
        """Ensure training doesn't regress in speed."""
        start_time = time.time()

        trainer = SmoLoRA("microsoft/Phi-1.5", "benchmark_dataset")
        trainer.train()

        training_time = time.time() - start_time

        # Assert training completes within reasonable time
        assert training_time < 300  # 5 minutes maximum
```

## Feature Implementation Patterns

### Adding New Model Architectures

```python
class ModelRegistry:
    """Registry for supporting different model architectures."""

    _target_modules = {
        "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "phi": ["q_proj", "v_proj", "dense"],
        "gpt": ["c_attn", "c_proj"],
        "bloom": ["query_key_value", "dense"],
        "opt": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    }

    _special_configs = {
        "llama": {"r": 16, "lora_alpha": 32},
        "gpt": {"r": 12, "lora_alpha": 24}
    }

    @classmethod
    def get_target_modules(cls, model_name: str) -> List[str]:
        """Get appropriate target modules for model architecture."""
        model_type = cls._detect_model_type(model_name)
        return cls._target_modules.get(model_type, ["q_proj", "v_proj"])

    @classmethod
    def get_special_config(cls, model_name: str) -> Dict[str, Any]:
        """Get special configuration for model type."""
        model_type = cls._detect_model_type(model_name)
        return cls._special_configs.get(model_type, {})

    @staticmethod
    def _detect_model_type(model_name: str) -> str:
        """Detect model type from model name."""
        name_lower = model_name.lower()
        for model_type in ModelRegistry._target_modules.keys():
            if model_type in name_lower:
                return model_type
        return "unknown"
```

### Implementing Custom Training Strategies

```python
class TrainingStrategy:
    """Base class for different training strategies."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def prepare_trainer(self, model, dataset, tokenizer):
        """Prepare trainer with strategy-specific configuration."""
        raise NotImplementedError

    def should_stop_early(self, metrics: Dict[str, float]) -> bool:
        """Determine if training should stop early."""
        return False

class AdaptiveLRStrategy(TrainingStrategy):
    """Training strategy with adaptive learning rate."""

    def prepare_trainer(self, model, dataset, tokenizer):
        """Setup trainer with adaptive learning rate."""
        from transformers import get_cosine_schedule_with_warmup

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps
        )

        return SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            optimizers=(optimizer, scheduler),
            args=self.config
        )

class EarlyStoppingStrategy(TrainingStrategy):
    """Training strategy with early stopping."""

    def __init__(self, config, patience=5, min_delta=0.001):
        super().__init__(config)
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.patience_counter = 0

    def should_stop_early(self, metrics: Dict[str, float]) -> bool:
        """Check if training should stop early."""
        current_loss = metrics.get('train_loss', float('inf'))

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.patience
```

### Dataset Pipeline Extensions

```python
class DatasetPipeline:
    """Extensible dataset processing pipeline."""

    def __init__(self):
        self.processors = []

    def add_processor(self, processor):
        """Add a processing step to the pipeline."""
        self.processors.append(processor)
        return self

    def process(self, dataset):
        """Apply all processors to the dataset."""
        for processor in self.processors:
            dataset = processor(dataset)
        return dataset

class TextCleaningProcessor:
    """Clean and normalize text data."""

    def __init__(self, remove_html=True, normalize_whitespace=True):
        self.remove_html = remove_html
        self.normalize_whitespace = normalize_whitespace

    def __call__(self, dataset):
        """Apply text cleaning to dataset."""
        def clean_text(examples):
            cleaned = []
            for text in examples['text']:
                if self.remove_html:
                    text = self._remove_html_tags(text)
                if self.normalize_whitespace:
                    text = self._normalize_whitespace(text)
                cleaned.append(text)
            return {'text': cleaned}

        return dataset.map(clean_text, batched=True)

    def _remove_html_tags(self, text):
        """Remove HTML tags from text."""
        import re
        return re.sub(r'<[^>]+>', '', text)

    def _normalize_whitespace(self, text):
        """Normalize whitespace in text."""
        import re
        return re.sub(r'\s+', ' ', text).strip()

# Usage example
pipeline = DatasetPipeline()
pipeline.add_processor(TextCleaningProcessor())
pipeline.add_processor(lambda ds: ds.filter(lambda x: len(x['text']) > 10))

cleaned_dataset = pipeline.process(raw_dataset)
```

## Contribution Guidelines

### Code Quality Standards

1. **Type Hints**: All public functions must include type hints
2. **Docstrings**: Google-style docstrings for all public methods
3. **Error Handling**: Comprehensive error handling with descriptive messages
4. **Testing**: 80%+ test coverage for new features
5. **Documentation**: Update relevant documentation for API changes

### Development Workflow

```bash
# Feature development workflow
git checkout -b feature/new-feature
# ... implement feature ...
python -m pytest tests/ --cov=smolora
black smolora/ tests/
flake8 smolora/ tests/
mypy smolora/
pre-commit run --all-files
git commit -m "feat: implement new feature"
git push origin feature/new-feature
# ... create pull request ...
```

<!-- ### Release Process

```bash
# Version bump and release
pip install bump2version
bump2version patch  # or minor, major
git push origin main --tags
python setup.py sdist bdist_wheel
twine upload dist/*
``` -->

### Documentation Standards

All code contributions must include:
- Comprehensive docstrings with examples
- Type hints for all parameters and return values
- Error handling documentation
- Performance considerations
- Integration test examples

This developer guide provides the foundation for contributing to and extending SmoLoRA while maintaining code quality and architectural consistency.
