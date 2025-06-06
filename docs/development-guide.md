# Development Guide

This guide provides advanced developer information for implementing new features and extending the SmoLoRA codebase.

## Development Environment Setup

### Prerequisites
- Python 3.8+
- Virtual environment management (venv, conda, or pipenv)
- Git for version control
- IDE with Python support (VSCode, PyCharm, etc.)

### Environment Configuration

```bash
# Create and activate virtual environment
python -m venv smolora-dev
source smolora-dev/bin/activate  # On Windows: smolora-dev\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 isort mypy pre-commit

# Install pre-commit hooks
pre-commit install
```

### IDE Configuration

#### VSCode Settings
```json
{
    "python.defaultInterpreterPath": "./smolora-dev/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true
}
```

## Core Architecture Understanding

### Class Hierarchy and Design Patterns

```python
class SmoLoRA:
    """
    Main trainer class implementing the Adapter pattern for LoRA fine-tuning.
    
    Design Patterns Used:
    - Adapter: Wraps transformers components
    - Builder: Configuration setup
    - Template Method: Training pipeline
    """
    
    def __init__(self, ...):
        # Initialization follows builder pattern
        self._setup_device()
        self._load_model()
        self._configure_lora()
        self._setup_tokenizer()
        self._prepare_dataset()
        self._initialize_trainer()
```

### Key Design Decisions

1. **Device Management**: Automatic MPS/CPU detection with fallback
2. **Configuration Encapsulation**: LoRA and SFT configs as class attributes
3. **Dataset Transformation**: Lazy loading with field mapping
4. **Memory Management**: Explicit cache clearing for MPS devices

## Implementation Patterns

### Adding New Model Support

```python
def _get_target_modules(self, model_type: str) -> List[str]:
    """Get LoRA target modules based on model architecture."""
    target_modules_map = {
        "llama": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "gpt2": ["c_attn", "c_proj"],
        "bert": ["query", "value", "key"],
        "t5": ["q", "v", "k", "o"]
    }
    return target_modules_map.get(model_type, ["q_proj", "v_proj"])

def _configure_lora(self):
    """Configure LoRA with model-specific parameters."""
    model_type = getattr(self.model.config, 'model_type', 'unknown')
    target_modules = self._get_target_modules(model_type)
    
    self.peft_config = LoraConfig(
        r=self.lora_r,
        lora_alpha=self.lora_alpha,
        target_modules=target_modules,
        # ... other config
    )
```

### Dataset Handler Extension

```python
class DatasetHandler:
    """Extensible dataset handling with custom preprocessors."""
    
    def __init__(self, text_field: str = "text"):
        self.text_field = text_field
        self.preprocessors = []
    
    def add_preprocessor(self, func: Callable):
        """Add custom preprocessing function."""
        self.preprocessors.append(func)
    
    def process_dataset(self, dataset):
        """Apply all preprocessors to dataset."""
        for preprocessor in self.preprocessors:
            dataset = dataset.map(preprocessor)
        return dataset
```

### Configuration Management

```python
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class LoRAConfig:
    """LoRA configuration with validation."""
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    def __post_init__(self):
        if self.r <= 0:
            raise ValueError("LoRA rank must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("Dropout must be between 0 and 1")

@dataclass
class TrainingConfig:
    """Training configuration with defaults."""
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    
    def to_sft_config(self) -> SFTConfig:
        """Convert to SFTConfig object."""
        return SFTConfig(**self.__dict__)
```

## Advanced Features Implementation

### Custom Loss Functions

```python
import torch.nn as nn
from typing import Optional

class WeightedMSELoss(nn.Module):
    """Custom weighted MSE loss for specific use cases."""
    
    def __init__(self, weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.weights = weights
    
    def forward(self, predictions, targets):
        loss = nn.MSELoss(reduction='none')(predictions, targets)
        if self.weights is not None:
            loss = loss * self.weights
        return loss.mean()

# Integration into SmoLoRA
def _setup_custom_loss(self):
    """Configure custom loss function if specified."""
    if hasattr(self, 'custom_loss_fn'):
        self.trainer.compute_loss = self.custom_loss_fn
```

### Memory Optimization Strategies

```python
def _optimize_memory(self):
    """Apply memory optimization techniques."""
    
    # Enable gradient checkpointing
    if hasattr(self.model, 'gradient_checkpointing_enable'):
        self.model.gradient_checkpointing_enable()
    
    # Use mixed precision training
    self.sft_config.fp16 = torch.cuda.is_available()
    self.sft_config.bf16 = torch.cuda.is_bf16_supported()
    
    # Optimize for memory efficiency
    self.sft_config.dataloader_pin_memory = False
    self.sft_config.dataloader_num_workers = 0

def _clear_memory_cache(self):
    """Clear memory cache based on device type."""
    if self.device.type == "mps":
        torch.mps.empty_cache()
    elif self.device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Force garbage collection
    import gc
    gc.collect()
```

### Checkpoint Management

```python
import json
import os
from pathlib import Path
from typing import Dict, Any

class CheckpointManager:
    """Advanced checkpoint management with metadata."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.metadata_file = self.base_dir / "checkpoint_metadata.json"
    
    def save_checkpoint(self, 
                       model, 
                       step: int, 
                       metrics: Dict[str, float],
                       config: Dict[str, Any]):
        """Save checkpoint with comprehensive metadata."""
        checkpoint_dir = self.base_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save_pretrained(checkpoint_dir)
        
        # Save metadata
        metadata = {
            "step": step,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "config": config,
            "path": str(checkpoint_dir)
        }
        
        self._update_metadata(metadata)
    
    def _update_metadata(self, new_metadata: Dict):
        """Update checkpoint metadata file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {"checkpoints": []}
        
        all_metadata["checkpoints"].append(new_metadata)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
```

## Testing Implementation Patterns

### Test Fixture Management

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_transformer_components():
    """Reusable fixture for transformer components."""
    with patch("smoLoRA.AutoModelForCausalLM") as mock_model_cls, \
         patch("smoLoRA.AutoTokenizer") as mock_tokenizer_cls:
        
        # Setup model mock
        mock_model = MagicMock()
        mock_model.config.use_cache = False
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model
        
        # Setup tokenizer mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        yield {
            "model_cls": mock_model_cls,
            "tokenizer_cls": mock_tokenizer_cls,
            "model": mock_model,
            "tokenizer": mock_tokenizer
        }

def test_custom_feature(mock_transformer_components):
    """Test using reusable fixture."""
    mocks = mock_transformer_components
    # Test implementation
```

### Custom Test Decorators

```python
from functools import wraps

def requires_gpu(func):
    """Skip test if GPU not available."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            pytest.skip("GPU required for this test")
        return func(*args, **kwargs)
    return wrapper

def mock_heavy_operations(func):
    """Decorator to mock resource-intensive operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch("smoLoRA.load_dataset") as mock_dataset, \
             patch("smoLoRA.AutoModelForCausalLM.from_pretrained") as mock_model:
            # Setup mocks
            return func(*args, **kwargs)
    return wrapper
```

## Performance Optimization

### Profiling Integration

```python
import cProfile
import pstats
from contextlib import contextmanager

@contextmanager
def profile_code(output_file: str = None):
    """Context manager for code profiling."""
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield profiler
    finally:
        profiler.disable()
        if output_file:
            profiler.dump_stats(output_file)
        else:
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)

# Usage
with profile_code("training_profile.prof"):
    trainer.train()
```

### Batch Processing Optimization

```python
def optimize_batch_processing(self):
    """Optimize batch processing for different hardware."""
    
    # Determine optimal batch size
    if self.device.type == "mps":
        # MPS has memory limitations
        recommended_batch_size = 2
    elif self.device.type == "cuda":
        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        recommended_batch_size = min(8, gpu_memory // (1024**3))  # Rough estimate
    else:
        recommended_batch_size = 1
    
    self.sft_config.per_device_train_batch_size = recommended_batch_size
    
    # Adjust gradient accumulation accordingly
    target_effective_batch_size = 16
    self.sft_config.gradient_accumulation_steps = (
        target_effective_batch_size // recommended_batch_size
    )
```

## Error Handling Strategies

### Comprehensive Exception Handling

```python
class SmoLoRAError(Exception):
    """Base exception for SmoLoRA errors."""
    pass

class ModelLoadError(SmoLoRAError):
    """Raised when model loading fails."""
    pass

class DatasetError(SmoLoRAError):
    """Raised when dataset operations fail."""
    pass

class TrainingError(SmoLoRAError):
    """Raised when training fails."""
    pass

def safe_model_loading(self):
    """Load model with comprehensive error handling."""
    try:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
    except Exception as e:
        raise ModelLoadError(
            f"Failed to load model {self.base_model_name}: {str(e)}"
        ) from e
```

### Validation Framework

```python
from typing import Union, List
import inspect

def validate_inputs(**validators):
    """Decorator for input validation."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each argument
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for {param_name}: {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@validate_inputs(
    r=lambda x: isinstance(x, int) and x > 0,
    alpha=lambda x: isinstance(x, (int, float)) and x > 0,
    dropout=lambda x: isinstance(x, float) and 0 <= x <= 1
)
def configure_lora(self, r: int, alpha: float, dropout: float):
    """Configure LoRA with validation."""
    pass
```

## Documentation Integration

### Docstring Standards

```python
def train(self, 
          num_epochs: int = None,
          save_steps: int = 500,
          eval_steps: int = 500) -> Dict[str, Any]:
    """
    Train the model using LoRA fine-tuning.
    
    This method initiates the training process using the configured SFTTrainer
    with LoRA adapters. Training progress is automatically saved at specified
    intervals.
    
    Args:
        num_epochs: Number of training epochs. If None, uses config default.
        save_steps: Number of steps between checkpoint saves.
        eval_steps: Number of steps between evaluations.
    
    Returns:
        Dict containing training metrics and final model state.
        
    Raises:
        TrainingError: If training fails due to configuration or resource issues.
        
    Example:
        >>> trainer = SmoLoRA(model_name="gpt2", dataset_name="wikitext")
        >>> results = trainer.train(num_epochs=3, save_steps=100)
        >>> print(f"Final loss: {results['train_loss']}")
        
    Note:
        Training automatically handles device placement and memory optimization.
        Checkpoints are saved to the configured output directory.
    """
    pass
```

## Debugging and Development Tools

### Debug Mode Implementation

```python
import logging
from typing import Optional

class DebugConfig:
    """Configuration for debug mode."""
    def __init__(self, 
                 log_level: str = "INFO",
                 profile_training: bool = False,
                 save_intermediate: bool = False):
        self.log_level = log_level
        self.profile_training = profile_training
        self.save_intermediate = save_intermediate

def setup_debugging(self, debug_config: Optional[DebugConfig] = None):
    """Setup debugging configuration."""
    if debug_config is None:
        return
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, debug_config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    self.logger = logging.getLogger(__name__)
    self.debug_config = debug_config
```

This development guide provides the foundation for extending and maintaining the SmoLoRA codebase while following established patterns and best practices.
