# SmoLoRA Customization Guide

**Level 2: Customizing and Extending SmoLoRA**

This guide is for users who want to modify SmoLoRA's behavior, customize training parameters, implement custom data processing, and extend functionality while using the existing framework.

## Advanced Configuration

### LoRA Parameters

Fine-tune the LoRA adaptation settings for your specific use case:

```python
from smolora import SmoLoRA

# Custom LoRA configuration
trainer = SmoLoRA("microsoft/Phi-1.5", "your_dataset")

# Access and modify LoRA config before training
trainer.peft_config.r = 16                    # Rank (higher = more capacity)
trainer.peft_config.lora_alpha = 32           # Scaling factor (usually 2×r)
trainer.peft_config.lora_dropout = 0.1        # Prevent overfitting
trainer.peft_config.target_modules = ["q_proj", "v_proj", "k_proj"]  # More modules

trainer.train()
```

#### Parameter Impact Guide

**Rank (r)**:
- `r=4`: Minimal parameters, fast training, basic adaptation
- `r=8`: Default, good balance for most tasks
- `r=16-32`: More capacity for complex tasks
- `r=64+`: Maximum adaptation, slower training

**Alpha (α)**:
- Low α (8-16): Conservative adaptation
- Medium α (16-32): Standard setting
- High α (64+): Aggressive adaptation

**Target Modules**:
- `["q_proj", "v_proj"]`: Standard attention adaptation
- `["q_proj", "v_proj", "k_proj", "o_proj"]`: Full attention adaptation
- `["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]`: Include feed-forward layers

### Training Configuration

Customize the training process:

```python
# Modify training parameters before calling train()
trainer.sft_config.num_train_epochs = 3
trainer.sft_config.per_device_train_batch_size = 2
trainer.sft_config.gradient_accumulation_steps = 4
trainer.sft_config.learning_rate = 5e-4
trainer.sft_config.max_steps = 1000
trainer.sft_config.warmup_steps = 100
trainer.sft_config.save_steps = 250

trainer.train()
```

#### Training Strategy Guide

**For Small Datasets (<1000 examples)**:
```python
trainer.sft_config.num_train_epochs = 5
trainer.sft_config.learning_rate = 2e-4
trainer.sft_config.warmup_steps = 50
```

**For Large Datasets (>10000 examples)**:
```python
trainer.sft_config.num_train_epochs = 1
trainer.sft_config.max_steps = 2000
trainer.sft_config.learning_rate = 1e-4
trainer.sft_config.warmup_steps = 200
```

**For Memory-Constrained Environments**:
```python
trainer.sft_config.per_device_train_batch_size = 1
trainer.sft_config.gradient_accumulation_steps = 8
trainer.sft_config.dataloader_num_workers = 0
```

## Custom Data Processing

### Advanced Dataset Handling

```python
from smolora.dataset import prepare_dataset, load_text_data

# Custom preprocessing for complex data
def custom_preprocess(examples):
    """Custom preprocessing function."""
    processed = []
    for text in examples['text']:
        # Custom cleaning and formatting
        cleaned = text.strip().lower()
        if len(cleaned) > 10:  # Filter short texts
            processed.append(cleaned)
    return {'text': processed}

# Load and preprocess data
dataset = prepare_dataset("your_data.jsonl", text_field="content")
dataset = dataset.map(custom_preprocess, batched=True)

# Use with SmoLoRA
trainer = SmoLoRA("microsoft/Phi-1.5", dataset)
```

### Text Chunking for Long Documents

```python
def chunk_long_texts(dataset, max_length=512):
    """Split long texts into manageable chunks."""
    def chunk_function(examples):
        chunked_texts = []
        for text in examples['text']:
            words = text.split()
            for i in range(0, len(words), max_length):
                chunk = ' '.join(words[i:i + max_length])
                if len(chunk.strip()) > 50:  # Minimum chunk size
                    chunked_texts.append(chunk)
        return {'text': chunked_texts}

    return dataset.map(chunk_function, batched=True, remove_columns=dataset.column_names)

# Usage
dataset = prepare_dataset("long_documents.txt")
chunked_dataset = chunk_long_texts(dataset)
trainer = SmoLoRA("microsoft/Phi-1.5", chunked_dataset)
```

### Multi-Field Data Processing

```python
def combine_fields(examples):
    """Combine multiple fields into training text."""
    combined = []
    for title, content, label in zip(examples['title'], examples['content'], examples['label']):
        # Create structured training examples
        formatted = f"Title: {title}\nContent: {content}\nCategory: {label}"
        combined.append(formatted)
    return {'text': combined}

# Load CSV with multiple columns
dataset = prepare_dataset("multi_column_data.csv")
dataset = dataset.map(combine_fields, batched=True)
trainer = SmoLoRA("microsoft/Phi-1.5", dataset)
```

## Model Architecture Customization

### Supporting New Model Types

```python
# Extend target modules for different architectures
def get_target_modules(model_name):
    """Get appropriate target modules for different model types."""
    if "llama" in model_name.lower():
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "phi" in model_name.lower():
        return ["q_proj", "v_proj", "dense"]
    elif "gpt" in model_name.lower():
        return ["c_attn", "c_proj"]
    else:
        return ["q_proj", "v_proj"]  # Safe default

# Apply custom target modules
trainer = SmoLoRA("custom/model", "dataset")
trainer.peft_config.target_modules = get_target_modules("custom/model")
```

### Device and Memory Management

```python
import torch

class CustomSmoLoRA(SmoLoRA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Custom device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Move model to device
        self.model = self.model.to(self.device)

    def _optimize_memory(self):
        """Custom memory optimization."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

        # Additional memory optimizations
        self.model.gradient_checkpointing_enable()
```

## Advanced Inference Patterns

### Batch Inference

```python
def batch_inference(trainer, prompts, **kwargs):
    """Process multiple prompts efficiently."""
    results = []
    batch_size = kwargs.pop('batch_size', 4)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_results = []

        for prompt in batch:
            result = trainer.inference(prompt, **kwargs)
            batch_results.append(result)

        results.extend(batch_results)

    return results

# Usage
prompts = ["Complete this story:", "Summarize this text:", "Answer this question:"]
responses = batch_inference(trainer, prompts, max_new_tokens=100)
```

### Custom Generation Strategies

```python
def guided_generation(trainer, prompt, keywords=None, temperature=0.7):
    """Generate text with keyword guidance."""
    if keywords:
        enhanced_prompt = f"{prompt}\nKey topics to include: {', '.join(keywords)}\n\nResponse:"
    else:
        enhanced_prompt = prompt

    return trainer.inference(
        enhanced_prompt,
        temperature=temperature,
        do_sample=True,
        max_new_tokens=200
    )

# Usage
response = guided_generation(
    trainer,
    "Write a product review",
    keywords=["quality", "price", "delivery"]
)
```

## Monitoring and Evaluation

### Training Progress Tracking

```python
import json
import os

def monitor_training(trainer):
    """Monitor training progress with custom metrics."""

    # Custom training callback
    class ProgressCallback:
        def __init__(self, output_dir):
            self.output_dir = output_dir
            self.metrics = []

        def on_log(self, args, state, control, model, logs=None, **kwargs):
            if logs:
                self.metrics.append(logs)
                # Save metrics
                with open(os.path.join(self.output_dir, "training_metrics.json"), "w") as f:
                    json.dump(self.metrics, f, indent=2)

    # Add callback to trainer
    callback = ProgressCallback(trainer.output_dir)
    trainer.trainer.add_callback(callback)

# Usage
monitor_training(trainer)
trainer.train()
```

### Model Performance Evaluation

```python
def evaluate_model(trainer, test_prompts, expected_patterns=None):
    """Evaluate model performance on test cases."""
    results = []

    for i, prompt in enumerate(test_prompts):
        response = trainer.inference(prompt, temperature=0.1)  # Low temp for consistency

        result = {
            'prompt': prompt,
            'response': response,
            'length': len(response.split()),
            'contains_expected': False
        }

        # Check for expected patterns
        if expected_patterns and i < len(expected_patterns):
            pattern = expected_patterns[i]
            result['contains_expected'] = pattern.lower() in response.lower()

        results.append(result)

    return results

# Usage
test_prompts = ["Complete this sentence:", "Generate a summary:"]
expected = ["completion", "summary"]
evaluation = evaluate_model(trainer, test_prompts, expected)
```

## Integration Patterns

### Using with Existing Pipelines

```python
class SmoLoRAWrapper:
    """Wrapper for integration with existing ML pipelines."""

    def __init__(self, model_name, dataset_name, **config):
        self.trainer = SmoLoRA(model_name, dataset_name)

        # Apply configuration
        for key, value in config.items():
            if hasattr(self.trainer.peft_config, key):
                setattr(self.trainer.peft_config, key, value)
            elif hasattr(self.trainer.sft_config, key):
                setattr(self.trainer.sft_config, key, value)

    def fit(self, X=None, y=None):
        """Scikit-learn style fit method."""
        self.trainer.train()
        self.trainer.save()
        return self

    def predict(self, prompts):
        """Scikit-learn style predict method."""
        if isinstance(prompts, str):
            return self.trainer.inference(prompts)
        return [self.trainer.inference(prompt) for prompt in prompts]

# Usage in ML pipeline
model = SmoLoRAWrapper("microsoft/Phi-1.5", "dataset", r=16, learning_rate=1e-4)
model.fit()
predictions = model.predict(["test prompt 1", "test prompt 2"])
```

## Best Practices for Customization

### 1. Systematic Experimentation
- Change one parameter at a time
- Keep detailed logs of configurations
- Use version control for your custom code
- Create baseline comparisons

### 2. Data Quality Focus
- Implement robust preprocessing pipelines
- Validate data at each step
- Handle edge cases explicitly
- Monitor data distribution changes

### 3. Resource Management
- Profile memory usage during training
- Implement automatic checkpointing
- Use gradient accumulation for large effective batch sizes
- Monitor training stability

### 4. Model Validation
- Create comprehensive test suites
- Implement automated evaluation metrics
- Compare against baseline models
- Test edge cases and failure modes

This customization guide provides the tools to adapt SmoLoRA to your specific needs while maintaining the simplicity and reliability of the core framework.
