# SmoLoRA User Guide

SmoLoRA is a simple toolkit for fine-tuning language models using LoRA (Low-Rank Adaptation). This guide covers everything you need to know to start using it immediately.

***Table of Contents***
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Common Use Cases](#common-use-cases)
- [Working with Different Data](#working-with-different-data)
- [Model Options](#model-options)
- [Configuration Options](#configuration-options)
- [Loading Saved Models](#loading-saved-models)
- [Tips for Success](#tips-for-success)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Installation
```bash
pip install smolora
```

### Basic Usage

```python
from smolora import SmoLoRA

# 1. Initialize with a model and dataset
trainer = SmoLoRA(
    base_model_name="microsoft/Phi-1.5",
    dataset_name="yelp_review_full"
)

# 2. Train the model
trainer.train()

# 3. Save the fine-tuned model
trainer.save()

# 4. Use for inference
response = trainer.inference("The restaurant was")
print(response)
```

That's it! You now have a fine-tuned model.

## Core Concepts

### What SmoLoRA Does
- Takes a pre-trained language model (like GPT or LLaMA)
- Fine-tunes it on your specific dataset
- Creates a customized model for your use case
- Uses LoRA to make this process fast and memory-efficient

### Key Benefits
- **Fast**: Training takes minutes, not hours
- **Memory Efficient**: Works on regular laptops (no GPU required)
- **Simple**: Just 4 lines of code to fine-tune a model
- **Flexible**: Works with any HuggingFace model and dataset

## Common Use Cases

### 1. Text Classification
```python
trainer = SmoLoRA("microsoft/Phi-1.5", "imdb")  # Movie review sentiment
trainer.train()
trainer.save()
```

### 2. Custom Text Generation
```python
trainer = SmoLoRA("microsoft/Phi-1.5", "your_custom_dataset")
trainer.train()
response = trainer.inference("Generate a story about")
```

### 3. Question Answering
```python
trainer = SmoLoRA("microsoft/Phi-1.5", "squad")
trainer.train()
answer = trainer.inference("What is the capital of France?")
```

## Working with Different Data

### Using Your Own Data

SmoLoRA supports multiple data formats:

#### Text Files (.txt)
One example per line:
```
This is my first training example.
This is my second training example.
```

#### CSV Files (.csv)
```csv
text,label
"First example",positive
"Second example",negative
```

#### JSON Lines (.jsonl)
```json
{"text": "First example", "category": "positive"}
{"text": "Second example", "category": "negative"}
```

### Using Custom Data
```python
# Point to your data file or directory
trainer = SmoLoRA(
    base_model_name="microsoft/Phi-1.5",
    dataset_name="./my_data.csv",
    text_field="content"  # if your text is in a different column
)
```

## Model Options

### Popular Base Models
- `microsoft/Phi-1.5` - Fast, good for experiments (1.3B parameters)
- `microsoft/Phi-3-mini-4k-instruct` - Balanced performance (3.8B parameters)
- `meta-llama/Llama-2-7b-hf` - High quality (7B parameters)

### Choosing a Model
- **Small models** (1-3B): Fast training, good for testing
- **Medium models** (7-13B): Better quality, longer training
- **Large models** (30B+): Best quality, requires more resources

## Configuration Options

### Basic Configuration
```python
trainer = SmoLoRA(
    base_model_name="microsoft/Phi-1.5",
    dataset_name="yelp_review_full",
    text_field="text",                    # Column name with your text
    output_dir="./my_fine_tuned_model"    # Where to save results
)
```

### Inference Options
```python
response = trainer.inference(
    "Your prompt here",
    max_new_tokens=100,      # How many words to generate
    temperature=0.7,         # Creativity (0.1=focused, 1.0=creative)
    do_sample=True          # Use sampling for variety
)
```

## Loading Saved Models

```python
# Load a previously trained model
trainer = SmoLoRA("microsoft/Phi-1.5", "dummy_dataset")
model, tokenizer = trainer.load_model("./my_fine_tuned_model/final_merged")

# Or use the inference method after training
trainer.train()
trainer.save()
response = trainer.inference("Your prompt")
```

## Tips for Success

### 1. Start Small
- Use small models first (Phi-1.5)
- Test with small datasets (1000 examples)
- Verify everything works before scaling up

### 2. Data Quality Matters
- Clean your text data (remove HTML, fix encoding)
- Ensure consistent formatting
- More diverse examples = better performance

### 3. Monitor Training
- Training should complete without errors
- Check that loss decreases during training
- Test inference immediately after training

### 4. Experiment with Prompts
- Try different prompt formats
- Include examples in your prompts
- Be specific about what you want

## Troubleshooting

### Common Issues

**Out of Memory**
- Use a smaller model (Phi-1.5 instead of Llama-7B)
- Reduce dataset size
- Close other applications

**Poor Results**
- Check your data quality
- Try more training examples
- Adjust your prompts
- Train for more steps

**Model Won't Load**
- Verify the model name is correct
- Check internet connection for downloads
- Ensure you have enough disk space

### Getting Help
- Check error messages carefully
- Start with working examples
- Use smaller datasets for debugging
- Verify your data format matches examples
