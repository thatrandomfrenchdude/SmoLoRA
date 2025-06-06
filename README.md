![SmoLoRA Logo](logo.png)

# Edge Language Model Fine-Tuning & Inference Toolkit

A lightweight, developer-friendly Python tool for fine-tuning a small language model using LoRA adapters and running on-device inference. Built for flexibility and rapid prototyping, this tool allows you to train, save, load, and generate text from a language model—all in just a few Python files.

---

## Table of Contents
[📦 Features](#-features)  
[🔧 Requirement Setup](#-requirement-setup)  
[📁 Configuration](#-configuration)  
[🚀 Class Structure Overview](#-class-structure-overview)  
[🚀 Quickstart Walkthrough](#-quickstart-walkthrough)  
[📂 Custom Dataset with Local Text Files](#-custom-dataset-with-local-text-files)  
[🛠️ General-Purpose Dataset Preparation](#-general-purpose-dataset-preparation)  
[🚀 Usage Example](#-usage-example)  
[🧠 Tips & Best Practices](#-tips--best-practices)  
[📂 Files](#-files)

---

## 📦 Features

- Fine-tune LLaMA models using PEFT LoRA adapters
- Use HuggingFace datasets or load your own custom text files
- Prepare your own text, CSV, or JSONL data for fine-tuning with a general-purpose dataset preparer
- Merge LoRA adapters into the base model for streamlined inference
- Save and reload fine-tuned models locally
- Clean and minimal class-based architecture for rapid prototyping

---

## 🔧 Requirement Setup

Install the required libraries. A virtual environment is strongly recommended.

```bash
# Global Install
pip install transformers datasets peft accelerate trl

---------------------------------------------------------------

# Virtual Environment Install
# Step 1: create the environment
python -m venv local-lora-venv

# Step 2: activate the environment (mac/linux)
source local-lora-venv/bin/activate

# Step 3: install the requirements
pip install transformers datasets peft accelerate trl
```

_Note: The current implementation does not explicitly support quantization via bitsandbytes._

---

## 📁 Configuration

When initializing the `LoRATrainer` class, you need to specify the following parameters:

| Parameter         | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `base_model_name` | Name or identifier of the base model (e.g. `"meta-llama/Llama-2-7b-hf"`)       |
| `dataset_name`    | HuggingFace dataset identifier or a dataset name string used by `load_dataset` |
| `text_field`      | The field in the dataset or in your custom data that contains the text       |
| `output_dir`      | Directory where the adapter checkpoint and merged model will be saved       |

Inside the class, the trainer:
- Loads the base model and tokenizer.
- Processes the chosen dataset (or expects a HuggingFace dataset if passed in a different way).
- Sets up a PEFT-based LoRA adapter configuration.
- Prepares the training configuration and trainer.

---

## 🚀 Class Structure Overview

```python
LoRATrainer:
  ├── __init__()         # Setup tokenizer, dataset, model, and training args.
  ├── train()            # Fine-tune the base model using a LoRA adapter.
  ├── save()             # Merge the adapter into the base model and save locally.
  ├── load_model()       # Reload the merged model for inference.
  └── inference()        # Run text generation on a provided prompt.
```

---

## 🚀 Quickstart Walkthrough

Below is a basic end-to-end example for fine-tuning and running inference:

```python
from LoRATrainer import LoRATrainer

# Initialize the trainer using a HuggingFace dataset.
trainer = LoRATrainer(
    base_model_name="meta-llama/Llama-2-7b-hf",
    dataset_name="yelp_review_full",
    text_field="text",
    output_dir="./output_model"
)

# Fine-tune the model.
trainer.train()

# Merge the adapter with the base model and save.
trainer.save()

# Load the merged model.
model, tokenizer = trainer.load_model("./output_model/final_merged")

# Run inference with a simple prompt.
prompt = "Write a review about a great coffee shop."
result = trainer.inference(prompt)
print("Generated output:", result)
```

---

## 📂 Custom Dataset with Local Text Files

If you want to fine-tune on your own text files, use the helper function from `local_text.py` to load your data:

```python
from datasets import Dataset
from local_text import load_text_data

# Load text data from a local folder (each .txt file can contain one or multiple text entries).
dataset = load_text_data("./my_text_data")

# Pass the dataset into the trainer by replacing the dataset name.
from LoRATrainer import LoRATrainer

trainer = LoRATrainer(
    base_model_name="meta-llama/Llama-2-7b-hf",
    dataset_name="yelp_review_full",  # placeholder: change as needed if supporting custom datasets
    text_field="text",
    output_dir="./custom_text_output"
)

# For custom in-memory datasets, adjust the trainer's dataset member as needed:
trainer.dataset = dataset

# Proceed with training, merging, loading, and inference as shown above.
```

---

## 🛠️ General-Purpose Dataset Preparation

You can use the `prepare_dataset.py` tool to convert your raw text, CSV, or JSONL data into a HuggingFace `Dataset` ready for fine-tuning.

### Prepare a dataset from text, CSV, or JSONL

**Command line usage:**
```bash
python prepare_dataset.py ./my_texts_folder --chunk_size 128 --save_path my_dataset.jsonl
python prepare_dataset.py ./data.csv --text_field content --save_path my_dataset.jsonl
python prepare_dataset.py ./data.jsonl --text_field message --chunk_size 256 --save_path my_dataset.jsonl
```

- `--chunk_size` (optional): Split long texts into chunks of N words.
- `--text_field` (optional): Specify the field name for CSV/JSONL files (default: `"text"`).
- `--save_path` (optional): Save the processed dataset as a `.jsonl` file.

**Python usage:**
```python
from prepare_dataset import prepare_dataset

# From a folder of .txt files
dataset = prepare_dataset("./my_texts_folder", chunk_size=128)

# From a CSV file
dataset = prepare_dataset("./data.csv", text_field="content")

# From a JSONL file
dataset = prepare_dataset("./data.jsonl", text_field="message", chunk_size=256)

# Use the resulting dataset for fine-tuning:
from LoRATrainer import LoRATrainer
trainer = LoRATrainer(
    base_model_name="meta-llama/Llama-2-7b-hf",
    dataset_name="yelp_review_full",  # placeholder
    text_field="text",
    output_dir="./output_model"
)
trainer.dataset = dataset
```

**What does the prepared dataset look like?**

Each entry is a dictionary with a `"text"` field:
```python
{'text': 'This is a single training example.'}
```
The dataset is a HuggingFace `Dataset` object, ready for use with the trainer.

---

## 🚀 Usage Example

The `usage.py` file provides a streamlined example which includes timing logs for each stage:

```python
from LoRATrainer import LoRATrainer
from local_text import load_text_data
from datetime import datetime

start = datetime.now()
print("Welcome to SmoLoRA!")
print("Initializing the trainer...")

# Choose your base model
base_model = "microsoft/Phi-1.5"
# Alternate: base_model = "meta-llama/Llama-2-7b-hf"

# Choose your dataset identifier
dataset = "yelp_review_full"
# Alternatively, use custom local text data:
# dataset = load_text_data("./my_text_data")
# Or use the general-purpose preparer:
# from prepare_dataset import prepare_dataset
# dataset = prepare_dataset("./my_texts_folder", chunk_size=128)

# Define a prompt for inference
prompt = "Write a review about a great coffee shop."

print("Initializing the trainer...")
trainer = LoRATrainer(
    base_model_name=base_model,
    dataset_name=dataset,
    text_field="text",
    output_dir="./output_model"
)

trainer_init_time = datetime.now()
print(f"Trainer initialized in {trainer_init_time - start}")

# Fine-tune the model.
print("Starting model tuning...")
trainer.train()
trainer_tune_time = datetime.now()
print(f"Model tuned in {trainer_tune_time - trainer_init_time}")

# Merge the LoRA adapter with the base model and save.
print("Merging the model and saving...")
trainer.save()
trainer_save_time = datetime.now()
print(f"Merged and saved in {trainer_save_time - trainer_tune_time}")

# Load the tuned (merged) model.
print("Loading the tuned model...")
model, tokenizer = trainer.load_model("./output_model/final_merged")
load_model_time = datetime.now()
print(f"Model loaded in {load_model_time - trainer_save_time}")

# Run a single inference.
print("Running inference...")
result = trainer.inference(prompt)
print("Generated output:", result)
inference_time = datetime.now()
print(f"Inference completed in {inference_time - load_model_time}")

print("Bye now!")
```

---

## 🧠 Tips & Best Practices

- For rapid prototyping, use a publicly available HuggingFace dataset such as `"yelp_review_full"`.
- When working with your custom text files, use the provided `load_text_data` function or the general-purpose `prepare_dataset.py` tool to load and format your data.
- After training, merging the adapter with the base model allows you to deploy a single, portable model.
- Adjust training parameters within the code if you need deeper control over hyperparameters.
- Use the timing logs in `usage.py` to evaluate performance during training and inference.

---

## 📂 Files

```
slm-tuner/
├── LoRATrainer.py          # Core tool implementation handling training & inference
├── local_text.py           # Utility to load custom text files into a dataset
├── prepare_dataset.py      # General-purpose dataset preparation tool (text, CSV, JSONL)
├── usage.py                # Example script with timing logs for a full workflow
└── README.md               # Project overview and usage instructions
```

---

Happy fine-tuning! 🦙💻✨
