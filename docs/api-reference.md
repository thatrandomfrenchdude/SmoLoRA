# API Reference

This document provides comprehensive API documentation for all public classes and functions in SmoLoRA.

## SmoLoRA Class

The main class for fine-tuning language models using LoRA (Low-Rank Adaptation) techniques.

### Constructor

```python
class SmoLoRA:
    def __init__(self,
                 base_model_name: str,
                 dataset_name: str,
                 text_field: str = "text",
                 output_dir: str = "./fine_tuned_model"):
```

**Parameters:**
- `base_model_name` (str): HuggingFace model identifier (e.g., "microsoft/Phi-1.5", "meta-llama/Llama-2-7b-hf")
- `dataset_name` (str): HuggingFace dataset identifier or custom dataset name
- `text_field` (str, optional): Field name containing text data in the dataset. Defaults to "text"
- `output_dir` (str, optional): Directory for saving checkpoints and final model. Defaults to "./fine_tuned_model"

**Raises:**
- `ValueError`: If model or dataset cannot be loaded
- `RuntimeError`: If device setup fails

**Example:**
```python
from smoLoRA import SmoLoRA

# Basic usage
trainer = SmoLoRA(
    base_model_name="microsoft/Phi-1.5",
    dataset_name="yelp_review_full"
)

# With custom parameters
trainer = SmoLoRA(
    base_model_name="meta-llama/Llama-2-7b-hf",
    dataset_name="my_custom_dataset",
    text_field="content",
    output_dir="./models/my_fine_tuned_model"
)
```

### Methods

#### train()

Executes the LoRA fine-tuning process.

```python
def train(self) -> None:
```

**Behavior:**
- Runs the SFT (Supervised Fine-Tuning) training loop
- Saves LoRA adapter weights to `{output_dir}/adapter_checkpoint`
- Sets `self.adapter_checkpoint` attribute for later use
- Prints training progress with timestamps

**Example:**
```python
trainer = SmoLoRA("microsoft/Phi-1.5", "yelp_review_full")
trainer.train()  # Starts training process
```

#### save()

Merges LoRA adapter with base model and saves the final merged model.

```python
def save(self) -> None:
```

**Behavior:**
- Loads base model fresh from HuggingFace
- Applies LoRA adapter weights
- Merges adapter into base model weights
- Saves merged model to `{output_dir}/final_merged`
- Clears GPU memory cache
- Sets `self.merged_model_path` attribute

**Requirements:**
- Must call `train()` first to create adapter checkpoint
- Sufficient disk space for merged model

**Example:**
```python
trainer.train()
trainer.save()  # Creates final merged model
```

#### load_model()

Loads a previously saved model for inference.

```python
def load_model(self, model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
```

**Parameters:**
- `model_path` (str): Path to saved model directory

**Returns:**
- Tuple of (model, tokenizer) ready for inference

**Example:**
```python
model, tokenizer = trainer.load_model("./models/my_fine_tuned_model/final_merged")
```

#### inference()

Generates text using the fine-tuned model.

```python
def inference(self, 
              prompt: str, 
              max_new_tokens: int = 200, 
              do_sample: bool = True, 
              temperature: float = 1.0) -> str:
```

**Parameters:**
- `prompt` (str): Input text to generate from
- `max_new_tokens` (int, optional): Maximum number of new tokens to generate. Defaults to 200
- `do_sample` (bool, optional): Whether to use sampling. Defaults to True
- `temperature` (float, optional): Sampling temperature for randomness. Defaults to 1.0

**Returns:**
- Generated text string (includes original prompt)

**Example:**
```python
# Basic generation
result = trainer.inference("Write a review about a great coffee shop.")

# With custom parameters
result = trainer.inference(
    "Describe a perfect day:",
    max_new_tokens=150,
    do_sample=True,
    temperature=0.8
)
```

### Attributes

After initialization, SmoLoRA instances have these key attributes:

- `model`: The loaded base model (AutoModelForCausalLM)
- `tokenizer`: The model's tokenizer (AutoTokenizer)
- `dataset`: Preprocessed training dataset
- `trainer`: SFTTrainer instance for fine-tuning
- `peft_config`: LoRA configuration object
- `sft_config`: Training configuration object
- `device`: PyTorch device (MPS on macOS, CPU fallback)
- `adapter_checkpoint`: Path to saved adapter (set after training)
- `merged_model_path`: Path to merged model (set after saving)

### Configuration Objects

#### LoRA Configuration

The class uses these LoRA settings by default:

```python
peft_config = LoraConfig(
    r=8,                          # Rank of adaptation
    lora_alpha=16,               # LoRA scaling parameter
    lora_dropout=0.1,            # Dropout probability
    bias="none",                 # Bias training strategy
    task_type="CAUSAL_LM",       # Task type
    target_modules=["q_proj", "v_proj"]  # Modules to adapt
)
```

#### Training Configuration

Default training parameters:

```python
sft_config = SFTConfig(
    output_dir=self.output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_steps=500,
    logging_steps=10,
    optim="adamw_torch",
    fp16=False,
    bf16=False,
    max_length=1024,
    dataset_text_field="text"
)
```

## Dataset Preparation Functions

### prepare_dataset()

General-purpose dataset preparation utility in `prepare_dataset.py`.

```python
def prepare_dataset(
    source: str,
    text_field: str = "text",
    chunk_size: int = 0,
    file_type: Optional[str] = None
) -> Dataset:
```

**Parameters:**
- `source` (str): Path to data source (folder of .txt files, .jsonl file, or .csv file)
- `text_field` (str, optional): Field name containing text in structured files. Defaults to "text"
- `chunk_size` (int, optional): Split texts into chunks of this many words. 0 = no chunking. Defaults to 0
- `file_type` (Optional[str], optional): Force file type ("txt", "jsonl", "csv"). None = auto-detect. Defaults to None

**Returns:**
- HuggingFace Dataset with standardized "text" field

**Raises:**
- `ValueError`: If file type cannot be inferred or is unsupported

**Examples:**
```python
from prepare_dataset import prepare_dataset

# From folder of text files
dataset = prepare_dataset("./my_texts/")

# From JSONL with custom field
dataset = prepare_dataset("data.jsonl", text_field="content")

# From CSV with chunking
dataset = prepare_dataset("data.csv", chunk_size=128)

# Force file type
dataset = prepare_dataset("./data", file_type="txt")
```

### load_text_data()

Simple text file loader in `local_text.py`.

```python
def load_text_data(data_folder: str) -> Dataset:
```

**Parameters:**
- `data_folder` (str): Path to folder containing .txt files

**Returns:**
- HuggingFace Dataset with "text" field

**Example:**
```python
from local_text import load_text_data

dataset = load_text_data("./my_text_files/")
```

## Utility Functions

### Dataset Processing Functions

Located in `prepare_dataset.py`:

#### read_txt_folder()
```python
def read_txt_folder(folder_path: str) -> List[str]:
```
Reads all .txt files from a folder and returns list of text lines.

#### read_jsonl()
```python
def read_jsonl(file_path: str, text_field: str = "text") -> List[str]:
```
Reads JSONL file and extracts text from specified field.

#### read_csv()
```python
def read_csv(file_path: str, text_field: str = "text") -> List[str]:
```
Reads CSV file and extracts text from specified column.

#### chunk_texts()
```python
def chunk_texts(texts: List[str], chunk_size: int = 0) -> List[str]:
```
Splits texts into smaller chunks of specified word count.

## Usage Patterns

### Basic Training Workflow

```python
from smoLoRA import SmoLoRA

# 1. Initialize
trainer = SmoLoRA(
    base_model_name="microsoft/Phi-1.5",
    dataset_name="yelp_review_full"
)

# 2. Train
trainer.train()

# 3. Save merged model
trainer.save()

# 4. Load for inference
model, tokenizer = trainer.load_model("./fine_tuned_model/final_merged")

# 5. Generate text
result = trainer.inference("Write a review about a great restaurant.")
```

### Custom Dataset Workflow

```python
from smoLoRA import SmoLoRA
from prepare_dataset import prepare_dataset

# 1. Prepare custom dataset
dataset = prepare_dataset("./my_data.jsonl", text_field="message")

# 2. Initialize trainer
trainer = SmoLoRA(
    base_model_name="microsoft/Phi-1.5",
    dataset_name="dummy"  # Will be replaced
)

# 3. Override dataset
trainer.dataset = dataset

# 4. Train and save
trainer.train()
trainer.save()
```

### Multiple Inference Workflow

```python
# Load once, use multiple times
trainer = SmoLoRA("microsoft/Phi-1.5", "yelp_review_full")
trainer.train()
trainer.save()

prompts = [
    "Write a positive review:",
    "Describe a bad experience:",
    "Rate this restaurant:"
]

results = {}
for prompt in prompts:
    result = trainer.inference(
        prompt, 
        max_new_tokens=100, 
        temperature=0.7
    )
    results[prompt] = result
```

## Error Handling

Common exceptions and how to handle them:

### Model Loading Errors
```python
try:
    trainer = SmoLoRA("invalid/model", "dataset")
except ValueError as e:
    print(f"Model loading failed: {e}")
```

### Training Errors
```python
try:
    trainer.train()
except RuntimeError as e:
    print(f"Training failed: {e}")
    # Check GPU memory, dataset format, etc.
```

### Inference Errors
```python
try:
    result = trainer.inference("prompt")
except Exception as e:
    print(f"Inference failed: {e}")
    # Ensure model is loaded first
```

## Best Practices

1. **Memory Management**: Call `save()` after training to free GPU memory
2. **Dataset Size**: Start with small datasets for testing
3. **Model Choice**: Use smaller models (Phi-1.5) for development
4. **Parameter Tuning**: Adjust `max_steps` and `learning_rate` based on your data
5. **Error Handling**: Always wrap model operations in try-except blocks
