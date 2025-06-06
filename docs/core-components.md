# Core Components

This document provides detailed explanations of SmoLoRA's main classes, modules, and their internal workings.

## SmoLoRA Class (`smoLoRA.py`)

The central class that orchestrates the entire fine-tuning workflow.

### Class Structure

```python
class SmoLoRA:
    def __init__(self, base_model_name, dataset_name, text_field="text", output_dir="./fine_tuned_model")
    def train(self)
    def save(self)
    def load_model(self, model_path)
    def inference(self, prompt, max_new_tokens=200, do_sample=True, temperature=1.0)
```

### Initialization Process

The constructor performs several critical setup tasks:

#### 1. Device Configuration
```python
self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

**Purpose**: Automatically selects the best available compute device.

**Logic**:
- Checks for MPS (Metal Performance Shaders) on macOS
- Falls back to CPU if MPS unavailable
- Sets device for all subsequent model operations

**Device Priorities**:
1. MPS (Apple Silicon Macs) - Hardware acceleration
2. CPU - Universal fallback

#### 2. Base Model Loading
```python
self.model = AutoModelForCausalLM.from_pretrained(
    self.base_model_name,
    trust_remote_code=True
).to(self.device)
self.model.config.use_cache = False
```

**Key Operations**:
- Downloads model from HuggingFace Hub (if not cached)
- Loads model weights into memory
- Moves model to selected device
- Disables key-value caching for training efficiency

**Memory Impact**: This is the largest memory allocation (2-14GB depending on model)

#### 3. LoRA Configuration Setup
```python
self.peft_config = LoraConfig(
    r=8,                          # Rank of adaptation matrices
    lora_alpha=16,               # Scaling parameter
    lora_dropout=0.1,            # Dropout for regularization
    bias="none",                 # No bias parameter adaptation
    task_type="CAUSAL_LM",       # Causal language modeling
    target_modules=["q_proj", "v_proj"]  # Which modules to adapt
)
```

**Parameter Explanation**:
- `r` (rank): Controls adapter size. Higher = more capacity, more parameters
- `lora_alpha`: Controls how much adapter influences output. Usually 2×r
- `lora_dropout`: Prevents overfitting in adapter layers
- `target_modules`: Which attention layers to adapt

**Design Rationale**:
- Conservative defaults for stable training
- Targets query and value projections (most impactful)
- Balanced between capacity and efficiency

#### 4. Tokenizer Setup
```python
self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
if self.tokenizer.pad_token_id is None:
    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
self.tokenizer.padding_side = "right"
```

**Critical Operations**:
- Loads tokenizer matching the base model
- Handles missing padding token (common issue)
- Sets padding side for consistent batching

**Padding Token Logic**: Many models lack a padding token, so we use the end-of-sequence token as a fallback.

#### 5. Dataset Loading and Preprocessing
```python
self.dataset = load_dataset(self.dataset_name, split="train")
self.dataset = self.dataset.map(lambda ex: { "text": ex[self.text_field] })
```

**Process**:
1. Load dataset from HuggingFace Hub
2. Extract only training split
3. Map custom text field to standard "text" field
4. Result: Uniform dataset structure regardless of source

#### 6. Training Configuration
```python
self.sft_config = SFTConfig(
    output_dir=self.output_dir,
    per_device_train_batch_size=1,      # Conservative for memory
    gradient_accumulation_steps=8,       # Effective batch size = 8
    learning_rate=2e-4,                 # Standard for LoRA
    max_steps=500,                      # Quick training cycles
    logging_steps=10,                   # Regular progress updates
    optim="adamw_torch",                # Stable optimizer
    fp16=False,                         # Full precision for stability
    bf16=False,                         # No bfloat16
    max_length=1024,                    # Token sequence limit
    dataset_text_field="text"           # Field containing training text
)
```

**Parameter Reasoning**:
- **Small batch size**: Fits on consumer hardware
- **Gradient accumulation**: Simulates larger batches
- **Conservative learning rate**: Prevents catastrophic forgetting
- **Quick iterations**: Fast experimentation cycles

#### 7. Trainer Initialization
```python
self.trainer = SFTTrainer(
    model=self.model,
    train_dataset=self.dataset,
    peft_config=self.peft_config,
    args=self.sft_config
)
```

This creates the supervised fine-tuning trainer that orchestrates the training loop.

### Training Method

```python
def train(self):
    print(f"[{datetime.now()}] Starting training...")
    self.trainer.train()
    adapter_ckpt = os.path.join(self.output_dir, "adapter_checkpoint")
    self.trainer.model.save_pretrained(adapter_ckpt)
    self.adapter_checkpoint = adapter_ckpt
    print(f"[{datetime.now()}] Training finished.")
```

**Process Flow**:
1. **Start Training Loop**: Delegates to SFTTrainer
2. **Save Adapter**: Saves only LoRA weights (not full model)
3. **Set Checkpoint Path**: Stores path for later merge operation

**What Happens During Training**:
- Forward passes through adapter-augmented model
- Backpropagation through LoRA parameters only
- Optimizer updates only adapter weights
- Base model weights remain frozen

### Save Method (Model Merging)

The save method is where the magic happens - merging LoRA adapters into the base model.

```python
def save(self):
    print(f"[{datetime.now()}] Starting model merge...")

    # Memory cleanup
    del self.model
    del self.trainer
    torch.mps.empty_cache()

    # Fresh base model load
    base_model = AutoModelForCausalLM.from_pretrained(
        self.base_model_name,
        trust_remote_code=True
    ).to(self.device)
    base_model.config.use_cache = False

    # Apply and merge adapter
    model_with_adapter = PeftModel.from_pretrained(base_model, self.adapter_checkpoint)
    merged_model = model_with_adapter.merge_and_unload()

    # Save final model
    merged_model_path = os.path.join(self.output_dir, "final_merged")
    merged_model.save_pretrained(merged_model_path)
    self.tokenizer.save_pretrained(merged_model_path)
    self.merged_model_path = merged_model_path

    print(f"[{datetime.now()}] Model merge finished.")
```

**Why This Process?**:
1. **Memory Management**: Clears training artifacts to free memory
2. **Fresh Base**: Ensures clean merge without training state
3. **PEFT Integration**: Uses PEFT's optimized merging
4. **Complete Package**: Saves both model and tokenizer

**Technical Details**:
- `merge_and_unload()` mathematically combines adapter weights with base weights
- Result is a standard model (no adapter dependencies)
- Tokenizer saved alongside for complete inference package

### Load Model Method

```python
def load_model(self, model_path: str):
    print(f"[{datetime.now()}] Loading model from {model_path}...")
    self.model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    ).to(self.device)
    self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"[{datetime.now()}] Model loaded.")
    return self.model, self.tokenizer
```

**Purpose**: Loads a previously merged model for inference.

**Returns**: Both model and tokenizer for external use.

### Inference Method

```python
def inference(self, prompt: str, max_new_tokens: int = 200, do_sample: bool = True, temperature: float = 1.0) -> str:
    print(f"[{datetime.now()}] Starting inference...")
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[{datetime.now()}] Inference finished.")
    return generated_text
```

**Process**:
1. **Tokenization**: Convert text to model input format
2. **Device Transfer**: Move tensors to model device
3. **Generation**: Run model's generate method
4. **Decoding**: Convert output tokens back to text

**Parameters Impact**:
- `max_new_tokens`: Controls output length
- `do_sample`: True = creative, False = deterministic
- `temperature`: Higher = more random, lower = more focused

## Dataset Preparation Module (`prepare_dataset.py`)

A comprehensive toolkit for preparing various data formats for training.

### Core Functions

#### read_txt_folder()
```python
def read_txt_folder(folder_path: str) -> List[str]:
    texts = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
    return texts
```

**Behavior**:
- Recursively processes all .txt files
- Splits on newlines (each line becomes a sample)
- Filters out empty lines
- Returns flat list of text strings

#### read_jsonl()
```python
def read_jsonl(file_path: str, text_field: str = "text") -> List[str]:
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if text_field in obj and obj[text_field]:
                texts.append(obj[text_field].strip())
    return texts
```

**JSONL Format**: Each line is a separate JSON object.
```json
{"text": "First example"}
{"text": "Second example"}
```

**Error Handling**: Silently skips malformed JSON or missing fields.

#### read_csv()
```python
def read_csv(file_path: str, text_field: str = "text") -> List[str]:
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if text_field in row and row[text_field]:
                texts.append(row[text_field].strip())
    return texts
```

**CSV Requirements**:
- First row must contain column headers
- Specified text_field must exist as column
- Handles quoted text with embedded commas

#### chunk_texts()
```python
def chunk_texts(texts: List[str], chunk_size: int = 0) -> List[str]:
    if chunk_size <= 0:
        return texts
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk:
                chunks.append(chunk)
    return chunks
```

**Use Cases**:
- Breaking long documents into training-sized pieces
- Creating more training examples from limited data
- Ensuring consistent sequence lengths

### Main Function: prepare_dataset()

```python
def prepare_dataset(
    source: str,
    text_field: str = "text",
    chunk_size: int = 0,
    file_type: Optional[str] = None
) -> Dataset:
```

**File Type Detection Logic**:
```python
if file_type is None:
    if os.path.isdir(source):
        file_type = "txt"
    elif source.endswith(".jsonl"):
        file_type = "jsonl"
    elif source.endswith(".csv"):
        file_type = "csv"
    else:
        raise ValueError("Cannot infer file type. Please specify file_type.")
```

**Processing Pipeline**:
1. **Detect or verify file type**
2. **Read texts using appropriate function**
3. **Apply chunking if requested**
4. **Remove empty/duplicate texts**
5. **Convert to HuggingFace Dataset format**

**Deduplication**:
```python
texts = list(dict.fromkeys(texts))  # Preserves order while removing duplicates
```

## Local Text Module (`local_text.py`)

Simplified text loading for basic use cases.

```python
def load_text_data(data_folder):
    texts = []
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".txt"):
            with open(os.path.join(data_folder, file_name), "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    clean_line = line.strip()
                    if clean_line:
                        texts.append({"text": clean_line})
    dataset = Dataset.from_list(texts)
    return dataset
```

**Differences from prepare_dataset**:
- Simpler interface (single function)
- Returns Dataset directly (not text list)
- No chunking support
- No deduplication
- Text files only

**When to Use**:
- Quick prototyping
- Simple text file collections
- When you don't need advanced features

## Usage Example Module (`usage.py`)

Demonstrates complete workflow patterns.

### Key Patterns Demonstrated

#### 1. Basic Training Loop
```python
trainer = LoRATrainer(  # Note: Should be SmoLoRA
    base_model_name=base_model,
    dataset_name=dataset,
    text_field="text",
    output_dir="./output_model"
)

trainer.train()
trainer.save()
model, tokenizer = trainer.load_model("./output_model/final_merged")
result = trainer.inference(prompt)
```

#### 2. Custom Dataset Integration
```python
# Using custom dataset
dataset = load_text_data("./my_text_data")
trainer.dataset = dataset  # Override default dataset
```

#### 3. Multiple Inference Patterns
```python
prompts = [
    "Write a glowing review about a spa experience.",
    "Describe a frustrating visit to a car dealership.",
    "Summarize a night out at a music concert."
]

for p in prompts:
    output = trainer.inference(p, max_new_tokens=250, temperature=0.7)
    # Process output...
```

### Performance Tracking
The usage example demonstrates timing different phases:
- Initialization time
- Training time
- Saving time
- Loading time
- Inference time

This helps users understand performance characteristics and optimize their workflows.

## Component Integration

### Data Flow Between Components

1. **prepare_dataset.py** → **SmoLoRA.__init__()**
   - Dataset preparation → Training initialization

2. **SmoLoRA.train()** → **SmoLoRA.save()**
   - LoRA adapter creation → Model merging

3. **SmoLoRA.save()** → **SmoLoRA.load_model()**
   - Merged model creation → Inference preparation

4. **SmoLoRA.load_model()** → **SmoLoRA.inference()**
   - Model loading → Text generation

### Error Propagation

Each component handles errors at its level:
- **File I/O errors**: Handled in dataset preparation
- **Model loading errors**: Handled in SmoLoRA initialization
- **Training errors**: Handled by SFTTrainer
- **Inference errors**: Handled in generation methods

### Memory Management

Components coordinate memory usage:
- **Dataset loading**: Loads all data into memory upfront
- **Model loading**: Single model instance at a time
- **Training**: Uses gradient accumulation for memory efficiency
- **Merging**: Explicitly deletes old objects before loading new ones

This design ensures predictable memory usage patterns and prevents memory leaks during long-running workflows.
