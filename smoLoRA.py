import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from datetime import datetime

class SmoLoRA:
    def __init__(self,
                 base_model_name: str,
                 dataset_name: str,
                 text_field: str = "text",
                 output_dir: str = "./fine_tuned_model"):
        self.base_model_name = base_model_name
        self.dataset_name = dataset_name
        self.text_field = text_field
        self.output_dir = output_dir

        # Set device to MPS if available
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Load the base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        ).to(self.device)
        self.model.config.use_cache = False

        # Set up LoRA configuration
        self.peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        # Load and preprocess dataset
        self.dataset = load_dataset(self.dataset_name, split="train")
        self.dataset = self.dataset.map(lambda ex: { "text": ex[self.text_field] })

        # Prepare SFT configuration
        self.sft_config = SFTConfig(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            max_steps=500,
            logging_steps=10,
            optim="adamw_torch",
            fp16=False,
            bf16=False,
            max_length=1024,  # Updated from max_seq_length to fix deprecation warning
            dataset_text_field="text"
        )

        # Initialize the trainer
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            peft_config=self.peft_config,
            args=self.sft_config
        )

    def train(self):
        print(f"[{datetime.now()}] Starting training...")
        self.trainer.train()
        adapter_ckpt = os.path.join(self.output_dir, "adapter_checkpoint")
        self.trainer.model.save_pretrained(adapter_ckpt)
        self.adapter_checkpoint = adapter_ckpt
        print(f"[{datetime.now()}] Training finished.")

    def save(self):
        print(f"[{datetime.now()}] Starting model merge...")
        del self.model
        del self.trainer
        torch.mps.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        ).to(self.device)
        base_model.config.use_cache = False

        model_with_adapter = PeftModel.from_pretrained(base_model, self.adapter_checkpoint)
        merged_model = model_with_adapter.merge_and_unload()

        merged_model_path = os.path.join(self.output_dir, "final_merged")
        merged_model.save_pretrained(merged_model_path)
        self.tokenizer.save_pretrained(merged_model_path)
        self.merged_model_path = merged_model_path
        print(f"[{datetime.now()}] Model merge finished.")

    def load_model(self, model_path: str):
        print(f"[{datetime.now()}] Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"[{datetime.now()}] Model loaded.")
        return self.model, self.tokenizer

    def inference(self, prompt: str, max_new_tokens: int = 200, do_sample: bool = True, temperature: float = 1.0) -> str:
        print(f"[{datetime.now()}] Starting inference...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[{datetime.now()}] Inference finished.")
        return generated_text