"""Contains sample usage for SmoLoRA."""

from datetime import datetime

# from local_text import load_text_data
from smoLoRA import SmoLoRA

# Optionally import the general-purpose preparer
# from prepare_dataset import prepare_dataset

start = datetime.now()
print("Welcome to SmoLoRA!")
print("Initializing the trainer...")

# base model to train
base_model = "microsoft/Phi-1.5"
# # or try a different base model
# base_model = "meta-llama/Llama-2-7b-hf"

# dataset to tune the model
dataset = "yelp_review_full"
# # or try a custom dataset
# dataset = load_text_data("./my_text_data")
# # or use the general-purpose preparer
# dataset = prepare_dataset("./my_texts_folder", chunk_size=128)
# dataset = prepare_dataset("./data.csv", text_field="content")
# dataset = prepare_dataset("./data.jsonl", text_field="message", chunk_size=256)

# test prompt
prompt = "Write a review about a great coffee shop."
# # try choosing your own prompt
# prompt = "Your custom prompt here."

# initialize the trainer
print("Initializing the trainer...")
trainer = SmoLoRA(
    base_model_name=base_model,
    dataset_name=dataset,
    text_field="text",
    output_dir="./output_model",
)

# # for custom in-memory datasets, set trainer.dataset directly
# trainer.dataset = dataset

# # try modifying some training arguments for deeper control
# trainer_tool.training_args.learning_rate = 1e-4
# trainer_tool.training_args.num_train_epochs = 3
# trainer_tool.training_args.per_device_train_batch_size = 2

trainer_init_time = datetime.now()
print(f"Trainer initialized in {trainer_init_time - start}s")

# fine-tune the model
print("Starting model tuning...")
trainer.train()
trainer_tune_time = datetime.now()
print(f"Model tuned in {trainer_tune_time - trainer_init_time}s")

# merge the LoRA adapter and save the final model
print("Merging the model and saving...")
trainer.save()
trainer_save_time = datetime.now()
print(f"Merged and saved in {trainer_save_time - trainer_tune_time}s")

# load the tuned model
print("Loading the tuned model...")
model, tokenizer = trainer.load_model("./output_model/final_merged")
load_model_time = datetime.now()
print(f"Model loaded in {load_model_time - trainer_save_time}s")

# run a single inference on the model
print("Running inference...")
result = trainer.inference(prompt)
# # or or try some different settings
# result = trainer.inference(prompt, max_new_tokens=150, do_sample=True, temperature=0.8)
print("Generated output:", result)
inference_time = datetime.now()
print(f"Inference completed in {inference_time - load_model_time}s")

# # or try running multiple inferences
# print("Running multiple inferences...")
# prompts = [
#     "Write a glowing review about a spa experience.",
#     "Describe a frustrating visit to a car dealership.",
#     "Summarize a night out at a music concert."
# ]

# inference_results = {}
# for p in prompts:
#     output = trainer.inference(p, max_new_tokens=250, do_sample=True, temperature=0.7)
#     print("Prompt:", p)
#     print("Response:", output)
#     print("-" * 50)
#     inference_results[p] = output
# print("Multiple inferences completed.")

# # optionally save multiple outputs
# import json
# with open("inference_results.json", "w") as f:
#     json.dump(inference_results, f, indent=2)

print("Bye now!")
