"""Module to load text data from a folder containing .txt files into a Hugging Face Dataset."""

import os

from datasets import Dataset


def load_text_data(data_folder: str) -> Dataset:
    """Load text data from a folder containing .txt files."""
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
