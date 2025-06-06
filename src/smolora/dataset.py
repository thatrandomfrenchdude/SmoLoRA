"""Dataset handling utilities for SmoLoRA."""

import csv
import json
import os
from typing import List, Optional

from datasets import Dataset


def load_text_data(data_folder: str) -> Dataset:
    """Load text data from a folder containing .txt files into a Hugging Face Dataset.

    Args:
        data_folder: Path to folder containing .txt files

    Returns:
        Dataset with text entries
    """
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


def read_txt_folder(folder_path: str) -> List[str]:
    """Read all .txt files in a folder and return a list of lines.

    Args:
        folder_path: Path to the folder containing .txt files.
    Returns:
        List of strings, each string is a line from the .txt files.
    """
    texts = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
    return texts


def read_jsonl(file_path: str, text_field: str = "text") -> List[str]:
    """Read a .jsonl file and extract the specified text field.

    Args:
        file_path: Path to the .jsonl file.
        text_field: Field name for text in .jsonl (default: "text").
    Returns:
        List of strings, each string is the text from the specified field.
    """
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if text_field in obj and obj[text_field]:
                texts.append(obj[text_field].strip())
    return texts


def read_csv(file_path: str, text_field: str = "text") -> List[str]:
    """Read a .csv file and extract the specified text field.

    Args:
        file_path: Path to the .csv file.
        text_field: Field name for text in .csv (default: "text").
    Returns:
        List of strings, each string is the text from the specified field.
    """
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if text_field in row and row[text_field]:
                texts.append(row[text_field].strip())
    return texts


def chunk_texts(texts: List[str], chunk_size: int = 0) -> List[str]:
    """Split texts into chunks of a specified number of words.

    Args:
        texts: List of strings to be chunked.
        chunk_size: Number of words per chunk (default: 0 = no chunking).
    Returns:
        List of strings, each string is a chunk of the original text.
    """
    if chunk_size <= 0:
        return texts
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk:
                chunks.append(chunk)
    return chunks


def prepare_dataset(
    source: str,
    text_field: str = "text",
    chunk_size: int = 0,
    file_type: Optional[str] = None,
) -> Dataset:
    """General-purpose dataset preparer.

    Args:
        source: Path to a folder of .txt files, or a .jsonl/.csv file.
        text_field: Field name for text in .jsonl/.csv (default: "text").
        chunk_size: If >0, splits texts into chunks of this many words.
        file_type: Force file type: "txt", "jsonl", or "csv". If None, inferred.

    Returns:
        HuggingFace Dataset with a single "text" column.
    """
    if file_type is None:
        if os.path.isdir(source):
            file_type = "txt"
        elif source.endswith(".jsonl"):
            file_type = "jsonl"
        elif source.endswith(".csv"):
            file_type = "csv"
        else:
            raise ValueError("Cannot infer file type. Please specify file_type.")

    if file_type == "txt":
        texts = read_txt_folder(source)
    elif file_type == "jsonl":
        texts = read_jsonl(source, text_field)
    elif file_type == "csv":
        texts = read_csv(source, text_field)
    else:
        raise ValueError(f"Unsupported file_type: {file_type}")

    if chunk_size > 0:
        texts = chunk_texts(texts, chunk_size)

    # Remove empty or duplicate texts
    texts = [t for t in texts if t]
    texts = list(dict.fromkeys(texts))

    data = [{"text": t} for t in texts]
    return Dataset.from_list(data)
