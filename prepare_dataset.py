import os
import json
import csv
from datasets import Dataset
from typing import List, Optional, Union

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

def read_jsonl(file_path: str, text_field: str = "text") -> List[str]:
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if text_field in obj and obj[text_field]:
                texts.append(obj[text_field].strip())
    return texts

def read_csv(file_path: str, text_field: str = "text") -> List[str]:
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if text_field in row and row[text_field]:
                texts.append(row[text_field].strip())
    return texts

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

def prepare_dataset(
    source: str,
    text_field: str = "text",
    chunk_size: int = 0,
    file_type: Optional[str] = None
) -> Dataset:
    """
    General-purpose dataset preparer.
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

    data = [{ "text": t } for t in texts]
    return Dataset.from_list(data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare a text dataset for fine-tuning.")
    parser.add_argument("source", help="Path to .txt folder, .jsonl, or .csv file")
    parser.add_argument("--text_field", default="text", help="Text field name for .jsonl/.csv (default: text)")
    parser.add_argument("--chunk_size", type=int, default=0, help="Chunk size in words (default: 0 = no chunking)")
    parser.add_argument("--file_type", choices=["txt", "jsonl", "csv"], default=None, help="Force file type")
    parser.add_argument("--save_path", default=None, help="Optional: path to save as .jsonl")

    args = parser.parse_args()
    dataset = prepare_dataset(
        source=args.source,
        text_field=args.text_field,
        chunk_size=args.chunk_size,
        file_type=args.file_type
    )
    print(f"Loaded {len(dataset)} samples.")

    if args.save_path:
        dataset.to_json(args.save_path)
        print(f"Saved dataset to {args.save_path}")