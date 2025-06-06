"""SmoLoRA: Edge Language Model Fine-Tuning & Inference Toolkit."""

__version__ = "0.1.0"
__author__ = "SmoLoRA Contributors"
__description__ = "A lightweight, developer-friendly Python tool for fine-tuning small language models using LoRA adapters."

from .smolora.core import SmoLoRA
from .smolora.dataset import load_text_data, prepare_dataset

__all__ = ["SmoLoRA", "prepare_dataset", "load_text_data"]
