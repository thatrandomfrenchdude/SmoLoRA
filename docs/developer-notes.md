# SmoLoRA Developer Notes

This guide is for developers who want to contribute to SmoLoRA by implementing new features and/or maintain the codebase. It describes the internal architecture and assumes familiarity with both basic usage and customization.

***Table of Contents***
- [Core Design Principles](#core-design-principles)
- [System Components](#system-components)
- [Key Classes and Responsibilities](#key-classes-and-responsibilities)

## Core Design Principles

SmoLoRA follows these architectural principles:

1. **Simplicity First**: Single class interface with sensible defaults
2. **Modularity**: Separate concerns for data, training, and inference
3. **Memory Efficiency**: Aggressive memory management for resource-constrained environments
4. **Device Agnostic**: Automatic device detection with fallback strategies

## System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        SmoLoRA Core                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Data Layer    │  │  Training Layer │  │ Inference    │ │
│  │                 │  │                 │  │ Layer        │ │
│  │ • Dataset Prep  │  │ • LoRA Config   │  │ • Model Load │ │
│  │ • Text Loading  │  │ • SFT Training  │  │ • Text Gen   │ │
│  │ • Preprocessing │  │ • Checkpointing │  │ • Batching   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Classes and Responsibilities

### SmoLoRA (`smolora/core.py`)
- **Primary Interface**: Single entry point for all functionality
- **State Management**: Handles model, tokenizer, and configuration state
- **Workflow Orchestration**: Coordinates training, saving, and inference phases
- **Device Management**: Automatic device selection and memory optimization

### DatasetHandler (`smolora/dataset.py`)
- **Format Detection**: Automatic file type detection (TXT, CSV, JSONL)
- **Preprocessing**: Text cleaning, chunking, and field mapping
- **Validation**: Data quality checks and error handling
- **HuggingFace Integration**: Seamless dataset loading and transformation
