# Testing Strategy

This document outlines the comprehensive testing approach used in the SmoLoRA project, focusing on mock-based testing patterns for machine learning components.

## Overview

SmoLoRA uses a robust testing strategy that heavily leverages mocking to isolate components and ensure reliable, fast tests without requiring actual model downloads or GPU resources.

## Testing Philosophy

### Mock-First Approach
- **Isolation**: Each component is tested in isolation using extensive mocking
- **Speed**: Tests run quickly without downloading large models or datasets
- **Reliability**: Tests are deterministic and don't depend on external resources
- **Coverage**: Focus on achieving 80%+ test coverage across all modules

### Test Categories

1. **Unit Tests**: Individual method and function testing
2. **Integration Tests**: Component interaction testing with mocked dependencies
3. **Workflow Tests**: End-to-end pipeline testing with comprehensive mocking

## Mock Patterns

### Model and Tokenizer Mocking

```python
@patch("smoLoRA.AutoModelForCausalLM")
@patch("smoLoRA.AutoTokenizer")
def test_model_initialization(mock_tokenizer_cls, mock_model_cls):
    # Mock model with essential attributes
    mock_model = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.config = MagicMock()
    mock_model.config.use_cache = False
    mock_model_cls.from_pretrained.return_value = mock_model
    
    # Mock tokenizer with required methods
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = None
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.padding_side = "right"
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
```

### Dataset Mocking Pattern

```python
@patch("smoLoRA.load_dataset")
def test_dataset_loading(mock_load_dataset):
    # Mock dataset with map functionality
    mock_dataset = MagicMock()
    mock_mapped_dataset = MagicMock()
    mock_dataset.map.return_value = mock_mapped_dataset
    mock_load_dataset.return_value = mock_dataset
    
    # Test dataset transformation
    trainer = SmoLoRA(...)
    mock_dataset.map.assert_called_once()
```

### Training Pipeline Mocking

```python
@patch("smoLoRA.SFTTrainer")
def test_training_workflow(mock_trainer_cls):
    # Mock trainer with model attribute
    mock_trainer = MagicMock()
    mock_trainer_model = MagicMock()
    mock_trainer_model.save_pretrained = MagicMock()
    mock_trainer.model = mock_trainer_model
    mock_trainer_cls.return_value = mock_trainer
    
    # Test training execution
    trainer = SmoLoRA(...)
    trainer.train()
    mock_trainer.train.assert_called()
```

## Key Testing Patterns

### Device Management Testing

```python
@patch("smoLoRA.torch.device")
@patch("smoLoRA.torch.backends.mps.is_available", return_value=True)
def test_mps_device_selection(mock_mps_available, mock_device):
    mock_device_obj = MagicMock()
    mock_device_obj.type = "mps"
    mock_device.return_value = mock_device_obj
    
    trainer = SmoLoRA(...)
    assert trainer.device.type == "mps"
```

### File Operations Mocking

```python
@patch("os.path.join")
@patch("os.path.exists", return_value=True)
def test_file_operations(mock_exists, mock_path_join):
    mock_path_join.return_value = "/mocked/path"
    # Test file-dependent functionality
```

### PEFT Model Testing

```python
@patch("smoLoRA.PeftModel")
def test_model_merging(mock_peft_model_cls):
    mock_peft_model = MagicMock()
    mock_merged_model = MagicMock()
    mock_peft_model.merge_and_unload.return_value = mock_merged_model
    mock_peft_model_cls.from_pretrained.return_value = mock_peft_model
    
    trainer = SmoLoRA(...)
    trainer.save()
    mock_peft_model.merge_and_unload.assert_called()
```

## Test Organization

### Test File Structure
```
test_smolora.py
├── Test imports and basic functionality
├── Initialization tests
├── Training workflow tests
├── Inference tests
├── Model saving/loading tests
├── Error handling tests
└── Edge case tests
```

### Test Naming Convention
- `test_[component]_[functionality]()` - Basic functionality tests
- `test_[component]_[scenario]_with_[condition]()` - Specific scenario tests
- `test_[component]_error_handling()` - Error condition tests

## Mock Configuration Best Practices

### Essential Mock Attributes

1. **Model Mocks**:
   - `.to()` method for device management
   - `.config` with `use_cache = False`
   - `.generate()` for inference testing
   - `.save_pretrained()` for persistence

2. **Tokenizer Mocks**:
   - `pad_token_id` and `eos_token_id`
   - `padding_side` attribute
   - Callable interface for encoding
   - `.decode()` method for text generation

3. **Dataset Mocks**:
   - `.map()` method for transformations
   - Indexable interface for data access
   - Iterator functionality if needed

### Memory and Resource Management

```python
@patch("smoLoRA.torch.mps.empty_cache")
def test_memory_cleanup(mock_empty_cache):
    # Test memory management functionality
    trainer = SmoLoRA(...)
    trainer.save()  # Should trigger cache cleanup
    mock_empty_cache.assert_called()
```

## Testing Inference Pipeline

### Generation Parameter Testing

```python
def test_inference_with_custom_parameters():
    # Mock model generation with parameter verification
    mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
    
    result = trainer.inference(
        "Test prompt",
        max_new_tokens=100,
        do_sample=False,
        temperature=0.8
    )
    
    # Verify parameters passed correctly
    mock_model.generate.assert_called_with(
        input_ids=[[1, 2, 3]], 
        attention_mask=[[1, 1, 1]],
        max_new_tokens=100,
        do_sample=False,
        temperature=0.8
    )
```

## Test Coverage Requirements

### Minimum Coverage Targets
- **Overall**: 80% line coverage
- **Core Components**: 90% line coverage
- **Critical Paths**: 95% line coverage

### Coverage Analysis
```bash
# Run tests with coverage
pytest --cov=smoLoRA --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Continuous Integration

### Pre-commit Testing
```bash
# Install pre-commit hooks
pre-commit install

# Run all tests before commit
pytest test_smolora.py -v
```

### Automated Test Execution
- All tests must pass before merge
- Coverage reports generated on each PR
- Performance regression detection

## Mock Maintenance

### Keeping Mocks Updated
1. **API Changes**: Update mocks when upstream APIs change
2. **Attribute Verification**: Ensure mocked attributes match real objects
3. **Behavior Consistency**: Mock behavior should reflect actual implementation

### Mock Validation
```python
def test_mock_consistency():
    """Verify mocks match actual API expectations"""
    # Test that mocked methods exist in real classes
    from transformers import AutoModelForCausalLM
    assert hasattr(AutoModelForCausalLM, 'from_pretrained')
```

## Debugging Test Failures

### Common Issues
1. **Missing Mock Attributes**: Add required attributes to mocks
2. **Call Count Mismatches**: Verify expected vs actual method calls
3. **Return Value Types**: Ensure mock returns match expected types

### Debugging Tools
```python
# Enable detailed mock call logging
import logging
logging.getLogger('unittest.mock').setLevel(logging.DEBUG)

# Print mock call history
print(mock_object.call_args_list)
```

## Future Testing Considerations

### Integration Testing
- Consider adding integration tests with smaller models
- Test with actual dataset samples for validation
- Performance benchmarking with controlled inputs

### Property-Based Testing
- Consider using hypothesis for property-based testing
- Test edge cases with generated inputs
- Validate invariants across different configurations

This testing strategy ensures robust, maintainable tests that provide confidence in the SmoLoRA implementation while remaining fast and reliable.
