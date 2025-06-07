"""Tests for SmoLoRA functionality including dataset loading, preparation, training, and inference."""

from pathlib import Path
from unittest.mock import MagicMock, patch


def test_load_text_data(tmp_path: Path) -> None:
    """Test loading text data from a folder."""
    from smolora.dataset import load_text_data

    # Create sample text files
    file1 = tmp_path / "a.txt"
    file2 = tmp_path / "b.txt"
    file1.write_text("hello world\nthis is a test\n\n")
    file2.write_text("another line\n")
    dataset = load_text_data(str(tmp_path))
    texts = [x["text"] for x in dataset]
    assert "hello world" in texts
    assert "this is a test" in texts
    assert "another line" in texts
    assert len(texts) == 3


def test_prepare_dataset_txt(tmp_path: Path) -> None:
    """Test preparing dataset from .txt files."""
    from smolora.dataset import prepare_dataset

    d = tmp_path / "txts"
    d.mkdir()
    (d / "1.txt").write_text("foo bar\nhello world\n")
    (d / "2.txt").write_text("baz\n")
    dataset = prepare_dataset(str(d))
    texts = [x["text"] for x in dataset]
    assert "foo bar" in texts
    assert "hello world" in texts
    assert "baz" in texts
    assert len(texts) == 3


def test_prepare_dataset_jsonl(tmp_path: Path) -> None:
    """Test preparing dataset from .jsonl files."""
    from smolora.dataset import prepare_dataset

    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text('{"text": "abc"}\n{"text": "def"}\n')
    dataset = prepare_dataset(str(jsonl))
    texts = [x["text"] for x in dataset]
    assert "abc" in texts
    assert "def" in texts
    assert len(texts) == 2


def test_prepare_dataset_csv(tmp_path: Path) -> None:
    """Test preparing dataset from .csv files."""
    from smolora.dataset import prepare_dataset

    csvf = tmp_path / "data.csv"
    csvf.write_text("text\nrow1\nrow2\n")
    dataset = prepare_dataset(str(csvf))
    texts = [x["text"] for x in dataset]
    assert "row1" in texts
    assert "row2" in texts
    assert len(texts) == 2


def test_prepare_dataset_chunking(tmp_path: Path) -> None:
    """Test preparing dataset with chunking enabled."""
    from smolora.dataset import prepare_dataset

    d = tmp_path / "txts"
    d.mkdir()
    (d / "1.txt").write_text("one two three four five six\n")
    dataset = prepare_dataset(str(d), chunk_size=2)
    texts = [x["text"] for x in dataset]
    assert "one two" in texts
    assert "three four" in texts
    assert "five six" in texts
    assert len(texts) == 3


def test_lora_trainer_init_and_inference() -> None:
    """Test SmoLoRA initialization and inference with mocks."""
    with patch("smolora.core.AutoModelForCausalLM") as mock_model_cls, patch(
        "smolora.core.AutoTokenizer"
    ) as mock_tokenizer_cls, patch(
        "smolora.core.load_dataset"
    ) as mock_load_dataset, patch(
        "smolora.core.SFTTrainer"
    ) as mock_trainer_cls, patch(
        "smolora.core.torch.device"
    ) as mock_device, patch(
        "smolora.core.torch.backends.mps.is_available", return_value=False
    ):
        from smolora import SmoLoRA

        # Set up device mock to return actual device object
        mock_device_obj = MagicMock()
        mock_device_obj.type = "cpu"
        mock_device.return_value = mock_device_obj

        # Set up model mock
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.config = MagicMock()
        mock_model.config.use_cache = False
        mock_model.config._name_or_path = "fake/model"
        mock_model.config.model_type = "llama"
        mock_model.generate.return_value = [[1, 2, 3, 4]]
        mock_model_cls.from_pretrained.return_value = mock_model

        # Set up tokenizer mock with proper tensor operations
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.padding_side = "right"

        # Mock tokenizer call to return tensor-like object with .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.decode.return_value = "output text"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Mock dataset with .map() method
        mock_dataset = MagicMock()
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.__getitem__.return_value = {"text": "sample"}
        mock_load_dataset.return_value = mock_dataset

        # Set up trainer mock - fix the model attribute issue
        mock_trainer = MagicMock()
        mock_trainer_model = MagicMock()
        mock_trainer_model.save_pretrained = MagicMock()
        mock_trainer.model = mock_trainer_model
        mock_trainer_cls.return_value = mock_trainer

        trainer = SmoLoRA(
            base_model_name="fake/model",
            dataset_name="fake_dataset",
            text_field="text",
            output_dir="/tmp/fake_out",
        )
        assert trainer.model is mock_model
        assert trainer.tokenizer is mock_tokenizer
        assert trainer.dataset[0]["text"] == "sample"
        # Test inference
        result = trainer.inference("prompt")
        assert result == "output text"


def test_lora_trainer_train_and_save(tmp_path: Path) -> None:
    """Test SmoLoRA training and saving with mocks."""
    with patch("smolora.core.AutoModelForCausalLM") as mock_model_cls, patch(
        "smolora.core.AutoTokenizer"
    ) as mock_tokenizer_cls, patch(
        "smolora.core.load_dataset"
    ) as mock_load_dataset, patch(
        "smolora.core.SFTTrainer"
    ) as mock_trainer_cls, patch(
        "smolora.core.PeftModel"
    ) as mock_peft_model_cls, patch(
        "smolora.core.torch.device"
    ) as mock_device, patch(
        "smolora.core.torch.backends.mps.is_available", return_value=False
    ):
        # , patch(
        #     "smolora.core.torch.mps.empty_cache"
        # ) as mock_empty_cache:
        from smolora import SmoLoRA

        # Set up device mock to return actual device object
        mock_device_obj = MagicMock()
        mock_device_obj.type = "cpu"
        mock_device.return_value = mock_device_obj

        # Set up model mock
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.config = MagicMock()
        mock_model.config.use_cache = False
        mock_model.config._name_or_path = "fake/model"
        mock_model.config.model_type = "llama"
        mock_model_cls.from_pretrained.return_value = mock_model

        # Set up tokenizer mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.padding_side = "right"
        mock_tokenizer.save_pretrained = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Mock dataset with .map() method
        mock_dataset = MagicMock()
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        # Set up trainer mock - fix the model attribute issue
        mock_trainer = MagicMock()
        mock_trainer_model = MagicMock()
        mock_trainer_model.save_pretrained = MagicMock()
        mock_trainer.model = mock_trainer_model
        mock_trainer_cls.return_value = mock_trainer

        # Set up PEFT mocks
        mock_peft_model = MagicMock()
        mock_merged_model = MagicMock()
        mock_peft_model_cls.from_pretrained.return_value = mock_peft_model
        mock_peft_model.merge_and_unload.return_value = mock_merged_model
        mock_merged_model.save_pretrained = MagicMock()

        trainer = SmoLoRA(
            base_model_name="fake/model",
            dataset_name="fake_dataset",
            text_field="text",
            output_dir=str(tmp_path),
        )
        # Test train
        trainer.train()
        mock_trainer.train.assert_called()
        mock_trainer_model.save_pretrained.assert_called()

        # Test save
        trainer.adapter_checkpoint = str(tmp_path / "adapter_checkpoint")
        trainer.save()
        mock_peft_model_cls.from_pretrained.assert_called()
        mock_peft_model.merge_and_unload.assert_called()
        mock_merged_model.save_pretrained.assert_called()
        mock_tokenizer.save_pretrained.assert_called()


def test_lora_trainer_load_model() -> None:
    """Test SmoLoRA load_model functionality with mocks."""
    with patch("smolora.core.AutoModelForCausalLM") as mock_model_cls, patch(
        "smolora.core.AutoTokenizer"
    ) as mock_tokenizer_cls, patch(
        "smolora.core.load_dataset"
    ) as mock_load_dataset, patch(
        "smolora.core.SFTTrainer"
    ) as mock_trainer_cls, patch(
        "smolora.core.torch.device"
    ) as mock_device, patch(
        "smolora.core.torch.backends.mps.is_available", return_value=False
    ), patch(
        "os.path.exists", return_value=True
    ), patch(
        "os.path.isdir", return_value=True
    ), patch(
        "os.path.isfile", return_value=True
    ):
        from smolora import SmoLoRA

        # Set up device mock to return actual device object
        mock_device_obj = MagicMock()
        mock_device_obj.type = "cpu"
        mock_device.return_value = mock_device_obj

        # Set up model mock
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.config = MagicMock()
        mock_model.config.use_cache = False
        mock_model.config._name_or_path = "fake/model"
        mock_model.config.model_type = "llama"
        mock_model_cls.from_pretrained.return_value = mock_model

        # Set up tokenizer mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.padding_side = "right"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Mock dataset with .map() method
        mock_dataset = MagicMock()
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        # Set up trainer mock - fix the model attribute issue
        mock_trainer = MagicMock()
        mock_trainer_model = MagicMock()
        mock_trainer_model.save_pretrained = MagicMock()
        mock_trainer.model = mock_trainer_model
        mock_trainer_cls.return_value = mock_trainer

        trainer = SmoLoRA(
            base_model_name="fake/model",
            dataset_name="fake_dataset",
            text_field="text",
            output_dir="/tmp/fake_out",
        )
        # Test load_model
        model, tokenizer = trainer.load_model("fake/model")
        assert model is mock_model
        assert tokenizer is mock_tokenizer


def test_smolora_initialization_with_different_parameters() -> None:
    """Test SmoLoRA initialization with different parameter configurations."""
    with patch("smolora.core.AutoModelForCausalLM") as mock_model_cls, patch(
        "smolora.core.AutoTokenizer"
    ) as mock_tokenizer_cls, patch(
        "smolora.core.load_dataset"
    ) as mock_load_dataset, patch(
        "smolora.core.SFTTrainer"
    ) as mock_trainer_cls, patch(
        "smolora.core.torch.device"
    ) as mock_device, patch(
        "smolora.core.torch.backends.mps.is_available", return_value=True
    ):
        from smolora import SmoLoRA

        # Set up device mock for MPS
        mock_device_obj = MagicMock()
        mock_device_obj.type = "mps"
        mock_device.return_value = mock_device_obj

        # Set up model mock
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.config = MagicMock()
        mock_model.config.use_cache = False
        mock_model_cls.from_pretrained.return_value = mock_model

        # Set up tokenizer mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 50256  # Test with existing pad token
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer.padding_side = "right"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        trainer = SmoLoRA(
            base_model_name="test/model",
            dataset_name="test_dataset",
            text_field="content",  # Different text field
            output_dir="/custom/output",
        )

        # Verify custom parameters were used
        assert trainer.base_model_name == "test/model"
        assert trainer.text_field == "content"
        assert trainer.output_dir == "/custom/output"
        # Verify tokenizer pad_token_id wasn't overridden when it already exists
        assert trainer.tokenizer.pad_token_id == 50256


def test_smolora_inference_with_custom_parameters() -> None:
    """Test inference method with custom generation parameters."""
    with patch("smolora.core.AutoModelForCausalLM") as mock_model_cls, patch(
        "smolora.core.AutoTokenizer"
    ) as mock_tokenizer_cls, patch(
        "smolora.core.load_dataset"
    ) as mock_load_dataset, patch(
        "smolora.core.SFTTrainer"
    ) as mock_trainer_cls, patch(
        "smolora.core.torch.device"
    ) as mock_device, patch(
        "smolora.core.torch.backends.mps.is_available", return_value=False
    ):
        from smolora import SmoLoRA

        # Set up mocks
        mock_device_obj = MagicMock()
        mock_device_obj.type = "cpu"
        mock_device.return_value = mock_device_obj

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.config = MagicMock()
        mock_model.config.use_cache = False
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 1
        mock_inputs = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        mock_inputs_obj = MagicMock()
        mock_inputs_obj.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs_obj
        mock_tokenizer.decode.return_value = "Custom generated text with more tokens"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_dataset = MagicMock()
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        trainer = SmoLoRA(
            base_model_name="fake/model",
            dataset_name="fake_dataset",
            text_field="text",
            output_dir="/tmp/fake_out",
        )

        # Test inference with custom parameters
        result = trainer.inference(
            "Test prompt", max_new_tokens=100, do_sample=False, temperature=0.8
        )

        # Verify the model.generate was called with correct parameters
        # The inputs are unpacked with **, so we check for the individual parameters
        mock_model.generate.assert_called_with(
            input_ids=[[1, 2, 3]],
            attention_mask=[[1, 1, 1]],
            max_new_tokens=100,
            do_sample=False,
            temperature=0.8,
        )
        assert result == "Custom generated text with more tokens"


def test_smolora_train_creates_checkpoint_directory() -> None:
    """Test that training creates the expected checkpoint directory."""
    with patch("smolora.core.AutoModelForCausalLM") as mock_model_cls, patch(
        "smolora.core.AutoTokenizer"
    ) as mock_tokenizer_cls, patch(
        "smolora.core.load_dataset"
    ) as mock_load_dataset, patch(
        "smolora.core.SFTTrainer"
    ) as mock_trainer_cls, patch(
        "smolora.core.torch.device"
    ) as mock_device, patch(
        "smolora.core.torch.backends.mps.is_available", return_value=False
    ), patch(
        "os.path.join"
    ) as mock_path_join:
        from smolora import SmoLoRA

        # Set up mocks
        mock_device_obj = MagicMock()
        mock_device_obj.type = "cpu"
        mock_device.return_value = mock_device_obj

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.config = MagicMock()
        mock_model.config.use_cache = False
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_dataset = MagicMock()
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        mock_trainer = MagicMock()
        mock_trainer_model = MagicMock()
        mock_trainer.model = mock_trainer_model
        mock_trainer_cls.return_value = mock_trainer

        # Mock os.path.join to return expected checkpoint path
        mock_path_join.return_value = "/custom/output/adapter_checkpoint"

        trainer = SmoLoRA(
            base_model_name="fake/model",
            dataset_name="fake_dataset",
            text_field="text",
            output_dir="/custom/output",
        )

        # Train the model
        trainer.train()

        # Verify trainer.train() was called
        mock_trainer.train.assert_called_once()

        # Verify checkpoint path was created correctly
        mock_path_join.assert_called_with("/custom/output", "adapter_checkpoint")

        # Verify model was saved to checkpoint
        mock_trainer_model.save_pretrained.assert_called_with(
            "/custom/output/adapter_checkpoint"
        )

        # Verify adapter_checkpoint attribute was set
        assert trainer.adapter_checkpoint == "/custom/output/adapter_checkpoint"


def test_smolora_dataset_mapping() -> None:
    """Test that dataset mapping works correctly with different text fields."""
    with patch("smolora.core.AutoModelForCausalLM") as mock_model_cls, patch(
        "smolora.core.AutoTokenizer"
    ) as mock_tokenizer_cls, patch(
        "smolora.core.load_dataset"
    ) as mock_load_dataset, patch(
        "smolora.core.SFTTrainer"
    ) as mock_trainer_cls, patch(
        "smolora.core.torch.device"
    ) as mock_device, patch(
        "smolora.core.torch.backends.mps.is_available", return_value=False
    ):
        from smolora import SmoLoRA

        # Set up basic mocks
        mock_device_obj = MagicMock()
        mock_device_obj.type = "cpu"
        mock_device.return_value = mock_device_obj

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.config = MagicMock()
        mock_model.config.use_cache = False
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Create a more realistic dataset mock
        mock_dataset = MagicMock()
        mock_mapped_dataset = MagicMock()
        mock_dataset.map.return_value = mock_mapped_dataset
        mock_load_dataset.return_value = mock_dataset

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        trainer = SmoLoRA(
            base_model_name="fake/model",
            dataset_name="fake_dataset",
            text_field="content",  # Using 'content' instead of 'text'
            output_dir="/tmp/fake_out",
        )

        # Verify dataset.map was called with the correct mapping function
        mock_dataset.map.assert_called_once()

        # Get the mapping function that was passed to dataset.map
        mapping_function = mock_dataset.map.call_args[0][0]

        # Test the mapping function
        test_example = {"content": "This is test content"}
        result = mapping_function(test_example)
        expected = {"text": "This is test content"}
        assert result == expected

        # Verify the mapped dataset is used
        assert trainer.dataset is mock_mapped_dataset


def test_smolora_error_handling() -> None:
    """Test error handling scenarios."""
    with patch("smolora.core.AutoModelForCausalLM") as mock_model_cls, patch(
        "smolora.core.AutoTokenizer"
    ) as mock_tokenizer_cls, patch(
        "smolora.core.load_dataset"
    ) as mock_load_dataset, patch(
        "smolora.core.SFTTrainer"
    ) as mock_trainer_cls, patch(
        "smolora.core.torch.device"
    ) as mock_device, patch(
        "smolora.core.torch.backends.mps.is_available", return_value=False
    ):
        from smolora import SmoLoRA

        # Test that the class gracefully handles initialization
        mock_device_obj = MagicMock()
        mock_device_obj.type = "cpu"
        mock_device.return_value = mock_device_obj

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.config = MagicMock()
        mock_model.config.use_cache = False
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_dataset = MagicMock()
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        # Test successful initialization
        trainer = SmoLoRA(
            base_model_name="fake/model",
            dataset_name="fake_dataset",
            text_field="text",
            output_dir="/tmp/fake_out",
        )

        # Verify all components were initialized
        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.dataset is not None
        assert trainer.trainer is not None
        assert trainer.peft_config is not None
        assert trainer.sft_config is not None


def test_smolora_save_workflow() -> None:
    """Test the complete save workflow including merge operations."""
    with patch("smolora.core.AutoModelForCausalLM") as mock_model_cls, patch(
        "smolora.core.AutoTokenizer"
    ) as mock_tokenizer_cls, patch(
        "smolora.core.load_dataset"
    ) as mock_load_dataset, patch(
        "smolora.core.SFTTrainer"
    ) as mock_trainer_cls, patch(
        "smolora.core.PeftModel"
    ) as mock_peft_model_cls, patch(
        "smolora.core.torch.device"
    ) as mock_device, patch(
        "smolora.core.torch.backends.mps.is_available", return_value=False
    ), patch(
        "smolora.core.torch.mps.empty_cache"
    ) as mock_empty_cache, patch(
        "os.path.join"
    ) as mock_path_join:
        from smolora import SmoLoRA

        # Set up mocks
        mock_device_obj = MagicMock()
        mock_device_obj.type = "cpu"
        mock_device.return_value = mock_device_obj

        # Mock the base model for both initialization and save
        mock_base_model = MagicMock()
        mock_base_model.to = MagicMock(return_value=mock_base_model)
        mock_base_model.config = MagicMock()
        mock_base_model.config.use_cache = False
        mock_model_cls.from_pretrained.return_value = mock_base_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.save_pretrained = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_dataset = MagicMock()
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        # Mock PEFT workflow
        mock_peft_model = MagicMock()
        mock_merged_model = MagicMock()
        mock_merged_model.save_pretrained = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged_model
        mock_peft_model_cls.from_pretrained.return_value = mock_peft_model

        # Mock path operations
        mock_path_join.side_effect = lambda *args: "/".join(args)

        trainer = SmoLoRA(
            base_model_name="fake/model",
            dataset_name="fake_dataset",
            text_field="text",
            output_dir="/test/output",
        )

        # Set up adapter checkpoint (normally set by train method)
        trainer.adapter_checkpoint = "/test/output/adapter_checkpoint"

        # Test save operation
        trainer.save()

        # Verify the complete save workflow
        # 1. Cache should be cleared
        mock_empty_cache.assert_called_once()

        # 2. Base model should be reloaded
        assert (
            mock_model_cls.from_pretrained.call_count >= 2
        )  # Once for init, once for save

        # 3. PEFT model should be loaded with adapter
        mock_peft_model_cls.from_pretrained.assert_called_with(
            mock_base_model, "/test/output/adapter_checkpoint"
        )

        # 4. Model should be merged and unloaded
        mock_peft_model.merge_and_unload.assert_called_once()

        # 5. Merged model should be saved
        mock_merged_model.save_pretrained.assert_called_with(
            "/test/output/final_merged"
        )

        # 6. Tokenizer should be saved
        mock_tokenizer.save_pretrained.assert_called_with("/test/output/final_merged")

        # 7. merged_model_path should be set
        assert trainer.merged_model_path == "/test/output/final_merged"
