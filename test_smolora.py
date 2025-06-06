import os
import tempfile
import shutil
import pytest
from unittest.mock import patch, MagicMock

# Test local_text.py
def test_load_text_data(tmp_path):
    from local_text import load_text_data
    # Create sample text files
    file1 = tmp_path / "a.txt"
    file2 = tmp_path / "b.txt"
    file1.write_text("hello world\nthis is a test\n\n")
    file2.write_text("another line\n")
    dataset = load_text_data(str(tmp_path))
    texts = [x['text'] for x in dataset]
    assert "hello world" in texts
    assert "this is a test" in texts
    assert "another line" in texts
    assert len(texts) == 3

# Test prepare_dataset.py for txt, csv, jsonl
def test_prepare_dataset_txt(tmp_path):
    from prepare_dataset import prepare_dataset
    d = tmp_path / "txts"
    d.mkdir()
    (d/"1.txt").write_text("foo bar\nhello world\n")
    (d/"2.txt").write_text("baz\n")
    dataset = prepare_dataset(str(d))
    texts = [x['text'] for x in dataset]
    assert "foo bar" in texts
    assert "hello world" in texts
    assert "baz" in texts
    assert len(texts) == 3

def test_prepare_dataset_jsonl(tmp_path):
    from prepare_dataset import prepare_dataset
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text('{"text": "abc"}\n{"text": "def"}\n')
    dataset = prepare_dataset(str(jsonl))
    texts = [x['text'] for x in dataset]
    assert "abc" in texts
    assert "def" in texts
    assert len(texts) == 2


def test_prepare_dataset_csv(tmp_path):
    from prepare_dataset import prepare_dataset
    csvf = tmp_path / "data.csv"
    csvf.write_text("text\nrow1\nrow2\n")
    dataset = prepare_dataset(str(csvf))
    texts = [x['text'] for x in dataset]
    assert "row1" in texts
    assert "row2" in texts
    assert len(texts) == 2

# Test chunking
def test_prepare_dataset_chunking(tmp_path):
    from prepare_dataset import prepare_dataset
    d = tmp_path / "txts"
    d.mkdir()
    (d/"1.txt").write_text("one two three four five six\n")
    dataset = prepare_dataset(str(d), chunk_size=2)
    texts = [x['text'] for x in dataset]
    assert "one two" in texts
    assert "three four" in texts
    assert "five six" in texts
    assert len(texts) == 3

# Test SmoLoRA with mocks
def test_lora_trainer_init_and_inference():
    with patch("smoLoRA.AutoModelForCausalLM") as mock_model_cls, \
         patch("smoLoRA.AutoTokenizer") as mock_tokenizer_cls, \
         patch("smoLoRA.load_dataset") as mock_load_dataset, \
         patch("smoLoRA.SFTTrainer") as mock_trainer_cls:
        from smoLoRA import SmoLoRA
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_trainer = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_load_dataset.return_value = [{"text": "sample"}]
        mock_trainer_cls.return_value = mock_trainer
        trainer = SmoLoRA(
            base_model_name="fake/model",
            dataset_name="fake_dataset",
            text_field="text",
            output_dir="/tmp/fake_out"
        )
        assert trainer.model is mock_model
        assert trainer.tokenizer is mock_tokenizer
        assert trainer.dataset[0]["text"] == "sample"
        # Test inference
        mock_tokenizer.__call__.return_value = {"input_ids": [[1,2,3]]}
        mock_tokenizer.decode.return_value = "output text"
        mock_model.generate.return_value = [[1,2,3,4]]
        trainer.model = mock_model
        trainer.tokenizer = mock_tokenizer
        result = trainer.inference("prompt")
        assert result == "output text"

# Test SmoLoRA.train and save with mocks
def test_lora_trainer_train_and_save(tmp_path):
    with patch("smoLoRA.AutoModelForCausalLM") as mock_model_cls, \
         patch("smoLoRA.AutoTokenizer") as mock_tokenizer_cls, \
         patch("smoLoRA.load_dataset") as mock_load_dataset, \
         patch("smoLoRA.SFTTrainer") as mock_trainer_cls, \
         patch("smoLoRA.PeftModel") as mock_peft_model_cls:
        from smoLoRA import SmoLoRA
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_trainer = MagicMock()
        mock_peft_model = MagicMock()
        mock_merged_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_load_dataset.return_value = [{"text": "sample"}]
        mock_trainer_cls.return_value = mock_trainer
        mock_peft_model_cls.from_pretrained.return_value = mock_peft_model
        mock_peft_model.merge_and_unload.return_value = mock_merged_model
        # Save methods
        mock_trainer.model.save_pretrained = MagicMock()
        mock_merged_model.save_pretrained = MagicMock()
        mock_tokenizer.save_pretrained = MagicMock()
        trainer = SmoLoRA(
            base_model_name="fake/model",
            dataset_name="fake_dataset",
            text_field="text",
            output_dir=str(tmp_path)
        )
        trainer.trainer = mock_trainer
        trainer.model = mock_model
        trainer.tokenizer = mock_tokenizer
        # Test train
        trainer.train()
        mock_trainer.train.assert_called()
        mock_trainer.model.save_pretrained.assert_called()
        # Test save
        trainer.adapter_checkpoint = str(tmp_path / "adapter_checkpoint")
        trainer.save()
        mock_peft_model_cls.from_pretrained.assert_called()
        mock_peft_model.merge_and_unload.assert_called()
        mock_merged_model.save_pretrained.assert_called()
        mock_tokenizer.save_pretrained.assert_called()

# Test SmoLoRA.load_model with mocks
def test_lora_trainer_load_model():
    with patch("smoLoRA.AutoModelForCausalLM") as mock_model_cls, \
         patch("smoLoRA.AutoTokenizer") as mock_tokenizer_cls:
        from smoLoRA import SmoLoRA
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        trainer = SmoLoRA(
            base_model_name="fake/model",
            dataset_name="fake_dataset",
            text_field="text",
            output_dir="/tmp/fake_out"
        )
        model, tokenizer = trainer.load_model("/tmp/fake_model")
        assert model is mock_model
        assert tokenizer is mock_tokenizer
