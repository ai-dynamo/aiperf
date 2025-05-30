#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from PIL import Image

from aiperf.common.tokenizer import Tokenizer
from aiperf.services.dataset.custom.file_input_retriever import (
    CustomDataConfig,
    FileInputRetriever,
    ImageConfig,
    PrefixPromptConfig,
)

FILE_INPUT_RETRIEVER_PREFIX = (
    "aiperf.services.dataset.custom.file_input_retriever.FileInputRetriever"
)
PROMPT_GENERATOR_PREFIX = "aiperf.services.dataset.generator.PromptGenerator"


class TestFileInputRetriever:
    @pytest.fixture
    def empty_tokenizer(self) -> Tokenizer:
        return Tokenizer()

    @pytest.fixture
    def config(self, empty_tokenizer) -> CustomDataConfig:
        return CustomDataConfig(
            filename=Path("dummy.jsonl"),
            tokenizer=empty_tokenizer,
            prefix_prompt=PrefixPromptConfig(num=0, length=0),
            image=ImageConfig(batch_size=1),
        )

    @staticmethod
    def image_open_side_effect(*args, **kwargs):
        """Use this side effect to set the format of the mock image."""
        img = Image.new("RGB", (10, 10))
        img.format = "PNG"
        return img

    @staticmethod
    def open_side_effect(filepath, *args, **kwargs):
        single_prompt = '{"text": "What is the capital of France?"}\n'
        multiple_prompts = (
            '{"text": "What is the capital of France?"}\n'
            '{"text": "Who wrote 1984?"}\n'
            '{"text": "What is quantum computing?"}\n'
        )
        single_image = '{"image": "image1.png"}\n'
        multiple_images = (
            '{"image": "image1.png"}\n'
            '{"image": "image2.png"}\n'
            '{"image": "image3.png"}\n'
        )
        multi_modal = (
            '{"text": "What is this image?", "image": "image1.png"}\n'
            '{"text": "Who is this person?", "image": "image2.png"}\n'
        )
        multi_modal_url = (
            '{"text": "What is this image?", "image": "https://some/path/to/image1.png"}\n'
            '{"text": "Who is this person?", "image": "s3://some/path/to/image2.png"}\n'
            '{"text": "What the color of the sky?", "image": "file://some/path/to/image3.png"}\n'
        )

        file_contents = {
            "single_prompt.jsonl": single_prompt,
            "multiple_prompts.jsonl": multiple_prompts,
            "single_image.jsonl": single_image,
            "multiple_images.jsonl": multiple_images,
            "multi_modal.jsonl": multi_modal,
            "multi_modal_url.jsonl": multi_modal_url,
        }
        filename = Path(filepath).name
        return mock_open(read_data=file_contents.get(filename))()

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", side_effect=open_side_effect)
    def test_retrieve_data_single_prompt(self, mock_file, mock_exists, config):
        config.filename = Path("single_prompt.jsonl")
        file_retriever = FileInputRetriever(config)

        data = file_retriever.retrieve_data()
        assert len(data.files_data) == 1
        assert "single_prompt.jsonl" in data.files_data

        file_data = data.files_data["single_prompt.jsonl"]
        assert len(file_data.rows) == 1
        assert file_data.rows[0].texts[0] == "What is the capital of France?"

    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        f"{FILE_INPUT_RETRIEVER_PREFIX}._handle_image_content",
        return_value="mock_base64_image",
    )
    @patch("builtins.open", side_effect=open_side_effect)
    def test_retrieve_data_multi_modal(
        self, mock_file, mock_image_content, mock_exists, config
    ):
        config.filename = Path("multi_modal.jsonl")
        file_retriever = FileInputRetriever(config)

        data = file_retriever.retrieve_data()
        assert len(data.files_data) == 1
        assert "multi_modal.jsonl" in data.files_data

        file_data = data.files_data["multi_modal.jsonl"]
        assert len(file_data.rows) == 2
        assert file_data.rows[0].texts[0] == "What is this image?"
        assert file_data.rows[0].images[0] == "mock_base64_image"
        assert file_data.rows[1].texts[0] == "Who is this person?"
        assert file_data.rows[1].images[0] == "mock_base64_image"

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", side_effect=open_side_effect)
    def test_get_input_file_multi_modal_url(self, mock_file, mock_exists, config):
        config.filename = Path("multi_modal_url.jsonl")
        file_retriever = FileInputRetriever(config)

        data = file_retriever.retrieve_data()
        assert len(data.files_data) == 1
        assert "multi_modal_url.jsonl" in data.files_data

        file_data = data.files_data["multi_modal_url.jsonl"]
        assert len(file_data.rows) == 3
        assert file_data.rows[0].texts[0] == "What is this image?"
        assert file_data.rows[0].images[0] == "https://some/path/to/image1.png"
        assert file_data.rows[1].texts[0] == "Who is this person?"
        assert file_data.rows[1].images[0] == "s3://some/path/to/image2.png"
        assert file_data.rows[2].texts[0] == "What the color of the sky?"
        assert file_data.rows[2].images[0] == "file://some/path/to/image3.png"

    @patch("pathlib.Path.exists", return_value=True)
    @patch("PIL.Image.open", return_value=Image.new("RGB", (10, 10)))
    @patch(
        f"{FILE_INPUT_RETRIEVER_PREFIX}._handle_image_content",
        return_value="mock_base64_image",
    )
    @patch("builtins.open", side_effect=open_side_effect)
    def test_get_input_file_single_image(
        self, mock_file, mock_image, mock_image_content, mock_exists, config
    ):
        config.filename = Path("single_image.jsonl")
        file_retriever = FileInputRetriever(config)
        file_data = file_retriever._get_input_dataset_from_file(
            Path("single_image.jsonl")
        )

        assert file_data is not None
        assert len(file_data.rows) == 1
        assert file_data.rows[0].images[0] == "mock_base64_image"

    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        f"{FILE_INPUT_RETRIEVER_PREFIX}._handle_image_content",
        side_effect=["mock_base64_image1", "mock_base64_image2", "mock_base64_image3"],
    )
    @patch("builtins.open", side_effect=open_side_effect)
    def test_get_input_file_multiple_images(
        self, mock_file, mock_image_content, mock_exists, config
    ):
        config.filename = Path("multiple_images.jsonl")
        file_retriever = FileInputRetriever(config)
        file_data = file_retriever._get_input_dataset_from_file(
            Path("multiple_images.jsonl")
        )

        assert file_data is not None
        assert len(file_data.rows) == 3
        expected_images = [
            "mock_base64_image1",
            "mock_base64_image2",
            "mock_base64_image3",
        ]
        for i, image in enumerate(expected_images):
            assert file_data.rows[i].images[0] == image

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", side_effect=open_side_effect)
    def test_get_input_file_single_prompt(self, mock_file, mock_exists, config):
        config.filename = Path("single_prompt.jsonl")
        file_retriever = FileInputRetriever(config)
        file_data = file_retriever._get_input_dataset_from_file(
            Path("single_prompt.jsonl")
        )

        assert file_data is not None
        assert len(file_data.rows) == 1
        assert file_data.rows[0].texts[0] == "What is the capital of France?"

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", side_effect=open_side_effect)
    def test_get_input_file_multiple_prompts(self, mock_file, mock_exists, config):
        config.filename = Path("multiple_prompts.jsonl")
        file_retriever = FileInputRetriever(config)
        file_data = file_retriever._get_input_dataset_from_file(
            Path("multiple_prompts.jsonl")
        )

        assert file_data is not None
        assert len(file_data.rows) == 3
        expected_prompts = [
            "What is the capital of France?",
            "Who wrote 1984?",
            "What is quantum computing?",
        ]
        for i, prompt in enumerate(expected_prompts):
            assert file_data.rows[i].texts[0] == prompt

    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        f"{FILE_INPUT_RETRIEVER_PREFIX}._handle_image_content",
        return_value="mock_base64_image",
    )
    @patch("builtins.open", side_effect=open_side_effect)
    def test_get_input_file_multi_modal(
        self, mock_file, mock_image_content, mock_exists, config
    ):
        config.filename = Path("multi_modal.jsonl")
        file_retriever = FileInputRetriever(config)
        file_data = file_retriever._get_input_dataset_from_file(
            Path("multi_modal.jsonl")
        )

        assert file_data is not None
        assert len(file_data.rows) == 2
        assert file_data.rows[0].texts[0] == "What is this image?"
        assert file_data.rows[0].images[0] == "mock_base64_image"

    def test_get_input_file_without_file_existing(self, config):
        config.filename = Path("nonexistent_file.jsonl")
        file_retriever = FileInputRetriever(config)

        with pytest.raises(FileNotFoundError):
            file_retriever._get_input_dataset_from_file(Path("nonexistent_file.jsonl"))

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_dir", return_value=True)
    @patch("pathlib.Path.glob", return_value=[])
    def test_get_input_datasets_from_dir_no_jsonl_files(
        self, mock_exists, mock_is_dir, mock_glob, config
    ):
        config.filename = Path("empty_dir")
        file_retriever = FileInputRetriever(config)

        with pytest.raises(ValueError, match="No JSONL files found in directory"):
            _ = file_retriever._get_input_datasets_from_dir()

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_dir", return_value=True)
    @patch(
        "pathlib.Path.glob",
        return_value=[
            Path("single_prompt.jsonl"),
            Path("multiple_prompts.jsonl"),
            Path("single_image.jsonl"),
            Path("multi_modal.jsonl"),
        ],
    )
    @patch(
        f"{FILE_INPUT_RETRIEVER_PREFIX}._handle_image_content",
        return_value="mock_base64_image",
    )
    @patch("builtins.open", side_effect=open_side_effect)
    def test_get_input_datasets_from_dir(
        self,
        mock_file,
        mock_image_content,
        mock_glob,
        mock_is_dir,
        mock_exists,
        config,
    ):
        config.filename = Path("test_dir")
        file_retriever = FileInputRetriever(config)
        file_data = file_retriever._get_input_datasets_from_dir()

        assert len(file_data) == 4
        assert len(file_data["single_prompt"].rows) == 1
        assert (
            file_data["single_prompt"].rows[0].texts[0]
            == "What is the capital of France?"
        )

        assert len(file_data["multiple_prompts"].rows) == 3
        expected_prompts = [
            "What is the capital of France?",
            "Who wrote 1984?",
            "What is quantum computing?",
        ]
        for i, prompt in enumerate(expected_prompts):
            assert file_data["multiple_prompts"].rows[i].texts[0] == prompt

        assert len(file_data["single_image"].rows) == 1
        assert file_data["single_image"].rows[0].images[0] == "mock_base64_image"

        assert len(file_data["multi_modal"].rows) == 2
        assert file_data["multi_modal"].rows[0].texts[0] == "What is this image?"
        assert file_data["multi_modal"].rows[0].images[0] == "mock_base64_image"

    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.is_dir", return_value=True)
    @patch("pathlib.Path.glob", return_value=[])
    def test_get_input_datasets_from_empty_dir(
        self, mock_exists, mock_is_dir, mock_glob, config
    ):
        config.filename = Path("empty_dir")
        file_retriever = FileInputRetriever(config)

        with pytest.raises(ValueError, match="No JSONL files found in directory"):
            _ = file_retriever._get_input_datasets_from_dir()

    @patch("builtins.open", side_effect=open_side_effect)
    @patch(f"{PROMPT_GENERATOR_PREFIX}.create_prefix_prompts_pool")
    @patch(
        f"{PROMPT_GENERATOR_PREFIX}.get_random_prefix_prompt",
        return_value="prefix prompt",
    )
    @patch("pathlib.Path.exists", return_value=True)
    def test_get_input_file_multiple_prompts_with_prefix_prompts(
        self,
        mock_exists,
        mock_random_prefix_prompt,
        mock_create_prefix_prompts_pool,
        mock_file,
        config,
    ):
        config.filename = Path("multiple_prompts.jsonl")
        config.prefix_prompt.num = 3
        config.prefix_prompt.length = 15

        file_retriever = FileInputRetriever(config)
        file_data = file_retriever._get_input_dataset_from_file(
            Path("multiple_prompts.jsonl")
        )

        assert file_data is not None
        assert len(file_data.rows) == 3
        mock_create_prefix_prompts_pool.assert_called_once()
        for row in file_data.rows:
            assert row.texts[0].startswith("prefix prompt ")

    @patch("pathlib.Path.exists", return_value=True)
    @patch("PIL.Image.open", side_effect=image_open_side_effect)
    @patch(
        "aiperf.services.dataset.utils.encode_image", return_value="mock_base64_image"
    )
    @patch("builtins.open", side_effect=open_side_effect)
    @pytest.mark.parametrize(
        "image_path, expected_result",
        [
            # URL paths
            ("file://path/to/image1.png", "file://path/to/image1.png"),
            ("https://path/to/image2.png", "https://path/to/image2.png"),
            ("s3://path/to/image3.png", "s3://path/to/image3.png"),
            # Local file paths
            ("/path/to/image1.png", "data:image/png;base64,mock_base64_image"),
            ("/path/to/image2.png", "data:image/png;base64,mock_base64_image"),
        ],
    )
    def test_handle_image_content(
        self,
        mock_file,
        mock_image,
        mock_encode_image,
        mock_exists,
        image_path,
        expected_result,
        config,
    ):
        config.filename = Path("dummy.jsonl")
        file_retriever = FileInputRetriever(config)

        parsed_img = file_retriever._handle_image_content(image_path)
        assert parsed_img == expected_result
