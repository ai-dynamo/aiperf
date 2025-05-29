#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from aiperf.common.tokenizer import Tokenizer
from aiperf.services.dataset import utils
from aiperf.services.dataset.custom.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)
from aiperf.services.dataset.generator import PromptGenerator

# TODO: Remove/Move to proper place once configurations are implemented
DEFAULT_BATCH_SIZE = 1


# TODO: temporary
@dataclass
class PrefixPromptConfig:
    num: int = 0
    length: int = 0


# TODO: temporary
@dataclass
class ImageConfig:
    batch_size: int = 1


# TODO: temporary
@dataclass
class TurnDelayConfig:
    ratio: float = 1.0


# TODO: temporary
@dataclass
class SyntheticTokensConfig:
    mean: int = 10
    stddev: int = 2


# XXX: Temporary class to hold the configuration for the custom data
@dataclass
class CustomDataConfig:
    filename: Path | None = None  # TODO: Add support for multiple files?
    tokenizer: Tokenizer | None = None
    prefix_prompt: PrefixPromptConfig | None = None

    # CLI related
    batch_size: int = 1
    num_dataset_entries: int = 100
    image: ImageConfig | None = None
    turn_delay: TurnDelayConfig | None = None
    synthetic_tokens: SyntheticTokensConfig | None = None


class FileInputRetriever:
    def __init__(self, config: CustomDataConfig):
        self.config = config
        self.tokenizer = config.tokenizer

    def retrieve_data(self) -> GenericDataset:
        """Retrieves the dataset from a file or directory.

        Returns:
            The dataset object containing file data.
        """

        files_data: dict[str, FileData] = {}
        if self.config.filename.is_dir():
            files_data = self._get_input_datasets_from_dir()
        else:
            file_data = self._get_input_dataset_from_file(self.config.filename)
            files_data = {str(self.config.filename): file_data}

        return GenericDataset(files_data)

    def _get_input_datasets_from_dir(self) -> dict[str, FileData]:
        """Retrieves the dataset from a directory containing multiple JSONL files.

        Args:
            directory: The directory path to process.

        Returns:
            The dataset in the required format with the content
            read from the files.
        """
        self.config.filename = cast(Path, self.config.filename)
        jsonl_files = list(self.config.filename.glob("*.jsonl"))
        if not jsonl_files:
            raise ValueError(
                f"No JSONL files found in directory '{self.config.filename}'."
            )

        files_data: dict[str, FileData] = {}
        for file in jsonl_files:
            file_data = self._get_input_dataset_from_file(file)
            files_data[file.stem] = file_data
        return files_data

    def _get_input_dataset_from_file(self, filename: Path) -> FileData:
        """Retrieves the dataset from a specific JSONL file.

        Args:
            filename: The path of the file to process.

        Returns:
            The dataset in the required format with the content read from the file.
        """
        utils.check_file_exists(filename)
        prompts, images = self._get_content_from_input_file(filename)
        return self._convert_content_to_data_file(prompts, images)

    def _get_content_from_input_file(
        self, filename: Path
    ) -> tuple[list[str], list[str]]:
        """Reads the content from a JSONL file and returns lists of each content type.

        Args:
            filename: The file path from which to read the content.

        Returns:
            A list of prompts and images read from the file.
        """
        prompts = []
        images = []

        use_prefix_prompts = self.config.prefix_prompt.num > 0
        if use_prefix_prompts:
            PromptGenerator.create_prefix_prompts_pool(
                self.tokenizer,
                self.config.prefix_prompt.num,
                self.config.prefix_prompt.length,
            )

        with open(filename, newline=None) as file:
            for line in file:
                if line.strip():
                    data = utils.load_json_str(line)
                    # None if not provided
                    prompt = data.get("text")
                    prompt_alt = data.get("text_input")
                    if prompt and prompt_alt:
                        raise ValueError(
                            "Each data entry must have only one of 'text_input' or 'text' key name."
                        )
                    prompt = prompt if prompt else prompt_alt
                    if use_prefix_prompts:
                        prefix_prompt = PromptGenerator.get_random_prefix_prompt()
                        prompt = f"{prefix_prompt} {prompt}"
                    if prompt is not None:
                        prompts.append(prompt.strip())

                    image = data.get("image")
                    if image is not None:
                        image = self._handle_image_content(image.strip())
                        images.append(image)
        return prompts, images

    def _handle_image_content(self, content: str) -> str:
        """Handles the image content by either encoding it to the base64 format
        if it's a local file or returning the content as is if it's a URL.

        Args:
            content: The content of the image. Either a local file path or a URL.

        Returns:
            The processed image content.
        """
        # Check if the content is a URL with a scheme and netloc
        url = urlparse(content)
        if url.scheme and url.netloc:
            return content
        elif any([url.scheme, url.netloc]):
            raise ValueError(f"Valid URL must have both a scheme and netloc: {content}")

        # Otherwise, it's a local file path
        img = utils.open_image(content)
        img_base64 = utils.encode_image(img, img.format)
        payload = f"data:image/{img.format.lower()};base64,{img_base64}"
        return payload

    def _convert_content_to_data_file(
        self, prompts: list[str], images: list[str]
    ) -> FileData:
        """Converts the content to a DataFile.

        Args:
            prompts: The list of prompts to convert.
            images: The list of images to convert.

        Returns:
            The DataFile containing the converted data.
        """
        data_rows: list[DataRow] = []

        if prompts and images:
            if self.config.batch_size > len(prompts):
                raise ValueError(
                    "Batch size for texts cannot be larger than the number of available texts"
                )
            if self.config.image.batch_size > len(images):
                raise ValueError(
                    "Batch size for images cannot be larger than the number of available images"
                )
            if (
                self.config.image.batch_size > DEFAULT_BATCH_SIZE
                or self.config.batch_size > DEFAULT_BATCH_SIZE
            ):
                for _ in range(self.config.num_dataset_entries):
                    sampled_texts = random.sample(prompts, self.config.batch_size)
                    sampled_images = random.sample(images, self.config.image.batch_size)
                    data_rows.append(
                        DataRow(texts=sampled_texts, images=sampled_images)
                    )
            else:
                for prompt, image in zip(prompts, images, strict=False):
                    data_rows.append(DataRow(texts=[prompt], images=[image]))
        elif prompts:
            if self.config.batch_size > len(prompts):
                raise ValueError(
                    "Batch size for texts cannot be larger than the number of available texts"
                )
            if self.config.batch_size > DEFAULT_BATCH_SIZE:
                for _ in range(self.config.num_dataset_entries):
                    sampled_texts = random.sample(prompts, self.config.batch_size)
                    data_rows.append(DataRow(texts=sampled_texts, images=[]))
            else:
                for prompt in prompts:
                    data_rows.append(DataRow(texts=[prompt], images=[]))
        elif images:
            if self.config.image.batch_size > len(images):
                raise ValueError(
                    "Batch size for images cannot be larger than the number of available images"
                )

            if self.config.image.batch_size > DEFAULT_BATCH_SIZE:
                for _ in range(self.config.num_dataset_entries):
                    sampled_images = random.sample(images, self.config.image.batch_size)
                    data_rows.append(DataRow(texts=[], images=sampled_images))
            else:
                for image in images:
                    data_rows.append(DataRow(texts=[], images=[image]))

        return FileData(data_rows)
