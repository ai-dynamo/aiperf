#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import random
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from aiperf.services.dataset import utils
from aiperf.services.dataset.composer.base import BaseConversationComposer
from aiperf.services.dataset.config import CustomDataConfig
from aiperf.services.dataset.conversation import Conversation
from aiperf.services.dataset.generator import PromptGenerator
from aiperf.services.dataset.generic_dataset import (
    DataRow,
    FileData,
)

# TODO: Remove/Move to proper place once configurations are implemented
DEFAULT_BATCH_SIZE = 1


class TraceGenerator:
    def __init__(self, config: CustomDataConfig):
        pass


class CustomConversationComposer(BaseConversationComposer):
    def __init__(self, config: CustomDataConfig):
        super().__init__(config)

    def create_conversation(self) -> Conversation:
        """Create a conversation from a file or directory.

        Returns:
            The conversation object.
        """
        # 1. decide if the input is trace, static, or hybrid
        # 2. if trace:
        #     - read the trace file
        #     - generate trace data (using data generators)
        # 3. if static:
        #     - read the static file
        # 4. if hybrid:
        #     - read the hybrid file
        #     - generate trace part of the data (using data generators)
        # 5. create the dataset

        # files_data: dict[str, FileData] = {}
        # if self.config.filename.is_dir():
        #    files_data = self._get_input_datasets_from_dir()
        # else:
        #    file_data = self._get_input_dataset_from_file(self.config.filename)
        #    files_data = {str(self.config.filename): file_data}
        # return GenericDataset(files_data)

        # TODO: (future) for K8s, we need to transfer file data from SC (across node)
        conversation = Conversation()
        _ = self._get_input_dataset_from_file()
        return conversation

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

    def _get_input_dataset_from_file(self) -> FileData:
        """Retrieves the dataset from a specific JSONL file.

        Returns:
            The dataset in the required format with the content read from the file.
        """
        utils.check_file_exists(self.config.filename)
        prompts, images = self._get_content_from_input_file()
        return self._convert_content_to_data_file(prompts, images)

    def _get_content_from_input_file(self) -> tuple[list[str], list[str]]:
        """Reads the content from a JSONL file and returns lists of each content type.

        Returns:
            A list of prompts and images read from the file.
        """
        prompts = []
        images = []

        with open(self.config.filename, newline=None) as file:
            for line in file:
                if line.strip():
                    data = utils.load_json_str(line)
                    prompt = data.get("text")
                    if self.add_prefix_prompt:
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
