#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from aiperf.services.dataset import utils

# TODO: temporary
from aiperf.services.dataset.custom.file_input_retriever import CustomDataConfig
from aiperf.services.dataset.custom.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)
from aiperf.services.dataset.generator import PromptGenerator

# TODO: Remove/Move to proper place once configurations are implemented
PAYLOAD_METADATA_FIELDS = ["timestamp", "delay", "session_id"]
PAYLOAD_METADATA_INT_FIELDS = ["timestamp", "delay"]


class PayloadInputRetriever:
    """
    A input retriever class that handles payload level input data provided by the user
    through a file.
    """

    def __init__(self, config: CustomDataConfig):
        self.config = config
        self.tokenizer = config.tokenizer

    def retrieve_data(self) -> GenericDataset:
        """Retrieves the dataset from a file.

        Returns:
            The dataset containing file data.
        """

        files_data: dict[str, FileData] = {}
        input_file = self.config.filename
        if input_file is None:
            raise ValueError("Input file cannot be None")
        file_data = self._get_input_dataset_from_file(input_file)
        files_data = {str(input_file): file_data}

        return GenericDataset(files_data)

    def _get_input_dataset_from_file(self, filename: Path) -> FileData:
        """Retrieves the dataset from a specific JSONL file.

        Args:
            filename: The path of the file to process.

        Returns:
            The dataset in the required format with the content
            read from the file.
        """
        self._verify_file(filename)
        data_dict = self._get_content_from_input_file(filename)
        return self._convert_content_to_data_file(data_dict)

    def _get_content_from_input_file(self, filename: Path) -> dict:
        """Reads the content from a JSONL file and returns lists of each content type.

        Args:
            filename: The file path from which to read the content.

        Returns:
            A dictionary containing extracted data:
                - "prompts": list[str] - Extracted prompts
                - "payload_metadata_list": list[dict] - Corresponding payload metadata list
                - "optional_data_list": list[dict] - Optional data list
        """
        prompts = []
        optional_data_list = []
        payload_metadata_list = []
        with open(filename, newline=None) as file:
            for line in file:
                if line.strip():
                    data = utils.load_json_str(line)
                    prompt = self._get_prompt(data)
                    prompts.append(prompt.strip() if prompt else prompt)
                    try:
                        payload_metadata_list.append(self._get_payload_metadata(data))
                    except Exception as e:
                        raise RuntimeError(
                            "Error while processing payload metadata."
                        ) from e
                    optional_data = self._get_optional_data(data)
                    optional_data_list.append(optional_data)
        return {
            "prompts": prompts,
            "payload_metadata_list": payload_metadata_list,
            "optional_data_list": optional_data_list,
        }

    def _get_prompt(self, data: dict) -> str:
        """Extracts or generates a prompt from the input data.

        Args:
            data: The dictionary containing input data.

        Returns:
            The extracted or generated prompt.

        Raises:
            ValueError: If both "text" and "text_input" fields are present.
        """
        input_length = data.get("input_length")
        prompt_tokens_mean = (
            input_length if input_length else self.config.synthetic_tokens.mean
        )
        prompt_tokens_stddev = (
            0 if input_length else self.config.synthetic_tokens.stddev
        )
        hash_ids = data.get("hash_ids")
        prompt = data.get("text")
        prompt_alt = data.get("text_input")
        # Check if only one of the keys is provided
        if prompt and prompt_alt:
            raise ValueError(
                "Each data entry must have only one of 'text_input' or 'text' key name."
            )
        # If none of the keys are provided, generate a synthetic prompt
        if not prompt and not prompt_alt:
            prompt = PromptGenerator.create_synthetic_prompt(
                self.tokenizer,
                prompt_tokens_mean,
                prompt_tokens_stddev,
                hash_ids,
            )
        prompt = prompt if prompt else prompt_alt
        return str(prompt)

    def _get_payload_metadata(self, data: dict) -> dict:
        """Retrieves and payload metadata from input data
        and scales the delay if the delay_ratio is not 1.0
        """

        metadata = {
            key: (int(data[key]) if key in PAYLOAD_METADATA_INT_FIELDS else data[key])
            for key in PAYLOAD_METADATA_FIELDS
            if key in data
        }

        if "delay" in metadata and self.config.turn_delay.ratio != 1.0:
            metadata["delay"] = int(metadata["delay"] * self.config.turn_delay.ratio)

        return metadata

    def _get_optional_data(self, data: dict) -> dict:
        """Extracts optional data from the input data. If "output_length" is present,
        it is explicitly renamed to "max_tokens".

        Args:
            data: The dictionary containing input data.

        Returns:
            A dictionary containing extracted optional data,
            with "output_length" renamed to "max_tokens" if present.
        """
        excluded_keys = {
            "text",
            "text_input",
            "hash_ids",
            "input_length",
            "output_length",
        }
        excluded_keys.update(PAYLOAD_METADATA_FIELDS)
        optional_data = {k: v for k, v in data.items() if k not in excluded_keys}
        max_tokens = data.get("output_length")
        if max_tokens:
            optional_data["max_tokens"] = max_tokens

        return optional_data

    def _convert_content_to_data_file(
        self,
        data_dict: dict,
    ) -> FileData:
        """Converts the content to a DataFile.

        Args:
            data_dict: A dictionary containing extracted lists of prompts,
                       payload metadata, and optional data.

        Returns:
            The DataFile containing the converted data.
        """
        prompt_list = data_dict["prompts"]
        payload_metadata_list = data_dict["payload_metadata_list"]
        optional_data_list = data_dict["optional_data_list"]

        data_rows: list[DataRow] = [
            DataRow(
                texts=[prompt],
                optional_data=optional_data_list[index],
                payload_metadata=payload_metadata_list[index],
            )
            for index, prompt in enumerate(prompt_list)
        ]

        return FileData(data_rows)
