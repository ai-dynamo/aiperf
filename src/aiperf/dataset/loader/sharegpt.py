# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

from aiperf.common import random_generator as rng
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.utils import load_json_str
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


class ShareGPTLoader(BaseFileLoader):
    """ShareGPT dataset loader for loading and processing ShareGPT conversation data.

    This loader parses ShareGPT conversation data from a local file.
    Public dataset downloads will be handled separately and cached locally.

    The loader filters conversations based on:
    - Minimum conversation length (at least 2 turns required)
    - Sequence length validation for prompt and completion tokens
    - Configurable max prompt length and total sequence length

    Example:
        >>> loader = ShareGPTLoader(user_config, tokenizer, "sharegpt.json")
        >>> dataset = loader.load_dataset()
        >>> conversations = loader.convert_to_conversations(dataset)
        >>> print(f"Loaded {len(conversations)} valid conversations")
    """

    tag = "ShareGPT"

    def __init__(
        self, user_config: UserConfig, tokenizer: Tokenizer, filename: str, **kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.user_config = user_config
        self.output_tokens_mean = self.user_config.input.prompt.output_tokens.mean
        self.turn_count = 0

        self._rng = rng.derive("dataset.loader.sharegpt")

        super().__init__(
            filename=filename, user_config=user_config, tokenizer=tokenizer, **kwargs
        )

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data format.

        ShareGPT entries include a "conversations" list with "value" fields.
        """
        if data is None:
            return False

        conversations = data.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            return False

        first_conversation = conversations[0]
        return isinstance(first_conversation, dict) and "value" in first_conversation

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred dataset sampling strategy for ShareGPT."""
        return DatasetSamplingStrategy.SEQUENTIAL

    def load_dataset(self) -> list[dict[str, Any]]:
        """
        Load the dataset from a local file.

        Returns:
            list[dict[str, Any]]: The loaded dataset.
        """
        with open(self.filename) as f:
            return load_json_str(f.read())

    def is_valid_sequence(
        self,
        prompt_len: int,
        output_len: int,
        min_seq_len: int = 4,
        max_prompt_len: int = 1024,
        max_total_len: int = 2048,
        skip_min_output_len_check: bool = False,
    ) -> bool:
        """Validate a sequence based on prompt and output lengths.

        Adopted from ``vllm/benchmarks/benchmark_dataset.py``.

        Args:
            prompt_len: The length of the prompt.
            output_len: The length of the output.
            min_seq_len: The minimum length of the sequence.
            max_prompt_len: The maximum length of the prompt.
            max_total_len: The maximum length of the total sequence.
            skip_min_output_len_check: Whether to skip the minimum output length check.

        Returns:
            True if the sequence is valid, False otherwise.
        """
        prompt_too_short = prompt_len < min_seq_len
        prompt_too_long = prompt_len > max_prompt_len
        output_too_short = (not skip_min_output_len_check) and (
            output_len < min_seq_len
        )
        combined_too_long = (prompt_len + output_len) > max_total_len

        return not (
            prompt_too_short or output_too_short or prompt_too_long or combined_too_long
        )

    # TODO: distribute this work across the processors
    def convert_to_conversations(
        self, dataset: list[dict[str, Any]]
    ) -> list[Conversation]:
        """
        Convert the loaded dataset to conversations.

        This method will construct `Conversation` objects from the dataset by filtering the dataset
        depending on the sequence lengths and the content sizes.

        Args:
            dataset (dict[str, Any]): The loaded dataset.

        Returns:
            list[Conversation]: The list of conversations.
        """
        self.info(
            f"Validating {self.tag} dataset and constructing conversation dataset"
        )
        filtered_dataset = []
        skipped_entries = 0
        for entry in dataset:
            conversations = entry.get("conversations", [])
            if not conversations or len(conversations) < 2:
                skipped_entries += 1
                continue

            prompt, completion = conversations[0]["value"], conversations[1]["value"]
            prompt_length = len(self.tokenizer.encode(prompt))
            completion_length = len(self.tokenizer.encode(completion))

            if not self.is_valid_sequence(
                prompt_len=prompt_length,
                output_len=completion_length,
                skip_min_output_len_check=self.output_tokens_mean is not None,
            ):
                skipped_entries += 1
                continue

            filtered_dataset.append(
                Conversation(
                    session_id=self.session_id_generator.next(),
                    turns=[
                        Turn(
                            model=self._select_model_name(),
                            texts=[Text(contents=[prompt])],
                            max_tokens=completion_length,
                        )
                    ],
                )
            )

        self.debug(
            lambda: (
                f"Filtered to {len(filtered_dataset)} dataset entries out of {len(dataset)} (skipped {skipped_entries})"
            )
        )
        return filtered_dataset

    def _select_model_name(self) -> str:
        selection_strategy = self.user_config.endpoint.model_selection_strategy
        if selection_strategy == ModelSelectionStrategy.RANDOM:
            return self._rng.choice(self.user_config.endpoint.model_names)
        elif selection_strategy == ModelSelectionStrategy.ROUND_ROBIN:
            model_name = self.user_config.endpoint.model_names[
                self.turn_count % len(self.user_config.endpoint.model_names)
            ]
            self.turn_count += 1
            return model_name
        else:
            raise ValueError(f"Invalid model selection strategy: {selection_strategy}.")
