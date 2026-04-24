# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.common import random_generator as rng
from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.utils import load_json_str
from aiperf.dataset.loader.base_public_dataset import BasePublicDatasetLoader
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class ShareGPTLoader(BasePublicDatasetLoader):
    """ShareGPT dataset loader for loading and processing ShareGPT conversation data.

    This loader downloads and processes the ShareGPT dataset from HuggingFace.
    It handles downloading, caching, validation, and conversion of ShareGPT
    conversations into the AIPerf conversation format.

    The loader filters conversations based on:
    - Minimum conversation length (at least 2 turns required)
    - Sequence length validation for prompt and completion tokens
    - Configurable max prompt length and total sequence length

    Example:
        >>> loader = ShareGPTLoader(user_config, tokenizer)
        >>> dataset = await loader.load_dataset()
        >>> conversations = await loader.convert_to_conversations(dataset)
        >>> print(f"Loaded {len(conversations)} valid conversations")
    """

    tag = "ShareGPT"
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    filename = "ShareGPT_V3_unfiltered_cleaned_split.json"

    def __init__(self, run: BenchmarkRun, tokenizer: Tokenizer | None = None, **kwargs):
        if tokenizer is None:
            raise ValueError(
                "ShareGPTLoader requires a tokenizer; ensure the endpoint supports tokenization."
            )
        self.tokenizer = tokenizer
        self.run = run
        self.dataset_config = run.cfg.get_default_dataset()
        # Get output tokens mean from dataset prompts config if available
        prompts = getattr(self.dataset_config, "prompts", None)
        osl = getattr(prompts, "osl", None) if prompts else None
        self.output_tokens_mean = osl.mean if osl else None
        self.turn_count = 0

        self._rng = rng.derive("dataset.loader.sharegpt")

        super().__init__(run=run, tokenizer=tokenizer, **kwargs)

    async def load_dataset(self) -> dict[str, Any]:
        """
        Load the dataset from the local cache or download it from the URL.

        Returns:
            dict[str, Any]: The loaded dataset.
        """
        loaded_dataset = await self._load_dataset(
            headers={"Accept": "application/json"}
        )
        return load_json_str(loaded_dataset)

    # TODO: distribute this work across the processors
    async def convert_to_conversations(
        self, dataset: dict[str, Any]
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
            lambda: f"Filtered to {len(filtered_dataset)} dataset entries out of {len(dataset)} (skipped {skipped_entries})"
        )
        return filtered_dataset

    def _select_model_name(self) -> str:
        selection_strategy = self.run.cfg.models.strategy
        model_names = self.run.cfg.get_model_names()
        if selection_strategy == ModelSelectionStrategy.RANDOM:
            return self._rng.choice(model_names)
        elif selection_strategy == ModelSelectionStrategy.ROUND_ROBIN:
            model_name = model_names[self.turn_count % len(model_names)]
            self.turn_count += 1
            return model_name
        else:
            raise ValueError(f"Invalid model selection strategy: {selection_strategy}.")

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred sampling strategy for this dataset."""
        return DatasetSamplingStrategy.SEQUENTIAL
