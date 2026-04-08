# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from aiperf.common.config import UserConfig
from aiperf.common.models import Conversation
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.composer.base import BaseDatasetComposer
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, PublicDatasetType


class PublicDatasetComposer(BaseDatasetComposer):
    """Composer for public benchmark datasets loaded from remote sources.

    Instantiates the appropriate public dataset loader using plugin metadata,
    loads the dataset, and finalizes all turns with model name and max_tokens.
    """

    def __init__(self, config: UserConfig, tokenizer: Tokenizer | None):
        super().__init__(config, tokenizer)

    def create_dataset(self) -> list[Conversation]:
        raise NotImplementedError("Use create_dataset_async() for public datasets")

    async def create_dataset_async(self) -> list[Conversation]:
        """Load and finalize a public benchmark dataset.

        Returns:
            list[Conversation]: Finalized conversations ready for benchmarking.
        """
        dataset_type: PublicDatasetType = self.config.input.public_dataset

        LoaderClass = plugins.get_class(PluginType.PUBLIC_DATASET_LOADER, dataset_type)
        self._set_sampling_strategy(dataset_type, LoaderClass)

        loader_kwargs = self._build_loader_kwargs(dataset_type)
        loader = LoaderClass(
            user_config=self.config,
            tokenizer=self.tokenizer,
            **loader_kwargs,
        )

        data = await loader.load_dataset()
        conversations = await loader.convert_to_conversations(data)

        for conversation in conversations:
            for turn in conversation.turns:
                self._finalize_turn(turn)

        self._finalize_conversations(conversations)
        return conversations

    def _set_sampling_strategy(
        self, dataset_type: PublicDatasetType, loader_class: type
    ) -> None:
        """Set the sampling strategy from the loader's preference if not already set by the user.

        Args:
            dataset_type: The public dataset type (for logging).
            loader_class: The loader class to query for its preferred strategy.
        """
        if self.config.input.dataset_sampling_strategy is None:
            preferred = loader_class.get_preferred_sampling_strategy()
            self.config.input.dataset_sampling_strategy = preferred
            self.info(
                f"Using preferred sampling strategy for {dataset_type}: {preferred}"
            )

    def _build_loader_kwargs(self, dataset_type: PublicDatasetType) -> dict[str, Any]:
        """Build loader constructor kwargs from plugin metadata.

        Reads HF-specific fields from the plugin metadata and returns only the
        non-None values so that non-HF loaders (e.g. ShareGPT) receive no
        unexpected kwargs.

        Args:
            dataset_type: The public dataset plugin name.

        Returns:
            dict of kwargs to pass to the loader constructor.
        """
        loader_metadata = plugins.get_public_dataset_loader_metadata(dataset_type)
        kwargs: dict[str, Any] = {}

        if loader_metadata.hf_dataset_name is not None:
            kwargs["hf_dataset_name"] = loader_metadata.hf_dataset_name
            kwargs["hf_split"] = loader_metadata.hf_split
            if loader_metadata.hf_subset is not None:
                kwargs["hf_subset"] = loader_metadata.hf_subset

        if loader_metadata.prompt_column is not None:
            kwargs["prompt_column"] = loader_metadata.prompt_column

        if loader_metadata.image_column is not None:
            kwargs["image_column"] = loader_metadata.image_column

        if loader_metadata.conversation_column is not None:
            kwargs["conversation_column"] = loader_metadata.conversation_column
            kwargs["message_content_key"] = loader_metadata.message_content_key

        if loader_metadata.streaming:
            kwargs["streaming"] = loader_metadata.streaming

        if loader_metadata.category is not None:
            kwargs["category"] = loader_metadata.category

        if loader_metadata.prompt_template is not None:
            kwargs["prompt_template"] = loader_metadata.prompt_template

        return kwargs
