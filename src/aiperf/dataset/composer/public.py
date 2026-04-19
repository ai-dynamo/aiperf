# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.common.models import Conversation
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.composer.base import BaseDatasetComposer
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class PublicDatasetComposer(BaseDatasetComposer):
    """Composer for public benchmark datasets loaded from remote sources.

    Instantiates the appropriate public dataset loader using plugin metadata,
    loads the dataset, and finalizes all turns with model name and max_tokens.
    """

    def __init__(self, run: BenchmarkRun, tokenizer: Tokenizer | None):
        self.tokenizer = tokenizer
        super().__init__(run, tokenizer)

    def create_dataset(self) -> list[Conversation]:
        raise NotImplementedError("Use create_dataset_async() for public datasets")

    async def create_dataset_async(self) -> list[Conversation]:
        """Load and finalize a public benchmark dataset.

        Returns:
            list[Conversation]: Finalized conversations ready for benchmarking.
        """
        dataset_type = self.dataset_config.name

        LoaderClass = plugins.get_class(PluginType.PUBLIC_DATASET_LOADER, dataset_type)

        loader_kwargs = self._build_loader_kwargs(dataset_type)
        loader = LoaderClass(
            run=self.run,
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

    def _build_loader_kwargs(self, dataset_type: str) -> dict[str, Any]:
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
            cli_subset = getattr(self.dataset_config, "hf_subset", None)
            subset = cli_subset if cli_subset is not None else loader_metadata.hf_subset
            if subset is not None:
                kwargs["hf_subset"] = subset

        if loader_metadata.prompt_column is not None:
            kwargs["prompt_column"] = loader_metadata.prompt_column

        if loader_metadata.image_column is not None:
            kwargs["image_column"] = loader_metadata.image_column

        if loader_metadata.video_column is not None:
            kwargs["video_column"] = loader_metadata.video_column

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
