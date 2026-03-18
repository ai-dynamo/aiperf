# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from abc import abstractmethod
from typing import Any

from datasets import load_dataset as hf_load_dataset

from aiperf.common.config.user_config import UserConfig
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation
from aiperf.dataset.loader.base_public_dataset import BasePublicDatasetLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


class BaseHFDatasetLoader(BasePublicDatasetLoader):
    """Base class for loading datasets from HuggingFace via the datasets library."""

    def __init__(
        self,
        user_config: UserConfig,
        hf_dataset_name: str,
        hf_split: str = "train",
        hf_subset: str | None = None,
        **kwargs,
    ) -> None:
        self.hf_dataset_name = hf_dataset_name
        self.hf_split = hf_split
        self.hf_subset = hf_subset
        super().__init__(user_config=user_config, **kwargs)

    async def load_dataset(self) -> dict[str, Any]:
        """Load the dataset from HuggingFace"""
        self.info(
            f"Loading HuggingFace dataset '{self.hf_dataset_name}' (split={self.hf_split})"
        )
        try:
            dataset = await asyncio.get_running_loop().run_in_executor(
                None, self._load_hf_dataset
            )
        except Exception as e:
            raise DatasetLoaderError(
                f"Failed to load HuggingFace dataset '{self.hf_dataset_name}': {e}"
            ) from e
        return {"dataset": dataset}

    def _load_hf_dataset(self) -> Any:
        return hf_load_dataset(
            self.hf_dataset_name,
            name=self.hf_subset,
            split=self.hf_split,
            trust_remote_code=False,
        )

    @abstractmethod
    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]: ...

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL
