# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from abc import abstractmethod
from typing import Any

from datasets import load_dataset as hf_load_dataset
from PIL import Image as PILImage

from aiperf.common.config.user_config import UserConfig
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation, Image
from aiperf.dataset import utils
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
        streaming: bool = False,
        **kwargs,
    ) -> None:
        self.hf_dataset_name = hf_dataset_name
        self.hf_split = hf_split
        self.hf_subset = hf_subset
        self.streaming = streaming
        super().__init__(user_config=user_config, **kwargs)

    async def load_dataset(self) -> dict[str, Any]:
        """Load the dataset from HuggingFace."""
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
            streaming=self.streaming,
        )

    def _pil_to_image(self, pil_image: PILImage.Image) -> Image:
        """Convert a PIL Image to an AIPerf Image with a base64 JPEG data URL."""
        b64 = utils.encode_image(pil_image, "JPEG")
        return Image(name="", contents=[f"data:image/jpeg;base64,{b64}"])

    def _extract_images(self, row: dict[str, Any], image_column: str) -> list[Image]:
        """Extract images from a dataset row column.

        Handles both a single PIL Image and a list of PIL Images,
        returning the first valid image found.
        """
        value = row.get(image_column)
        if isinstance(value, PILImage.Image):
            return [self._pil_to_image(value)]
        if isinstance(value, list):
            pil = next((v for v in value if isinstance(v, PILImage.Image)), None)
            if pil:
                return [self._pil_to_image(pil)]
        return []

    def _max_conversations(self) -> int | None:
        """Return the maximum number of conversations to build from the dataset.

        Returns None for non-streaming datasets.

        For streaming datasets, caps at request_count when set, otherwise
        num_dataset_entries (--num-prompts, default 100), to prevent fetching
        the entire remote dataset in duration-based benchmarks.
        """
        if not self.streaming:
            return None
        request_count = self.user_config.loadgen.request_count
        if request_count is not None:
            return request_count
        return self.user_config.input.conversation.num_dataset_entries

    @abstractmethod
    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]: ...

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL
