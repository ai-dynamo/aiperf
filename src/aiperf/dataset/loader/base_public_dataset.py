# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, ClassVar

from aiperf.common.config.user_config import UserConfig
from aiperf.common.environment import Environment
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation, RequestRecord
from aiperf.dataset.loader.base_loader import BaseLoader
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.transports.aiohttp_client import AioHttpClient

AIPERF_DATASET_CACHE_DIR = Path(".cache/aiperf/datasets")


class BasePublicDatasetLoader(BaseLoader):
    """Base class for loading public datasets from remote URLs with caching support.

    This abstract base class provides a common interface and implementation for downloading,
    caching, and loading public datasets. It handles the HTTP download logic, local caching
    to avoid redundant downloads, and provides utilities for validating dataset entries.

    The class follows a two-step process:
    1. Load/download the raw dataset (with automatic caching)
    2. Convert the dataset to AIPerf's standardized Conversation format

    Example:
        >>> class MyDatasetLoader(BasePublicDatasetLoader):
        ...     tag = "MyDataset"
        ...     url = "https://example.com/dataset.json"
        ...     filename = "my_dataset.json"
        ...
        ...     async def load_dataset(self) -> list[Conversation]:
        ...         # Custom dataset loading logic here
        ...         return dataset

        >>> loader = MyDatasetLoader()
        >>> conversations = await loader.load_dataset()
        >>> print(f"Loaded {len(conversations)} conversations")
    """

    tag: ClassVar[str]
    url: ClassVar[str]
    filename: ClassVar[str] = "dataset.json"

    def __init__(self, user_config: UserConfig, **kwargs):
        super().__init__(user_config=user_config, **kwargs)
        self.http_client = AioHttpClient(
            timeout=Environment.DATASET.PUBLIC_DATASET_TIMEOUT
        )
        self.cache_filepath = AIPERF_DATASET_CACHE_DIR / self.filename

    async def load_dataset(self) -> dict[str, Any]:
        """Load the dataset and convert it to AIPerf Conversation format.

        This is the main entry point for dataset loading. Subclasses must implement
        this method to define their specific loading logic.

        Returns:
            dict[str, Any]: A dictionary containing the loaded dataset data.

        Raises:
            NotImplementedError: Always raised as this is an abstract method.
        """
        raise NotImplementedError("Subclasses must implement this method")

    async def convert_to_conversations(
        self, dataset: dict[str, Any]
    ) -> list[Conversation]:
        """
        Convert the loaded dataset into a list of AIPerf Conversation objects.

        This abstract method must be implemented by subclasses to define how the
        specific dataset format should be converted to AIPerf's standardized
        conversation format.

        Args:
            dataset: The parsed dataset object (typically from JSON). The exact
                    structure depends on the specific dataset being loaded.

        Returns:
            list[Conversation]: A list of Conversation objects ready for use
                               in AIPerf benchmarking.

        Raises:
            NotImplementedError: Always raised as this is an abstract method.
        """
        raise NotImplementedError("Subclasses must implement this method")

    async def _load_dataset(self, headers: dict[str, str]) -> str:
        """
        Load the dataset from the local cache or download it from the URL.

        This method first checks if a cached version exists locally. If not, it downloads
        the dataset from the configured URL and saves it to the cache for future use.

        Args:
            headers: Optional HTTP headers to include in the download request.
                    Useful for setting Accept headers or authentication tokens.

        Returns:
            str: The raw dataset content as a string. Subclasses are responsible
                 for parsing this content (e.g., JSON parsing).
        """
        if not self.cache_filepath.exists():
            self.info(f"No local dataset cache found, downloading from {self.url}")
            record: RequestRecord = await self.http_client.get_request(
                self.url, headers=headers
            )
            await self.http_client.close()

            dataset = record.responses[0].text
            self._save_to_local(dataset)
            return dataset

        return self._load_from_local()

    def _save_to_local(self, dataset: str):
        """
        Save the dataset to the local cache.

        Args:
            dataset: The raw dataset payload downloaded from the URL.
        """
        self.info(f"Saving {self.tag} dataset to local cache {self.cache_filepath}")
        try:
            self.cache_filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_filepath, "w") as f:
                f.write(dataset)
        except Exception as e:
            raise DatasetLoaderError(f"Error saving dataset to local cache: {e}") from e

    def _load_from_local(self) -> str:
        """
        Load the raw dataset payload from the local cache.

        Returns:
            str: The raw dataset payload.
        """
        self.info(f"Loading {self.tag} dataset from local cache {self.cache_filepath}")
        try:
            with open(self.cache_filepath) as f:
                return f.read()
        except Exception as e:
            raise DatasetLoaderError(
                f"Error loading dataset from local cache: {e}"
            ) from e

    def is_valid_sequence(
        self,
        prompt_len: int,
        output_len: int,
        min_seq_len: int = 4,
        max_prompt_len: int = 1024,
        max_total_len: int = 2048,
        skip_min_output_len_check: bool = False,
    ) -> bool:
        """
        Validate a sequence based on prompt and output lengths.
        Adopted from `vllm/benchmarks/benchmark_dataset.py`.

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
        # Check for invalid conditions
        prompt_too_short = prompt_len < min_seq_len
        prompt_too_long = prompt_len > max_prompt_len
        output_too_short = (not skip_min_output_len_check) and (
            output_len < min_seq_len
        )
        combined_too_long = (prompt_len + output_len) > max_total_len

        return not (
            prompt_too_short or output_too_short or prompt_too_long or combined_too_long
        )

    def get_recommended_sampling_strategy(self) -> "DatasetSamplingStrategy":
        """Get the recommended sampling strategy for this dataset.

        Subclasses should override this method to return their preferred strategy.

        Returns:
            DatasetSamplingStrategy: The recommended sampling strategy for this loader.
        """
        raise NotImplementedError("Subclasses must implement this method")
