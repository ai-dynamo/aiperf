# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from pathlib import Path
from typing import Any, TypeAlias

from pydantic import ValidationError

from aiperf.common import random_generator as rng
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import CustomDatasetType, DatasetSamplingStrategy, MediaType
from aiperf.common.factories import CustomDatasetFactory
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.mixins import MediaConversionMixin
from aiperf.dataset.loader.models import RandomPool

# Type aliases
Filename: TypeAlias = str


@CustomDatasetFactory.register(CustomDatasetType.RANDOM_POOL)
class RandomPoolDatasetLoader(BaseFileLoader, MediaConversionMixin):
    """A dataset loader that loads data from a single file or a directory.

    Each line in the file represents single-turn conversation data,
    and files create individual pools for random sampling:
      - Single file: All lines form one single pool (to be randomly sampled from)
      - Directory: Each file becomes a separate pool, then pools are randomly sampled
                   and merged into conversations later.

    The random pool custom dataset
      - supports multi-modal data (e.g. text, image, audio)
      - supports client-side batching for each data (e.g. batch size > 1)
      - supports named fields for each modality (e.g. text_field_a, text_field_b, etc.)
      - DOES NOT support multi-turn or its features (e.g. delay, sessions, etc.)

    Example:

    1. Single file
    ```jsonl
    {"text": "Who are you?", "image": "/path/to/image1.png"}
    {"text": "Explain what is the meaning of life.", "image": "/path/to/image2.png"}
    ...
    ```
    The file will form a single pool of text and image data that will be used
    to generate conversations.

    2. Directory

    Directory will be useful if user wants to
      - create multiple pools of different modalities separately (e.g. text, image)
      - specify different field names for the same modality.

    data/queries.jsonl
    ```jsonl
    {"texts": [{"name": "query", "contents": ["Who are you?"]}]}
    {"texts": [{"name": "query", "contents": ["What is the meaning of life?"]}]}
    ...
    ```

    data/passages.jsonl
    ```jsonl
    {"texts": [{"name": "passage", "contents": ["I am a cat."]}]}
    {"texts": [{"name": "passage", "contents": ["I am a dog."]}]}
    ...
    ```

    The loader will create two separate pools for each file: queries and passages.
    Each pool is a text dataset with a different field name (e.g. query, passage),
    and loader will later sample from these two pools to create conversations.
    """

    def __init__(
        self,
        *,
        filename: str,
        user_config: UserConfig,
        num_conversations: int = 1,
        **kwargs,
    ):
        super().__init__(filename=filename, user_config=user_config, **kwargs)
        self._rng = rng.derive("dataset.loader.random_pool")
        self.num_conversations = num_conversations

    @staticmethod
    def _validate_path(path: Path) -> int:
        """Validate all files and directories recursively against the RandomPool model.

        Args:
            path: The path to the file or directory to validate.

        Returns:
            int: Count of files with at least one valid line.

        Raises:
            ValidationError: If any file contains invalid data.
        """
        valid_count = 0

        if path.is_dir():
            # if path is a directory, recursively call this function for each child
            # if any child fails validation, it will exit early with an exception
            for file in path.iterdir():
                valid_count += RandomPoolDatasetLoader._validate_path(file)

        elif path.is_file():
            # if path is a file, validate the first non-empty line against the RandomPool model
            # if the line is valid, increment the valid count and break the loop,
            # otherwise a ValidationError will be raised and the function will exit early
            with open(path) as f:
                for line in f:
                    if not (line := line.strip()):
                        continue
                    RandomPool.model_validate_json(line)
                    valid_count += 1
                    break

        return valid_count

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data format.

        RandomPool is the only loader that supports directory inputs.
        For structural detection, RandomPool format is ambiguous with SingleTurn
        (both have modality fields), so explicit 'type' field or directory path is required.

        Returns:
            True only if filename is a directory with at least one valid file.
            False otherwise (including for regular files without explicit type).
        """

        if data is not None and data.get("type") == CustomDatasetType.RANDOM_POOL:
            try:
                RandomPool.model_validate(data)
                return True
            except ValidationError:
                return False

        if filename is not None:
            try:
                path = Path(filename) if isinstance(filename, str) else filename
                # Only match directories - files are ambiguous with SingleTurn
                if path.is_dir():
                    valid_count = cls._validate_path(path)
                    return valid_count > 0
                return False
            except ValidationError:
                return False

        # RandomPool schema is very similar to SingleTurn, so we can't reliably
        # distinguish without an explicit type field or directory path
        return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred dataset sampling strategy for RandomPool."""
        return DatasetSamplingStrategy.SHUFFLE

    def load_dataset(self) -> dict[Filename, list[RandomPool]]:
        """Load random pool data from a file or directory.

        If filename is a file, reads and parses using the RandomPool model.
        If filename is a directory, reads each file in the directory and merges
        items with different modality names into combined RandomPool objects.

        Returns:
            A dictionary mapping filename to list of RandomPool objects.
        """
        path = Path(self.filename)

        if path.is_file():
            dataset_pool = self._load_dataset_from_file(path)
            return {path.name: dataset_pool}

        return self._load_dataset_from_dir(path)

    def _load_dataset_from_file(self, file_path: Path) -> list[RandomPool]:
        """Load random pool data from a single file.

        Args:
            file_path: The path to the file containing the data.

        Returns:
            A list of RandomPool objects.
        """
        dataset_pool: list[RandomPool] = []

        with open(file_path) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue  # Skip empty lines

                random_pool_data = RandomPool.model_validate_json(line)
                dataset_pool.append(random_pool_data)

        return dataset_pool

    def _load_dataset_from_dir(
        self, dir_path: Path
    ) -> dict[Filename, list[RandomPool]]:
        """Load random pool data from all files in a directory.

        Args:
            dir_path: The path to the directory containing the files.

        Returns:
            A dictionary mapping filename to list of RandomPool objects.
        """
        data: dict[Filename, list[RandomPool]] = defaultdict(list)

        for file_path in sorted(dir_path.iterdir()):
            if file_path.is_file():
                dataset_pool = self._load_dataset_from_file(file_path)
                data[file_path.name].extend(dataset_pool)

        return data

    def convert_to_conversations(
        self, data: dict[Filename, list[RandomPool]]
    ) -> list[Conversation]:
        """Convert random pool data to conversation objects.

        Each RandomPool entry becomes a single-turn conversation with a unique session ID.

        Args:
            data: A dictionary mapping filename to list of RandomPool objects.

        Returns:
            A list of conversations.
        """
        conversations = [
            Conversation(session_id=self.session_id_generator.next())
            for _ in range(self.num_conversations)
        ]

        # F x N (F: num of files, N: num of conversations)
        sampled_dataset: dict[Filename, list[Turn]] = {}

        # Randomly sample (with replacement) from each dataset pool
        for filename, dataset_pool in data.items():
            samples = self._rng.choices(dataset_pool, k=self.num_conversations)
            turns: list[Turn] = []
            for sample in samples:
                media = self.convert_to_media_objects(sample, name=Path(filename).stem)
                turns.append(
                    Turn(
                        texts=media[MediaType.TEXT],
                        images=media[MediaType.IMAGE],
                        audios=media[MediaType.AUDIO],
                        videos=media[MediaType.VIDEO],
                    )
                )
            sampled_dataset[filename] = turns

        # Merge turns for each conversation
        for i, batched_turns in enumerate(zip(*sampled_dataset.values(), strict=False)):
            turn = self._merge_turns(batched_turns)
            conversations[i].turns.append(turn)

        return conversations

    def _merge_turns(self, turns: list[Turn]) -> Turn:
        """Merge turns into a single turn.

        Args:
            turns: A list of turns.

        Returns:
            A single turn.
        """
        merged_turn = Turn(
            texts=[text for turn in turns for text in turn.texts],
            images=[image for turn in turns for image in turn.images],
            audios=[audio for turn in turns for audio in turn.audios],
            videos=[video for turn in turns for video in turn.videos],
        )
        return merged_turn
