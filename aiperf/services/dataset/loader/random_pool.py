# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TypeAlias

from aiperf.common.enums import CustomDatasetType, MediaType
from aiperf.common.factories import CustomDatasetFactory
from aiperf.common.models import Conversation, Turn
from aiperf.services.dataset.loader.mixins import MediaConversionMixin
from aiperf.services.dataset.loader.models import RandomPool

# Type aliases
Filename: TypeAlias = str


@CustomDatasetFactory.register(CustomDatasetType.RANDOM_POOL)
class RandomPoolDatasetLoader(MediaConversionMixin):
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

    def __init__(self, filename: str, num_conversations: int = 1):
        self.filename = filename
        self.num_conversations = num_conversations

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

        for file_path in dir_path.iterdir():
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
            Conversation(session_id=str(uuid.uuid4()))
            for _ in range(self.num_conversations)
        ]

        # F x N (F: num of files, N: num of conversations)
        sampled_dataset: dict[Filename, list[Turn]] = {}

        # Randomly sample (with replacement) from each dataset pool
        for filename, dataset_pool in data.items():
            samples = random.choices(dataset_pool, k=self.num_conversations)
            turns: list[Turn] = []
            for sample in samples:
                media = self.convert_to_media_objects(sample, name=Path(filename).stem)
                turns.append(
                    Turn(
                        texts=media[MediaType.TEXT],
                        images=media[MediaType.IMAGE],
                        audios=media[MediaType.AUDIO],
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
        )
        return merged_turn
